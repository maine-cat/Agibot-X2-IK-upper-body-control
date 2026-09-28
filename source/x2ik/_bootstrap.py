#!/usr/bin/env python3
"""X2 上肢到点控制的部署入口。

机器人身份与部署目录由现场配置提供，不内置特定机器的参数。
只在已处于 UPPERBODY_REMOTE_SPLIT（URS）的机器人上运行 upper_body 点位，
禁止任何状态切换，也不在本流程启动、停止或重启运控服务。

操作入口见 README.md / QUICK_START.md；标准验收见 CONVERGE_TEST_GUIDE.md；
MoveJ / MoveL / FK / IK 的 Python 契约见 API_INTERFACE.md。
MoveJ 本身没有 TCP 1 mm 闭环；标准流程是 MoveJ 接近后追加闭环 MoveL。
位置精度以反馈关节角经模型 FK 得到的 TCP 误差计，不包含装配误差。

本文件加载 x2ik.conf、查找 ROS Python / aimdk_msgs 并设置子进程环境。
机器人端 DDS / RMW 配置以 DEPLOY_GUIDE.md 为准，不能照搬仿真配置。

当前可用的只读或离线命令：

    ./x2ik.py doctor                        # 本机环境检查，不发送运动
    ./x2ik.py ros state                     # 只读关节反馈和当前 action
    ./x2ik.py api                           # FK / IK 离线自检
    ./x2ik.py verify                        # 数学验证，不连接机器人
    ./x2ik.py frames                        # 离线坐标系说明
    python3 x2_converge_test.py --offline   # 标准测试路径离线预检
    python3 x2_converge_test.py --preflight # 板卡只读现场预检

运动前单独运行预检，操作命令按 QUICK_START.md 执行。
历史仿真、状态切换、直接 HAL 控制和服务管理入口仍保留兼容逻辑，
不属于当前现场流程；本部署包不包含完整仿真 / 绘图 / 抓取扩展脚本。
"""

from __future__ import annotations

import glob
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent

# ROS 安装前缀。换 distro 用 X2_ROS_ROOT 覆盖。
ROS_ROOT = Path(os.environ.get("X2_ROS_ROOT", "/opt/ros/humble"))

# 仿真/运控/消息包的搜索根。默认扫用户的 mujoco_env 树 + 容器习惯路径 /opt/x2_simulation。
SEARCH_ROOTS = [
    Path(os.environ["X2_SEARCH_ROOT"]) if os.environ.get("X2_SEARCH_ROOT")
    else Path.home() / "LINGXI" / "mujoco_env",
    Path("/opt/x2_simulation"),
]

# 相对搜索根的 glob。深度写到 4 层,足以覆盖 `1.1SDK测试版/1.1_X2_SIM/sim_mujoco-x86-*`
# 这种两层包装,也覆盖 `X2_simulation/sim_mujoco` 这种一层的。
_DEPTHS = ("", "*/", "*/*/", "*/*/*/")


# ------------------------------------------------------------------ 路径发现

# ELF header 里的 e_machine。SDK 同时铺了 aarch64 预编译包和 x86 包,目录名看不出来,
# 只能读 .so 的头。挑错架构的后果不是干脆的 ImportError,而是
# "No module named 'aimdk_msgs.aimdk_msgs_s__rosidl_typesupport_c'" ——
# python 找的是 cpython-310-x86_64-linux-gnu.so,而包里躺的是 ...-aarch64-...so。
_ELF_MACHINE = {"x86_64": 0x3E, "aarch64": 0xB7, "armv7l": 0x28, "arm64": 0xB7}


def _elf_machine(path: Path) -> Optional[int]:
    try:
        head = path.open("rb").read(20)
    except OSError:
        return None
    if len(head) < 20 or head[:4] != b"\x7fELF":
        return None
    return int.from_bytes(head[18:20], "little" if head[5] == 1 else "big")


def _is_native(root: Path) -> bool:
    """root 下的 aimdk_msgs 二进制是不是本机架构。判不出来就放行,不误杀。"""
    want = _ELF_MACHINE.get(platform.machine())
    if want is None:
        return True
    for pat in ("lib/libaimdk_msgs*.so", "aimdk_msgs/*.so",
                "*/lib/python3.*/dist-packages/aimdk_msgs/*.so", "*.so"):
        for so in sorted(root.glob(pat))[:3]:
            got = _elf_machine(so)
            if got is not None:
                return got == want
    return True


CONF = Path(os.environ.get("X2IK_CONFIG", str(HERE / "x2ik.conf"))).expanduser()


def __getattr__(name):
    # Source checkout compatibility; the runtime package exports only Robot / HOME.
    if name in ("Robot", "HOME"):
        from .x2_movej import Robot, HOME
        return {"Robot": Robot, "HOME": HOME}[name]
    raise AttributeError(name)


def load_conf() -> List[str]:
    """读 x2ik.conf,把里面的 X2_* 当作环境变量的默认值(环境里已有的优先)。

    自动发现是按 mtime 取最新,机器上同时存在多套 SDK 时结果会随文件时间漂。
    要固定用哪一套就往这个文件里写 `X2_SIM_HOME=...`,比每次 export 可靠,
    也不需要动任何 SDK 目录。删掉此文件即恢复自动发现。
    """
    if not CONF.exists():
        return []
    pinned = []
    for raw in CONF.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = (x.strip() for x in line.split("=", 1))
        if not key.startswith("X2_"):
            continue
        pinned.append(key)
        os.environ.setdefault(key, os.path.expanduser(val))
    return pinned


def _candidates(pattern: str) -> List[Path]:
    """在所有搜索根下按各层深度展开 pattern,返回存在的路径,新的排前面。"""
    hits: List[Path] = []
    for root in SEARCH_ROOTS:
        for depth in _DEPTHS:
            hits.extend(Path(p) for p in glob.glob(str(root / (depth + pattern))))
    uniq = {p.resolve(): p for p in hits if p.exists()}
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _pick(env_var: str, pattern: str, marker: str) -> Optional[Path]:
    """环境变量优先;否则取最新的、含有 marker 的候选目录。"""
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        return p if (p / marker).exists() else None
    for cand in _candidates(pattern):
        if (cand / marker).exists():
            return cand
    return None


def find_sim_home() -> Optional[Path]:
    return _pick("X2_SIM_HOME", "sim_mujoco*", "bin/start_sim.sh")


def find_mc_home() -> Optional[Path]:
    return _pick("X2_MC_HOME", "mc*", "bin/em_run.sh")


def find_sim_model(sim_home: Optional[Path]) -> Path:
    """仿真真正加载的 MJCF。找不到就退回本目录的 x2_ultra.xml。

    两者只差 model name / meshdir / pelvis 初始位姿,运动学与动力学一致
    (x2_frames.cross_check_mjcf 已验证到 1 µm),所以退回是安全的 —— 但
    `sim` 子命令的意义就是跑仿真那一份,doctor 会把实际用的是哪个打出来。
    """
    override = os.environ.get("X2_SIM_MODEL")
    if override and Path(override).exists():
        return Path(override)
    if sim_home:
        found = sorted(sim_home.glob("configuration/robot/*/model_info/x2.xml"))
        if found:
            return found[0]
    return HERE / "x2_ultra.xml"


def find_msgs_prefix() -> Optional[Path]:
    """aimdk_msgs 的安装前缀(含 lib/ 与 pythonX.Y 的 dist-packages)。

    SDK 自带的 `aimdk/src/aimdk_msgs` 在 stock humble 上编不过(interface 目录嵌套
    超过一层,`rosidl_generate_interfaces` 解析 idl tuple 会越界),而 aarch64 预编译
    包在 x86 上用不了。所以这里优先用已经编好的 x86 前缀 —— 逐条比对过,本项目用到的
    msg/srv 定义与 1.1 SDK 源码逐字节相同。哪天 aimdk_msgs 编出来了,
    install 前缀会自动排在前面(按 mtime 排序)。
    """
    override = os.environ.get("X2_AIMDK_MSGS")
    if override:
        p = Path(override)
        return p if _msgs_pythonpath(p) else None
    for pattern in ("aimdk/install/aimdk_msgs", "aimdk_msgs"):
        for cand in _candidates(pattern):
            if _msgs_pythonpath(cand) and _is_native(cand):
                return cand
    return None


def foreign_msgs_prefixes() -> List[Path]:
    """找得到但架构不对的 aimdk_msgs —— doctor 用来解释"为什么没选那一份"。"""
    out = []
    for pattern in ("aimdk/install/aimdk_msgs", "aimdk_msgs"):
        for cand in _candidates(pattern):
            if _msgs_pythonpath(cand) and not _is_native(cand) and cand not in out:
                out.append(cand)
    return out


def _msgs_pythonpath(prefix: Path) -> Optional[Path]:
    """消息包里 aimdk_msgs 这个 python 包所在的目录。布局在不同打包方式下不一样。"""
    for sub in ("local/lib/python3.*/dist-packages", "lib/python3.*/dist-packages",
                "lib/python3.*/site-packages"):
        for hit in sorted(prefix.glob(sub)):
            if (hit / "aimdk_msgs").is_dir():
                return hit
    return None


# ------------------------------------------------------------------ 解释器发现

def _probe(py: str, expr: str) -> bool:
    try:
        return subprocess.run([py, "-c", expr], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _probe_sourced(py: str, expr: str) -> bool:
    """在 source 过 ROS setup.bash 的环境里探测。

    探测必须和真正执行(run_with_ros)用同一个环境,否则会出现
    "doctor 说没有解释器能 import rclpy,但命令其实跑得起来"的错位 ——
    反过来说,没 source ROS 的干净 shell 里也不该因此判死。
    """
    setup = ROS_ROOT / "setup.bash"
    if not setup.exists():
        return _probe(py, expr)
    script = f'. {shlex.quote(str(setup))} >/dev/null 2>&1; exec "$@"'
    try:
        return subprocess.run(["bash", "-c", script, "x2ik", py, "-c", expr],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ros_python() -> Optional[str]:
    """能 import rclpy 的解释器。rclpy 的 C 扩展只对 ROS 编译时那个小版本有效。"""
    override = os.environ.get("X2_ROS_PYTHON")
    cands = [override] if override else []
    for dist in sorted(ROS_ROOT.glob("*/lib/python3.*/dist-packages")) + \
                sorted(ROS_ROOT.glob("lib/python3.*/dist-packages")):
        cands.append(f"/usr/bin/{dist.parent.name}")     # e.g. /usr/bin/python3.10
    cands += ["/usr/bin/python3", sys.executable]
    for py in dict.fromkeys(c for c in cands if c):
        if Path(py).exists() and _probe_sourced(py, "import rclpy"):
            return py
    return None


def mujoco_python() -> Optional[str]:
    """能 import mujoco 的解释器。"""
    override = os.environ.get("X2_MUJOCO_PYTHON")
    cands = [override, sys.executable, "python3", "/usr/bin/python3"]
    for py in dict.fromkeys(c for c in cands if c):
        if _probe(py, "import mujoco, numpy"):
            return py
    return None


def plot_python() -> Optional[str]:
    """能 import matplotlib 的解释器。和 numpy_python 分开挑是有原因的:
    本机 conda 那个装了 mujoco/numpy 但**没有** matplotlib,系统 python3 反过来 ——
    合成一个"通用 numpy 解释器"必然挑错一边。"""
    for py in dict.fromkeys(c for c in (os.environ.get("X2_PLOT_PYTHON"),
                                        "/usr/bin/python3", sys.executable,
                                        os.environ.get("X2_MUJOCO_PYTHON"),
                                        "python3") if c):
        if _probe(py, "import numpy, matplotlib"):
            return py
    return None


def numpy_python() -> Optional[str]:
    """纯 numpy 就够的命令(verify / frames)用,优先复用 mujoco 那个解释器。"""
    for py in dict.fromkeys(c for c in (os.environ.get("X2_MUJOCO_PYTHON"),
                                        sys.executable, "python3",
                                        "/usr/bin/python3") if c):
        if _probe(py, "import numpy"):
            return py
    return None


# ------------------------------------------------------------------ 环境组装

def sim_rmw_with_source(sim_home: Optional[Path]) -> tuple:
    """要用哪个 RMW,以及**这个值是哪来的**。

    返回 (rmw, 来源说明)。来源必须一起返回:真机部署时机器上没有仿真安装,
    值其实来自 `X2_RMW` 或写死的兜底,而 doctor 以前一律打
    "(从 start_sim.sh 读出)" —— 在部署机上这句是假的,会让人以为
    "已经和真机对齐了",而真机用哪个 RMW 恰恰是必须现场确认的一项。
    """
    if os.environ.get("X2_RMW"):
        return os.environ["X2_RMW"], "来自 X2_RMW / x2ik.conf"
    if sim_home:
        script = sim_home / "bin" / "start_sim.sh"
        try:
            for line in script.read_text(errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("export RMW_IMPLEMENTATION="):
                    return (line.split("=", 1)[1].strip().strip('"\''),
                            "从 start_sim.sh 读出")
        except OSError:
            pass
    return "rmw_fastrtps_cpp", "兜底默认值 —— 真机上必须确认,不一致则回调一个不进"


def sim_rmw(sim_home: Optional[Path]) -> str:
    """客户端和对端必须是同一个 RMW,否则话题看得见、数据收不到。"""
    return sim_rmw_with_source(sim_home)[0]


def app_binary(home: Optional[Path]) -> Optional[str]:
    """安装目录里那个主进程可执行文件名。

    不能写死:旧 SDK 的运控叫 `mc_app_main`,1.1 改成了 `aima-mc-app-main` ——
    写死一个会让 doctor 对另一套 SDK 永远报"未运行"。
    """
    if not home:
        return None
    for pat in ("bin/aima-*-app*", "bin/mc_app_main", "bin/*_app_main", "bin/*-app"):
        for f in sorted((home).glob(pat)):
            if f.is_file() and os.access(f, os.X_OK):
                return f.name
    return None


def _running_from(home: Path, name: str) -> Optional[int]:
    """在 home/bin 里跑着 name 的进程 pid。找不到返回 None。

    三个坑,所以既不用 `pgrep -f 名字` 也不用 `comm`:
      * 启动脚本 `cd` 到 bin/ 后用 `./aima-sim-app` 起,cmdline 里是相对路径,
        按绝对路径 pgrep 匹配不上;
      * 这两个程序会改自己的进程名 —— comm 实际是 `soc_sim` / `mc_main`,不是二进制名;
      * `pgrep -f 名字` 会匹配任何提到该名字的命令行(包括查询者自己),假阳性。
    用 /proc/<pid>/cwd 定位是哪一套 SDK,再用 cmdline 确认是哪个程序,两者都对才算。
    """
    want = (home / "bin").resolve()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.joinpath("cwd").resolve() != want:
                continue
            argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
            if argv and Path(argv[0].decode("utf-8", "replace")).name == name:
                return int(proc.name)
        except (OSError, ValueError):
            continue
    return None


def _prune(value: str) -> str:
    """剔掉两类条目:已经不存在的(bashrc 里残留的 /opt/x2_simulation/*),
    以及架构对不上的 aimdk_msgs —— `source aimdk/install/setup.bash` 很容易
    把 aarch64 预编译包挂进 PYTHONPATH/AMENT_PREFIX_PATH,它会抢在正确的包前面。
    """
    keep = []
    for entry in value.split(os.pathsep):
        if not entry or not Path(entry).exists():
            continue
        if "aimdk" in entry and not _is_native(Path(entry)):
            continue
        keep.append(entry)
    return os.pathsep.join(dict.fromkeys(keep))


def _prepend(env: Dict[str, str], key: str, path: str) -> None:
    cur = _prune(env.get(key, ""))
    env[key] = path + (os.pathsep + cur if cur else "")


def ros_env(sim_home: Optional[Path], msgs: Optional[Path]) -> Dict[str, str]:
    """给 ROS 子进程用的环境增量(在 source ROS setup.bash *之前* 设好)。"""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "AMENT_PREFIX_PATH"):
        if env.get(key):
            env[key] = _prune(env[key])
    if msgs:
        pypath = _msgs_pythonpath(msgs)
        if pypath:
            _prepend(env, "PYTHONPATH", str(pypath))
        if (msgs / "lib").is_dir():
            _prepend(env, "LD_LIBRARY_PATH", str(msgs / "lib"))
        _prepend(env, "AMENT_PREFIX_PATH", str(msgs))
    mc = find_mc_home()
    if mc:
        # x2_sim_ros 要读 mc 的 action_ruler.yaml —— 可选的 action 随 SDK 版本变,
        # 硬编码一张表一定会过期(1.1 就没有 UPPERBODY_REMOTE_SPLIT)。
        env["X2_MC_HOME"] = str(mc)
    env["RMW_IMPLEMENTATION"] = sim_rmw(sim_home)
    env.setdefault("ROS_LOCALHOST_ONLY", "0")
    # BLAS 线程数必须在 **import numpy 之前** 定,进程里再改没用 —— 所以只能
    # 在这儿(exec 前)设,和 PYTHONPATH/LD_LIBRARY_PATH 同理。
    #
    # 本项目的矩阵全是 7x7 / 6x7 这种小家伙,多线程 BLAS 一点速度都买不到,
    # 只会把所有核拉起来空转。实测(双臂 800 帧 track,本机 18 核):
    #     不限线程  墙钟中位 0.70 ms   CPU 合计 1159 ms
    #     限 1 线程 墙钟中位 0.69 ms   CPU 合计  596 ms   <- 墙钟不变,CPU 减半
    # 板卡上这一半 CPU 直接换成功耗、发热,以及被 mc/HAL 抢走的核。
    # setdefault:外面显式设了就听外面的,方便对拍。
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(_v, "1")
    return env


def run_with_ros(argv: Sequence[str], env: Dict[str, str]) -> int:
    """source ROS setup.bash 之后再 exec argv。source 是 shell 操作,只能借 bash 完成。"""
    setup = ROS_ROOT / "setup.bash"
    if not setup.exists():
        print(f"[err] 找不到 {setup};用 X2_ROS_ROOT 指到正确的 ROS 前缀", file=sys.stderr)
        return 2
    script = f'. {shlex.quote(str(setup))} >/dev/null 2>&1; exec "$@"'
    return subprocess.call(["bash", "-c", script, "x2ik", *argv], env=env)


# ------------------------------------------------------------------ 子命令

def cmd_env(ctx) -> int:
    """打印 export 行,给 `eval "$(./x2ik.py env)"` 用。

    只**前置** aimdk_msgs 那几条,不整条覆盖:setup.bash 自己会往这三个变量里塞
    ROS 的 dist-packages,把算好的整值 export 回去会把它们冲掉 ——
    症状是 `ros2` 命令报 `PackageNotFoundError: ros2cli`。
    """
    msgs, sim_home = ctx["msgs"], ctx["sim_home"]
    print(f". {shlex.quote(str(ROS_ROOT / 'setup.bash'))}")
    adds = []
    if msgs:
        pypath = _msgs_pythonpath(msgs)
        if pypath:
            adds.append(("PYTHONPATH", str(pypath)))
        if (msgs / "lib").is_dir():
            adds.append(("LD_LIBRARY_PATH", str(msgs / "lib")))
        adds.append(("AMENT_PREFIX_PATH", str(msgs)))
    for key, val in adds:
        print(f'export {key}={shlex.quote(val)}${{{key}:+:${key}}}')
    print(f"export RMW_IMPLEMENTATION={shlex.quote(sim_rmw(sim_home))}")
    print('export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"')
    print(f"export X2_SIM_MODEL={shlex.quote(str(ctx['model']))}")
    return 0


def cmd_doctor(ctx) -> int:
    sim_home, mc_home, msgs, model = (ctx["sim_home"], ctx["mc_home"],
                                      ctx["msgs"], ctx["model"])
    bad = 0

    def line(ok: bool, label: str, detail: str = "") -> None:
        nonlocal bad
        if not ok:
            bad += 1
        print(f"  {'OK ' if ok else 'FAIL'}  {label:<22} {detail}")

    if ctx.get("pinned"):
        print(f"[0] {CONF.name} 固定了: " + "  ".join(ctx["pinned"]))
    print("[1] 仿真安装")
    line(sim_home is not None, "sim_mujoco", str(sim_home or "未找到(设 X2_SIM_HOME)"))
    line(mc_home is not None, "mc(运控)", str(mc_home or "未找到(设 X2_MC_HOME)"))
    # 不用 Path.is_relative_to —— 它是 3.9+。机器本体 PC2(Orin NX)如果是
    # Ubuntu 20.04 / Foxy,系统 python 是 3.8,doctor 会在这一行崩掉。
    from_sim = sim_home is not None and str(model).startswith(str(sim_home) + os.sep)
    line(model.exists(), "MJCF 模型",
         f"{model}{'' if from_sim else '   ← 退回本目录副本,不是仿真那一份'}")

    print("[2] 解释器")
    mj, rp = ctx["mujoco_python"], ctx["ros_python"]
    line(mj is not None, "mujoco python", mj or "没有解释器能 import mujoco(pip install mujoco)")
    line(rp is not None, "ros python", rp or "没有解释器能 import rclpy")

    print("[3] aimdk_msgs")
    line(msgs is not None, f"前缀({platform.machine()})",
         str(msgs or "未找到本机架构的包(设 X2_AIMDK_MSGS)"))
    for other in foreign_msgs_prefixes():
        print(f"  SKIP  {'架构不符,已跳过':<22} {other}")
    if msgs and rp:
        env = ros_env(sim_home, msgs)
        code = run_with_ros([rp, "-c", "import aimdk_msgs; from aimdk_msgs.msg import "
                                       "UpperBodyCommandArray, JointCommandArray, JointStateArray; "
                                       "from aimdk_msgs.srv import SetMcAction, GetMcAction"], env)
        line(code == 0, "msg/srv 导入", "本项目用到的类型齐全" if code == 0 else "导入失败")

    print("[4] RMW 一致性")
    want, why = sim_rmw_with_source(sim_home)
    have = os.environ.get("RMW_IMPLEMENTATION", "(未设)")
    line(True, "将使用", f"{want}   ({why})")
    # shell 里不一致不算故障 —— 子进程一律被强制改成仿真那个。只有裸跑 ros2 CLI
    # 或手工 python 才会踩到,所以提示而不计入失败。
    print(f"  {'OK ' if have == want else 'WARN'}  {'当前 shell':<22} " + have +
          ("" if have == want else "   ← 不一致;x2ik 起的子进程会强制改成上面那个,"
                                   "但你自己敲 ros2 topic list 会看不到话题"))

    print("[5] 进程")
    # 按**绝对路径**查,不按名字。两个理由:机器上同时有多套 SDK,按名字分不清在跑的是
    # 哪一套;而且 `pgrep -f <名字>` 会匹配到任何提到这个名字的命令行(包括查询者自己),
    # 是实打实的假阳性来源。
    for home, hint in ((sim_home, "./x2ik.py up sim"), (mc_home, "./x2ik.py up mc")):
        name = app_binary(home)
        if not name:
            print(f"  --    {'(未知)':<22} 找不到安装目录")
            continue
        pid = _running_from(home, name)
        up = pid is not None
        print(f"  {'RUN ' if up else '--  '}  {name:<22} "
              f"{f'pid {pid}' if up else '未运行 → ' + hint}")

    print(f"\n{'全部通过' if bad == 0 else f'{bad} 项需要处理'}")
    return 0 if bad == 0 else 1


def cmd_up(target: str, rest: Sequence[str], ctx) -> int:
    """前台起仿真或运控。两边都要用仿真那一份 RMW —— em_run.sh 自己不设,
    裸跑会继承 shell 里的 cyclonedds,和仿真的 fastrtps 互相看不见。"""
    home = ctx["sim_home"] if target == "sim" else ctx["mc_home"]
    script = "bin/start_sim.sh" if target == "sim" else "bin/em_run.sh"
    if home is None:
        print(f"[err] 找不到 {target} 的安装目录", file=sys.stderr)
        return 2
    env = ros_env(ctx["sim_home"], ctx["msgs"])
    print(f"[x2ik] {home / script}  (RMW={env['RMW_IMPLEMENTATION']})")
    return subprocess.call(["bash", str(home / script), *rest],
                           cwd=str(home / "bin"), env=env)


def _run_offline(script: str, rest: Sequence[str], ctx, need_mujoco: bool) -> int:
    py = ctx["mujoco_python"] if need_mujoco else ctx["numpy_python"]
    if py is None:
        want = "mujoco" if need_mujoco else "numpy"
        print(f"[err] 没有解释器能 import {want}", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["X2_SIM_MODEL"] = str(ctx["model"])
    return subprocess.call([py, str(HERE / script), *rest], cwd=str(HERE), env=env)


def cmd_plot(rest: Sequence[str], ctx) -> int:
    """离线 3D 可视化。不碰 ROS,所以不用 source、也不要求仿真在跑。"""
    py = ctx["plot_python"]
    if py is None:
        print("[err] 没有解释器能 import matplotlib。\n"
              "      装一个:  /usr/bin/python3 -m pip install --user matplotlib\n"
              "      或指定:  X2_PLOT_PYTHON=/path/to/python ./x2ik.py plot ...",
              file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["X2_SIM_MODEL"] = str(ctx["model"])
    return subprocess.call([py, str(HERE / "x2_viz.py"), *rest], cwd=str(HERE), env=env)


def cmd_ros(rest: Sequence[str], ctx) -> int:
    # `ros calibrate` 是纯文件 IO(读写 calibration/<SN>.json),不连仿真也不连真机。
    # 拿系统 python 直接跑,别让 rclpy/aimdk_msgs 缺失把它一起挡在门外 ——
    # 开发机上常常没配消息包,但一样需要看/改标定文件。
    if rest and rest[0] == "calibrate":
        env = dict(os.environ)
        env["X2_SIM_MODEL"] = str(ctx["model"])
        py = ctx["ros_python"] or sys.executable
        return subprocess.call([py, str(HERE / "x2_sim_ros.py"), *rest],
                               cwd=str(HERE), env=env)
    if ctx["ros_python"] is None:
        print("[err] 没有解释器能 import rclpy,先跑 ./x2ik.py doctor", file=sys.stderr)
        return 2
    if ctx["msgs"] is None:
        print("[err] 找不到 aimdk_msgs,先跑 ./x2ik.py doctor", file=sys.stderr)
        return 2
    env = ros_env(ctx["sim_home"], ctx["msgs"])
    env["X2_SIM_MODEL"] = str(ctx["model"])
    return run_with_ros([ctx["ros_python"], str(HERE / "x2_sim_ros.py"), *rest], env)


def cmd_raw(rest: Sequence[str], ctx) -> int:
    """跑 x2_upper_raw.py —— 解耦排障工具,不含本项目任何算法。

    工具本身是零项目依赖的独立脚本(只 import rclpy + aimdk_msgs),现场也可以
    自己 source 环境后直接 `python3 x2_upper_raw.py` 跑。这里只是替它把
    PYTHONPATH / LD_LIBRARY_PATH / RMW 摆好 —— 这些必须在进程**启动前**设,
    脚本内部补不了(消息包的 typesupport 是 .so,靠 LD_LIBRARY_PATH 找)。
    """
    if ctx["ros_python"] is None:
        print("[err] 没有解释器能 import rclpy,先跑 ./x2ik.py doctor", file=sys.stderr)
        return 2
    if ctx["msgs"] is None:
        print("[err] 找不到 aimdk_msgs,先跑 ./x2ik.py doctor", file=sys.stderr)
        return 2
    env = ros_env(ctx["sim_home"], ctx["msgs"])
    return run_with_ros([ctx["ros_python"], str(HERE / "x2_upper_raw.py"), *rest], env)


# ------------------------------------------------------------------ 入口

def build_context() -> Dict[str, object]:
    pinned = load_conf()
    sim_home = find_sim_home()
    return {
        "pinned": pinned,
        "sim_home": sim_home,
        "mc_home": find_mc_home(),
        "msgs": find_msgs_prefix(),
        "model": find_sim_model(sim_home),
        "ros_python": ros_python(),
        "mujoco_python": mujoco_python(),
        "numpy_python": numpy_python(),
        "plot_python": plot_python(),
    }


USAGE = """x2ik.py —— X2 上肢到点控制部署入口

机器人身份与部署目录由现场配置提供。
仅在已经处于 URS 时使用 upper_body；禁止任何状态切换和运控服务操作。
每次运动前单独做只读预检，勿同时启动多个控制或预检程序。

两个主要入口（均保留）：
  ./x2ik.py mdi              MDI 终端操作前端，供人输入关节角 / TCP 位姿
  x2_api.X2Arm               MoveJ 等开放接口，供其他模块集成
  MDI 启动和退出的动作语义见 MDI_GUIDE.md；预览可用：
  ./x2ik.py mdi --dry --no-home-first --relax 0 --duration 8 --settle 2
                              只读反馈并预览输入，不发布保持或运动命令
  不带 --dry 的 MDI 会持续发送；MDI 与其他 API 控制程序不能并行运行。

文档入口：
  README.md                  项目状态与文档导航
  QUICK_START.md             今天怎样开始机器人端到点测试
  MDI_GUIDE.md               前端操作：输入目标、预览、切换手臂与退出
  CONVERGE_TEST_GUIDE.md     标准点位、闭环参数、结果与达标判据
  API_INTERFACE.md           MoveJ / MoveL / FK / IK 的 Python 调用契约
  DEPLOY_GUIDE.md            板卡环境、机器 SN、部署和版本校验

只读诊断（在目标板卡终端运行）：
  ./x2ik.py doctor            检查当前终端的解释器、消息包和环境
  ./x2ik.py env               打印 ROS 环境变量，不修改当前终端
  ./x2ik.py ros state         读取关节反馈及当前 mc action，不切状态
  ./x2ik.py ros imu           读取 IMU 和重力估计
  ./x2ik.py ros calibrate show
                              显示按 X2_ROBOT_SN 选择的配置文件
  python3 x2_converge_test.py --preflight
                              只读检查 URS、发布独占、完整反馈及 pelvis IMU
  python3 examples/movej_point_test.py --preflight
                              只读检查 Python API 连接，不发送运动

离线检查（不需要 ROS 或机器人）：
  ./x2ik.py api               四接口的 FK / IK 数学自检
  ./x2ik.py verify            数学验证
  ./x2ik.py frames            坐标系说明
  ./x2ik.py jump <csv>        分析已有轨迹 CSV
  python3 x2_converge_test.py --offline
                              检查标准点位与短 MoveL 路径
  python3 examples/movej_point_test.py
                              默认只计算 MoveJ 示例目标
  python3 -m unittest discover -s tests -v
                              离线软件测试，不连接硬件

到点测试：按 QUICK_START.md 的只读预检 → 小规模验证 → 完整对照操作。
MoveJ = X2Arm.move_j(q)，做关节空间插值，本身不包含 TCP 1 mm 闭环。
闭环 MoveL = X2Arm.move_l(pos, rpy, converge=8)，须检查 converged。
标准测试器采用 MoveJ 接近 + 闭环 MoveL；精度是反馈关节角 FK 口径，
不包含装配误差，方法返回后也不后台持续保位。

坐标与单位：torso 系（X 前 / Y 左 / Z 上），位置 m。
CLI 姿态用 deg，Python API 姿态 / 关节角用 rad；详见 docs/COORDINATES.md。

历史扩展和控制入口仍保留兼容逻辑，--help 中存在不等于本轮获准使用。
状态切换、直接 HAL 控制、服务操作及缺失的仿真 / 绘图 / 抓取扩展，
都不属于当前板卡到点测试流程。完整仿真边界见 SIM_GUIDE.md。
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, rest = args[0], args[1:]
    # 透传型子命令走手写分发而不是 argparse 的 REMAINDER —— REMAINDER 碰到
    # 以 `-` 开头的第一个 token(`ros --mode ... home`)会当成未知选项报错。
    ctx = build_context()

    if cmd == "doctor":
        return cmd_doctor(ctx)
    if cmd == "env":
        return cmd_env(ctx)
    if cmd == "up":
        if not rest or rest[0] not in ("sim", "mc"):
            print("用法: ./x2ik.py up sim|mc", file=sys.stderr)
            return 2
        return cmd_up(rest[0], rest[1:], ctx)
    if cmd == "verify":
        return _run_offline("verify_x2_arm.py", rest, ctx, need_mujoco=False)
    if cmd == "frames":
        return _run_offline("x2_frames.py", rest, ctx, need_mujoco=False)
    if cmd == "grasp":
        return _run_offline("x2_grasp_adapter.py", rest, ctx, need_mujoco=False)
    if cmd == "sim":
        return _run_offline("x2_mujoco_arm.py", rest, ctx, need_mujoco=True)
    if cmd == "jump":
        # 只吃 CSV + numpy,不需要 ROS —— 所以走 offline 那条路,
        # 在没配 rclpy 的办公机上也能分析真机录下来的轨迹。
        return _run_offline("x2_record.py", rest, ctx, need_mujoco=False)
    if cmd == "api":
        return _run_offline("x2_api.py", rest, ctx, need_mujoco=False)
    if cmd == "plot":
        return cmd_plot(rest, ctx)
    if cmd == "ros":
        return cmd_ros(rest, ctx)
    if cmd == "raw":
        return cmd_raw(rest, ctx)
    # mdi / viz 是 ros 的子命令,但这两个用得最勤,给个顶层直达的别名。
    if cmd in ("mdi", "viz"):
        return cmd_ros([cmd, *rest], ctx)

    print(f"未知子命令 {cmd!r}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
