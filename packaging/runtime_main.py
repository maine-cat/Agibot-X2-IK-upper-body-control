"""Only the MDI and MoveJ entry points are public."""
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python3 -m x2ik")
    parser.add_argument("--ros-ready", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="entry", required=True)
    mdi = sub.add_parser("mdi", help="MDI terminal or SSH desktop session; URS only",
                        description="Requires existing URS mode. Switch using the robot's own controls first; MDI cannot switch modes.")
    mdi.add_argument("--stdio", action="store_true", help="Desktop JSONL session over SSH")
    mdi.add_argument("--demo", action="store_true", help="Offline desktop protocol demo")
    mdi.add_argument("--dry", action="store_true", help="Terminal preview, no motion")
    mdi.add_argument("--side", choices=["left", "right"], default="right")
    mdi.add_argument("--duration", type=float, default=8.)
    mdi.add_argument("--settle", type=float, default=2.)
    move = sub.add_parser("movej", help="MoveJ example, default offline")
    move.add_argument("--side", choices=["left", "right", "both"], default="right")
    move.add_argument("--q", nargs=7, type=float, help="Joint target, radians; default HOME")
    move.add_argument("--duration", type=float, default=8.)
    move.add_argument("--settle", type=float, default=2.)
    move.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    from . import _bootstrap as boot
    config = Path(os.environ.get("X2IK_CONFIG", Path.cwd() / "x2ik.conf")).expanduser().resolve()
    boot.CONF = config
    boot.load_conf()
    os.environ.setdefault("X2IK_CONFIG", str(config))
    if (config.parent / "calibration").is_dir():
        os.environ.setdefault("X2IK_CALIB_DIR", str(config.parent / "calibration"))
    online = (args.entry == "mdi" and not args.demo) or (args.entry == "movej" and args.execute)
    if online and not args.ros_ready:
        py, msgs = boot.ros_python(), boot.find_msgs_prefix()
        if not py or not msgs:
            parser.error("ROS Python / aimdk_msgs unavailable; check X2IK_CONFIG and ROS installation")
        env = boot.ros_env(boot.find_sim_home(), msgs)
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        return boot.run_with_ros([py, "-m", "x2ik", "--ros-ready", *argv], env)
    if args.entry == "mdi":
        if args.stdio:
            from . import x2_mdi_bridge
            return x2_mdi_bridge.main(["--demo"] if args.demo else [])
        if args.demo:
            parser.error("--demo requires --stdio")
        from . import x2_sim_ros
        sys.argv = ["x2_sim_ros", "--mode", "upper_body", "mdi", "--no-home-first", "--relax", "0",
                    "--side", args.side, "--duration", str(args.duration), "--settle", str(args.settle)]
        if args.dry:
            sys.argv.append("--dry")
        return x2_sim_ros.main()
    from . import Robot, HOME
    target = HOME.copy() if args.q is None else args.q
    if not args.execute:
        print("Offline MoveJ target (rad):", list(target), "side:", args.side)
        print("Add --execute on the robot to send motion.")
        return 0
    with Robot() as robot:
        options = dict(duration=args.duration, settle=args.settle)
        print(robot.moveJ(q_left=target, q_right=target, **options) if args.side == "both"
              else robot.moveJ(target, side=args.side, **options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
