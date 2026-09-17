"""Exercise the shipped __main__ template without building or importing ROS."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import MagicMock, Mock, patch

import x2_movej
import x2_tcp


class RuntimeTcpEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="x2-runtime-tcp-")
        self.addCleanup(self.temp.cleanup)
        self.tool_file = Path(self.temp.name) / "custom tool.json"
        tools = x2_tcp.load_tcp_tools("gripper")
        for tool in tools.values():
            tool.pop("mode")
        self.tool_file.write_text(json.dumps(dict(schema_version=1, units="m", tools=tools)))
        name = "_x2_runtime_entry_test"
        self.package = ModuleType(name)
        self.package.__path__ = []
        self.package.HOME = x2_movej.HOME
        self.package.Robot = MagicMock()
        self.boot = ModuleType(name + "._bootstrap")
        for method in ("load_conf", "ros_python", "find_msgs_prefix", "ros_env",
                       "find_sim_home", "run_with_ros"):
            setattr(self.boot, method, Mock())
        self.bridge = ModuleType(name + ".x2_mdi_bridge")
        self.bridge.main = Mock(return_value=0)
        self.terminal = ModuleType(name + ".x2_sim_ros")
        self.terminal.main = Mock(return_value=0)
        modules = {name: self.package, name + ".x2_tcp": x2_tcp,
                   self.boot.__name__: self.boot, self.bridge.__name__: self.bridge,
                   self.terminal.__name__: self.terminal}
        replacement = patch.dict(sys.modules, modules)
        replacement.start()
        self.addCleanup(replacement.stop)
        argv = patch.object(sys, "argv", ["runtime"])
        argv.start()
        self.addCleanup(argv.stop)
        path = Path(__file__).resolve().parents[1] / "packaging/runtime_main.py"
        spec = importlib.util.spec_from_file_location(name + ".__main__", path)
        self.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.entry)

    def test_invalid_custom_fails_before_bootstrap_or_robot(self):
        for args in (["mdi", "--stdio", "--tcp-mode", "custom"],
                     ["movej", "--execute", "--tcp-mode", "none",
                      "--tcp-file", str(self.tool_file)],
                     ["mdi", "--tcp-mode", "custom", "--tcp-file", "/missing/tool.json"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.entry.main(args)
                self.assertEqual(error.exception.code, 2)
        self.boot.load_conf.assert_not_called()
        self.boot.ros_python.assert_not_called()
        self.package.Robot.assert_not_called()

    def test_desktop_stdio_forwards_all_modes_and_robot_side_file(self):
        for mode in x2_tcp.TCP_MODES:
            options = ["--tcp-mode", mode]
            if mode == "custom":
                options += ["--tcp-file", str(self.tool_file)]
            self.assertEqual(self.entry.main(["mdi", "--stdio", "--demo", *options]), 0)
            self.bridge.main.assert_called_with(["--demo", *options])
        self.boot.ros_python.assert_not_called()

    def test_terminal_forwards_selection_and_keeps_urs_only_start(self):
        self.assertEqual(self.entry.main(["--ros-ready", "mdi", "--dry", "--tcp-mode", "custom",
                                          "--tcp-file", str(self.tool_file)]), 0)
        self.terminal.main.assert_called_once_with()
        forwarded = sys.argv
        self.assertEqual(forwarded[forwarded.index("--tcp-file") + 1], str(self.tool_file))
        self.assertEqual(forwarded[forwarded.index("--tcp-mode") + 1], "custom")
        self.assertIn("--no-home-first", forwarded)
        self.assertIn("--dry", forwarded)

    def test_offline_movej_prints_selected_mode_without_connecting(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(self.entry.main(["movej", "--tcp-mode", "hand"]), 0)
        self.assertIn("TCP mode: hand", output.getvalue())
        self.package.Robot.assert_not_called()
        self.boot.ros_python.assert_not_called()

    def test_online_movej_passes_selection_without_changing_joint_target(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.entry.main(["--ros-ready", "movej", "--execute", "--side", "both",
                                              "--tcp-mode", "custom", "--tcp-file", str(self.tool_file)]), 0)
        self.package.Robot.assert_called_once_with(tcp_mode="custom", tcp_file=str(self.tool_file))
        robot = self.package.Robot.return_value.__enter__.return_value
        self.assertEqual(robot.moveJ.call_args.kwargs["q_left"].tolist(), x2_movej.HOME.tolist())
        self.assertEqual(robot.moveJ.call_args.kwargs["q_right"].tolist(), x2_movej.HOME.tolist())


if __name__ == "__main__":
    unittest.main()
