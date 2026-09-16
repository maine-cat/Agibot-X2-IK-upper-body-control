"""公共 MoveJ 封装的选择歧义和失败清理测试，无 ROS / 真机连接。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import x2_movej

spec = importlib.util.spec_from_file_location("movej_minimal", ROOT / "examples/movej_minimal.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class MoveJFacadeTests(unittest.TestCase):
    def setUp(self):
        self.factory_patch = mock.patch.object(x2_movej, "_X2Arm", autospec=True)
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)
        self.arm = self.factory.return_value
        self.robot = x2_movej.Robot(robot_sn="test-sn", verbose=False)

    def test_connection_does_not_move_or_request_state(self):
        self.factory.assert_called_once_with("right", robot_sn="test-sn", verbose=False)
        self.assertEqual(self.arm.method_calls, [])

    def test_explicit_left_then_right_use_same_connection_and_return_feedback(self):
        target = x2_movej.HOME.copy()
        for side, method in (("left", self.arm.L_move_J), ("right", self.arm.R_move_J)):
            result = self.robot.moveJ(target, side=side, duration=9.0, settle=2.5)
            self.assertIs(result, method.return_value)
            method.assert_called_once_with(target, duration=9.0, settle=2.5)
        self.factory.assert_called_once()
        self.arm.move_j_both.assert_not_called()

    def test_default_side_is_right_and_home_uses_underlying_default(self):
        self.robot.moveJ()
        self.arm.R_move_J.assert_called_once_with(None, duration=8.0, settle=2.0)
        self.arm.L_move_J.assert_not_called()

    def test_both_is_one_synchronous_operation(self):
        left, right = x2_movej.HOME.copy(), x2_movej.HOME.copy()
        result = self.robot.moveJ(q_left=left, q_right=right)
        self.arm.move_j_both.assert_called_once_with(q_left=left, q_right=right,
                                                    duration=8.0, settle=2.0)
        self.assertIs(result, self.arm.move_j_both.return_value)
        self.arm.L_move_J.assert_not_called()
        self.arm.R_move_J.assert_not_called()

    def test_ambiguous_or_incomplete_target_is_rejected_before_any_motion(self):
        target = x2_movej.HOME.copy()
        invalid = [dict(side="both", q=target), dict(side="LEFT", q=target),
                   dict(q_left=target), dict(q_right=target),
                   dict(q=target, q_left=target, q_right=target),
                   dict(side="left", q_left=target, q_right=target),
                   dict(side="right", q_left=target, q_right=target)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.robot.moveJ(**kwargs)
        self.assertEqual(self.arm.method_calls, [])

    def test_motion_failure_closes_without_retry_or_recovery(self):
        failure = RuntimeError("feedback lost")
        self.arm.R_move_J.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            with self.robot:
                self.robot.moveJ(x2_movej.HOME)
                self.robot.moveJ(x2_movej.HOME, side="left")
        self.assertIs(raised.exception, failure)
        self.assertEqual([name for name, _, _ in self.arm.method_calls], ["R_move_J", "close"])

    def test_public_facade_has_no_old_kinematics_or_state_methods(self):
        for name in ("move_l", "fk", "ik", "set_action", "home"):
            self.assertFalse(hasattr(self.robot, name))


class MinimalExampleTests(unittest.TestCase):
    def test_default_preview_does_not_connect(self):
        public = SimpleNamespace(HOME=x2_movej.HOME, Robot=mock.Mock())
        with mock.patch.dict(sys.modules, {"x2ik": public}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(example.main([]), 0)
        public.Robot.assert_not_called()

    def test_explicit_execution_moves_once_and_closes(self):
        public = SimpleNamespace(HOME=x2_movej.HOME, Robot=mock.MagicMock())
        robot = public.Robot.return_value.__enter__.return_value
        robot.moveJ.return_value = {"err_max": 0.001}
        with mock.patch.dict(sys.modules, {"x2ik": public}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(example.main(["--execute"]), 0)
        public.Robot.assert_called_once_with()
        robot.moveJ.assert_called_once()
        kwargs = robot.moveJ.call_args.kwargs
        self.assertEqual(kwargs, dict(side="right", duration=8.0, settle=2.0))
        public.Robot.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
