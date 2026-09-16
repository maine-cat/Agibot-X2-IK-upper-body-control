"""MoveJ 示例的模式选择与失败停点测试，全程使用假的运动 API。"""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import x2_api

spec = importlib.util.spec_from_file_location("movej_point_test", ROOT / "examples/movej_point_test.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class FakeArm:
    def __init__(self, side, connect=False, failure=None):
        self.side, self.connect, self.failure = side, connect, failure
        self.model = x2_api.ArmModel(side)
        self.cli = SimpleNamespace(q=lambda side: x2_api.HOME.copy())
        self.connection_config = {"robot_sn": "fake"}
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def fk(self, q):
        pos, rot = self.model.forward_kinematics(q)
        return pos, x2_api.matrix_to_rpy(rot)

    def joints(self):
        return x2_api.HOME.copy()

    def _joint_result(self, q, side):
        return dict(q=q.copy(), err=np.zeros(7), err_max=0.,
                    stale=self.failure in ("joint", "joint_" + side))

    def move_j(self, *args, **kwargs):
        raise AssertionError("示例应使用显式左右入口")

    def R_move_J(self, q, **kwargs):
        self.calls.append(("RJ", q.copy(), kwargs))
        return self._joint_result(q, "right")

    def L_move_J(self, q, **kwargs):
        self.calls.append(("LJ", q.copy(), kwargs))
        return self._joint_result(q, "left")

    def move_j_both(self, *, q_left, q_right, **kwargs):
        self.calls.append(("BJ", {"left": q_left.copy(), "right": q_right.copy()}, kwargs))
        return {"left": self._joint_result(q_left, "left"),
                "right": self._joint_result(q_right, "right")}

    def move_l(self, pos, rpy, **kwargs):
        self.calls.append(("L", pos.copy(), kwargs))
        return dict(pos_err=.003 if self.failure == "cartesian" else .0005,
                    converged=self.failure != "cartesian", converge_reason="tolerance",
                    converge_iterations=2, stale=False, trajectory_valid=True)


class ExampleTests(unittest.TestCase):
    def invoke(self, argv, failure=None):
        made = []

        def factory(side, connect=False):
            arm = FakeArm(side, connect, failure)
            made.append(arm)
            return arm

        with mock.patch.object(x2_api, "X2Arm", side_effect=factory), \
                mock.patch.object(example, "bootstrap", return_value=None) as bootstrap, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = example.main(argv)
        made[0].created_instances = made
        return status, made[0], bootstrap

    def test_default_offline_never_bootstraps_or_moves(self):
        status, arm, bootstrap = self.invoke([])
        self.assertEqual(status, 0)
        self.assertFalse(arm.connect)
        self.assertEqual(arm.calls, [])
        self.assertTrue(arm.closed)
        bootstrap.assert_not_called()

    def test_preflight_connects_but_never_moves_and_blocks_actions(self):
        status, arm, bootstrap = self.invoke(["--preflight"])
        self.assertEqual(status, 0)
        self.assertTrue(arm.connect)
        self.assertEqual(arm.calls, [])
        bootstrap.assert_called_once()
        with self.assertRaises(RuntimeError):
            arm.cli.set_action("URS")

    def test_execute_calls_only_single_arm_movej_in_the_planned_order(self):
        status, arm, _ = self.invoke(["--execute", "--side", "left"])
        self.assertEqual(status, 0)
        self.assertEqual([call[0] for call in arm.calls], ["LJ", "LJ", "LJ"])
        self.assertEqual(arm.side, "left")
        np.testing.assert_allclose([call[1][0] for call in arm.calls], [.4, math.radians(55), .4])
        self.assertTrue(all(call[2] == {"duration": 8., "settle": 2.} for call in arm.calls))

    def test_optional_converge_forwards_fixed_safety_limits(self):
        status, arm, _ = self.invoke(["--execute", "--converge", "8"])
        self.assertEqual(status, 0)
        self.assertEqual([call[0] for call in arm.calls], ["RJ", "L", "RJ", "L", "RJ", "L"])
        for _, _, kwargs in arm.calls[1::2]:
            self.assertEqual(kwargs, dict(duration=2., settle=2., converge=8, converge_tol=.001,
                                           converge_step=math.radians(.5), converge_total=math.radians(3.)))

    def test_stale_movej_stops_without_followup_or_recovery_motion(self):
        status, arm, _ = self.invoke(["--execute", "--converge", "8"], failure="joint")
        self.assertEqual(status, 1)
        self.assertEqual([call[0] for call in arm.calls], ["RJ"])
        self.assertTrue(arm.closed)

    def test_failed_convergence_stops_before_next_point(self):
        status, arm, _ = self.invoke(["--execute", "--converge", "8"], failure="cartesian")
        self.assertEqual(status, 1)
        self.assertEqual([call[0] for call in arm.calls], ["RJ", "L"])
        self.assertTrue(arm.closed)

    def test_both_offline_and_preflight_do_not_move_and_use_one_online_instance(self):
        for mode in ([], ["--preflight"]):
            with self.subTest(mode=mode):
                status, arm, bootstrap = self.invoke(["--side", "both", *mode])
                self.assertEqual(status, 0)
                self.assertEqual(arm.side, "right")
                self.assertEqual(sum(a.connect for a in arm.created_instances), bool(mode))
                self.assertTrue(all(not a.calls and a.closed for a in arm.created_instances))
                if not mode:
                    bootstrap.assert_not_called()

    def test_both_execute_uses_one_synchronous_call_per_point(self):
        status, arm, _ = self.invoke(["--execute", "--side", "both"])
        self.assertEqual(status, 0)
        self.assertEqual(sum(a.connect for a in arm.created_instances), 1)
        self.assertEqual([call[0] for call in arm.calls], ["BJ", "BJ", "BJ"])
        for side in ("left", "right"):
            np.testing.assert_allclose([call[1][side][0] for call in arm.calls],
                                       [.4, math.radians(55), .4])
        self.assertTrue(all(call[2] == {"duration": 8., "settle": 2.} for call in arm.calls))
        self.assertTrue(all(not a.calls for a in arm.created_instances if not a.connect))

    def test_either_arm_failure_stops_both_sequence_before_next_point(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                status, arm, _ = self.invoke(["--execute", "--side", "both"],
                                             failure="joint_" + side)
                self.assertEqual(status, 1)
                self.assertEqual([call[0] for call in arm.calls], ["BJ"])
                self.assertTrue(all(a.closed for a in arm.created_instances))

    def test_both_converge_is_rejected_before_bootstrap_or_instance_creation(self):
        with mock.patch.object(example, "bootstrap") as bootstrap, \
                mock.patch.object(x2_api, "X2Arm") as factory, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                example.main(["--execute", "--side", "both", "--converge", "8"])
            self.assertEqual(result.exception.code, 2)
            bootstrap.assert_not_called()
            factory.assert_not_called()

    def test_invalid_timing_fails_before_ros_bootstrap(self):
        for option, value in (("--duration", "7.9"), ("--settle", "1"), ("--duration", "nan")):
            with mock.patch.object(example, "bootstrap") as bootstrap, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    example.main(["--execute", option, value])
                self.assertEqual(result.exception.code, 2)
                bootstrap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
