"""点位测试器安全门禁：离线，不导入 rclpy，不连接机器人。"""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_converge_test as runner
import x2_sim_ros as ros


class Client:
    def __init__(self):
        self.models = {side: ros.ArmModel(side) for side in ("left", "right")}
        self.measured = {side: ros.HOME_Q.copy() for side in self.models}
        self._last_sent = {side: ros.HOME_Q.copy() for side in self.models}
        self.state_names = [f"{side}_{suffix}" for side in self.models for suffix in ros.ARM_JOINT_SUFFIX]
        self.state_count = 1
        self.imu_count = {"pelvis": 1}
        self.imu_has_orientation = {"pelvis": True}
        self.pubs = 1
        self.node = SimpleNamespace(count_publishers=lambda topic: self.pubs,
                                    count_subscribers=lambda topic: 1)
        self.sent = []

    def q(self, side):
        return self.measured[side].copy()

    def dq(self, side):
        return np.zeros(7)

    def send(self, left, right, dl=None, dr=None):
        self.sent.append((np.array(left), np.array(right)))
        self._last_sent = {"left": np.array(left), "right": np.array(right)}


class RunnerTests(unittest.TestCase):
    def test_converge_tuning_cannot_expand_safety_limits(self):
        good = runner.parser().parse_args(["--converge", "8", "--converge-step", ".25"])
        runner.validate_args(good)
        for option, value in (("--converge-step", ".6"), ("--converge-total", "4"),
                              ("--converge-tol", "2"), ("--converge-step", "nan")):
            with self.assertRaises(ValueError):
                runner.validate_args(runner.parser().parse_args([option, value]))

    def test_numeric_buffer_keeps_original_frame_schema_and_copies_joint_data(self):
        buffer = runner.FrameBuffer(1)
        q = np.arange(7, dtype=float)
        buffer.append(2., "approach", 123, [q] * 6, guard_s=.001, send_s=.002)
        q[:] = -1
        self.assertEqual(buffer[0]["q_sent_left"], list(range(7)))
        self.assertEqual(buffer[0]["state_count"], 123)
        self.assertEqual(buffer[0]["phase"], "approach")
        self.assertEqual(buffer[0]["send_call_s"], .002)
        self.assertEqual(runner.json_safe(buffer), [buffer[0]])
        self.assertFalse(runner.gc.is_tracked(buffer.data))
        with self.assertRaises(RuntimeError):
            buffer.append(3., "approach", 124, [q] * 6)

    def test_gc_timing_uses_bounded_storage_and_restores_callback(self):
        timing = runner.GcTiming(10., capacity=1)
        old_callbacks = list(runner.gc.callbacks)
        timing.start()
        self.assertIn(timing.callback, runner.gc.callbacks)
        timing.stop()
        with mock.patch.object(runner.time, "monotonic", return_value=11.):
            timing._on_gc("start", {"generation": 2})
        with mock.patch.object(runner.time, "monotonic", return_value=11.1):
            timing._on_gc("stop", {"generation": 2, "collected": 4})
            timing._on_gc("stop", {"generation": 2})
        self.assertEqual(runner.gc.callbacks, old_callbacks)
        self.assertEqual(timing.size, 1)
        self.assertEqual(timing.dropped, 1)
        self.assertAlmostEqual(timing.report()["events"][0][1], .1)

    def test_default_plan_has_five_valid_repeats_and_one_warmup_per_group_point(self):
        tasks = runner.plan(runner.parser().parse_args([]))
        self.assertEqual(len(tasks), 36)
        for group in ("off", "on"):
            for name in ("home", "lateral", "forward"):
                rows = [r for r in tasks if r["group"] == group and r["point"] == name]
                self.assertEqual(sum(r["discarded"] for r in rows), 1)
                self.assertEqual(sum(not r["discarded"] for r in rows), 5)
        self.assertTrue(all(a["point"] != b["point"] for a, b in zip(tasks, tasks[1:])))

    def test_unknown_or_non_urs_action_rejected(self):
        for action in (None, "", "JD", "PASSIVE_DEFAULT", "URS-ish", "WHOLE_BODY_TELEOP", "URSW(32)"):
            self.assertFalse(runner.is_urs(action))
        for action in ("URS", "UPPERBODY_REMOTE_SPLIT", "UPPERBODY_REMOTE_SPLIT(19)"):
            self.assertTrue(runner.is_urs(action))
        with self.assertRaises(RuntimeError):
            runner.forbid_action("URS")

    def test_missing_or_blank_expected_sn_rejected_before_calibration_load(self):
        for expected in (None, "", " \t\n"):
            with self.subTest(expected=expected):
                env = {"X2_ROBOT_SN": "TEST_ROBOT"}
                if expected is not None:
                    env["X2_TEST_EXPECTED_SN"] = expected
                loader = mock.Mock()
                with mock.patch.dict(runner.os.environ, env, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "X2_TEST_EXPECTED_SN"):
                        runner.calibration(SimpleNamespace(load_calibration=loader))
                loader.assert_not_called()

    def test_wrong_or_missing_robot_sn_rejected_before_calibration_load(self):
        for sn in (None, "", " \t", "OTHER_ROBOT"):
            with self.subTest(sn=sn):
                env = {"X2_TEST_EXPECTED_SN": "TEST_ROBOT"}
                if sn is not None:
                    env["X2_ROBOT_SN"] = sn
                loader = mock.Mock()
                with mock.patch.dict(runner.os.environ, env, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "X2_ROBOT_SN"):
                        runner.calibration(SimpleNamespace(load_calibration=loader))
                loader.assert_not_called()

    def test_matching_explicit_robot_identity_loads_matching_calibration(self):
        data = dict(sn="TEST_ROBOT", stiffness=40, bias_limit_deg=12, gravity_source="pelvis")
        loader = mock.Mock(return_value=data)
        with mock.patch.dict(runner.os.environ, {
                "X2_TEST_EXPECTED_SN": "TEST_ROBOT", "X2_ROBOT_SN": "TEST_ROBOT"}, clear=True):
            self.assertIs(runner.calibration(SimpleNamespace(load_calibration=loader)), data)
        loader.assert_called_once_with("TEST_ROBOT")

    def test_calibration_must_exist_and_match_verified_identity(self):
        for data in (None, {}, dict(sn="OTHER_ROBOT", stiffness=40,
                                   bias_limit_deg=12, gravity_source="pelvis")):
            with self.subTest(calibration=data):
                loader = mock.Mock(return_value=data)
                with mock.patch.dict(runner.os.environ, {
                        "X2_TEST_EXPECTED_SN": "TEST_ROBOT", "X2_ROBOT_SN": "TEST_ROBOT"}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "标定文件"):
                        runner.calibration(SimpleNamespace(load_calibration=loader))
                loader.assert_called_once_with("TEST_ROBOT")

    def test_incomplete_latest_frame_and_nan_rejected(self):
        cli = Client()
        cli.state_names.pop()
        with self.assertRaises(RuntimeError):
            runner.check_feedback(cli, ros)
        cli = Client()
        cli.measured["left"][0] = math.nan
        with self.assertRaises(RuntimeError):
            runner.check_feedback(cli, ros)

    def test_monitor_freezes_other_arm_input_despite_feedback_drift(self):
        cli = Client()
        monitor = runner.MotionMonitor(cli, ros, "right")
        expected = monitor.fixed_other.copy()
        cli.measured["left"] += .05
        monitor.send(np.ones(7), ros.HOME_Q)
        np.testing.assert_array_equal(cli.sent[0][0], expected)
        self.assertEqual(monitor.rows[0]["state_count"], 1)

    def test_monitor_stops_before_publishing_after_expiration_or_contention(self):
        cli = Client()
        with mock.patch.object(runner.time, "monotonic", return_value=10.):
            monitor = runner.MotionMonitor(cli, ros, "right")
            monitor.send(ros.HOME_Q, ros.HOME_Q)
        with mock.patch.object(runner.time, "monotonic", return_value=10.21):
            with self.assertRaises(RuntimeError):
                monitor.send(ros.HOME_Q, ros.HOME_Q)
        self.assertEqual(len(cli.sent), 1)
        cli = Client()
        cli.pubs = 2
        monitor = runner.MotionMonitor(cli, ros, "right")
        with self.assertRaises(RuntimeError):
            monitor.send(ros.HOME_Q, ros.HOME_Q)
        self.assertEqual(cli.sent, [])

    def test_monitor_rejects_frozen_feedback_even_with_continuous_commands(self):
        cli = Client()
        with mock.patch.object(runner.time, "monotonic", return_value=10.):
            monitor = runner.MotionMonitor(cli, ros, "right")
            monitor.send(ros.HOME_Q, ros.HOME_Q)
        for stamp in (10.05, 10.10, 10.15):
            with mock.patch.object(runner.time, "monotonic", return_value=stamp):
                monitor.send(ros.HOME_Q, ros.HOME_Q)
        with mock.patch.object(runner.time, "monotonic", return_value=10.21):
            with self.assertRaises(RuntimeError):
                monitor.send(ros.HOME_Q, ros.HOME_Q)
        self.assertEqual(len(cli.sent), 4)

    def test_all_nominal_point_tails_pass_real_ik(self):
        for side in ("left", "right"):
            model = ros.ArmModel(side)
            ik = ros.SrsArmIK(model)
            for point in runner.make_points(ros, model, side).values():
                check = runner.validate_path(ros, model, ik, point["q"], point, 2.)
                self.assertLess(check["position_error_m"], .0005)

    def test_invalid_motion_result_stops_sequence(self):
        for result in ({"stale": True}, {"trajectory_valid": False}, {"ik_fails": 1},
                       {"converge_reason": "diverging"}, {"pos_err": math.nan}):
            self.assertIsNotNone(runner.result_error(result))
        self.assertIsNone(runner.result_error({"converge_reason": "max_iterations", "pos_err": .002}))

    def test_nan_report_is_valid_json_and_summary_discards_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "data.json"
            runner.save_report(output, {"error": "invalid_feedback", "raw": np.array([math.nan, math.inf, 1.])})
            self.assertEqual(json.loads(output.read_text())["raw"], [None, None, 1.])
        rows = [dict(group="on", point="home", discarded=True, result=dict(pos_err=.1)),
                dict(group="on", point="home", discarded=False, result=dict(pos_err=.0005)),
                dict(group="on", point="home", discarded=False, result=dict(pos_err=.0007))]
        stats = runner.summary(rows)["on/home"]
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["median_mm"], .6)
        self.assertAlmostEqual(stats["range_mm"], .2)

    def test_existing_output_is_never_overwritten_or_client_created(self):
        args = runner.parser().parse_args(["--preflight"])
        with tempfile.TemporaryDirectory() as tmp:
            args.output = Path(tmp) / "old.json"
            args.output.write_text("old raw data")
            with mock.patch.object(runner, "calibration", return_value={}), \
                    mock.patch.object(ros, "X2ArmClient") as client:
                with self.assertRaises(RuntimeError):
                    runner.run(args, ros)
            client.assert_not_called()
            self.assertEqual(args.output.read_text(), "old raw data")


if __name__ == "__main__":
    unittest.main()
