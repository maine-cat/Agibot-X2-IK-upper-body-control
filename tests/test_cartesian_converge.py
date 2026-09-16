"""离线闭环回归：fake 时钟/编码器，不导入 ROS、不连接机器人。"""
import argparse
import contextlib
import io
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_sim_ros as ros


class Clock:
    def __init__(self):
        self.now = 10.0

    def time(self):
        return self.now


class Model:
    def __init__(self):
        self.q_min = np.full(7, -10.0)
        self.q_max = np.full(7, 10.0)

    def forward_kinematics(self, q):
        return np.asarray(q[:3]).copy(), np.eye(3)

    def clamp(self, q):
        return np.clip(q, self.q_min, self.q_max)


class IK:
    def sew_angle(self, q):
        return 0.0

    def project_to_workspace(self, pos, rot):
        return pos, False

    def track(self, pos, rot, **kwargs):
        q = np.zeros(7)
        q[:3] = pos
        return SimpleNamespace(q=q, psi=0.0, elbow_branch=1,
                               shoulder_branch=1, wrist_branch=1)


class Client:
    rate = 50.0

    def __init__(self, clock, side="right", bias=0.004):
        self.clock, self.side = clock, side
        self.models = {s: Model() for s in ("left", "right")}
        self.iks = {s: IK() for s in ("left", "right")}
        self.measured = {s: np.zeros(7) for s in self.models}
        self.other = "left" if side == "right" else "right"
        self.measured[self.other][:] = 0.12
        self.state_count = 1
        self.state_names = [f"{s}_{suffix}" for s in ("left", "right")
                            for suffix in ros.ARM_JOINT_SUFFIX]
        self.node = object()
        self.rclpy = SimpleNamespace(spin_once=self.spin_once)
        self.recorder = None
        self.sent = []
        self.fresh_calls = 0
        self.feedback_after = -math.inf
        self.feedback_enabled = True
        self.response = lambda q: q - np.array([bias, 0, 0, 0, 0, 0, 0])

    def q(self, side):
        return self.measured[side].copy()

    def send(self, left, right, dl=None, dr=None):
        self.sent.append((self.clock.now, np.array(left), np.array(right),
                          None if dl is None else np.array(dl),
                          None if dr is None else np.array(dr)))

    def spin_once(self, node, timeout_sec):
        self.clock.now += max(timeout_sec, 1e-8)
        if self.feedback_enabled and self.clock.now >= self.feedback_after:
            self.state_count += 1
            if self.sent:
                index = 1 if self.side == "left" else 2
                self.measured[self.side] = self.response(self.sent[-1][index].copy())

    def fresh_state(self, timeout=0.5):
        self.fresh_calls += 1
        self.spin_once(self.node, 0.01)
        return self.feedback_enabled


class ConvergeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.client = Client(self.clock)
        self.time_patch = mock.patch.object(ros.time, "time", self.clock.time)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)

    def run_move(self, **kwargs):
        options = dict(duration=0.04, settle=0.04)
        options.update(kwargs)
        return ros.goto_cartesian(self.client, self.client.side,
                                  np.array([0.02, 0, 0]), np.eye(3), **options)

    def assert_cadence(self, start=0):
        gaps = np.diff([frame[0] for frame in self.client.sent[start:]])
        self.assertTrue(np.all(gaps >= 0.02 - 1e-7), gaps)
        self.assertTrue(np.all(gaps < 0.2), gaps)

    def test_default_keeps_original_send_trace_fresh_call_and_result_schema(self):
        result = self.run_move()
        self.assertEqual(self.client.fresh_calls, 1)
        self.assertEqual(len(self.client.sent), 4)
        self.assertEqual(result["sent_frames"], 2)
        for index, (timestamp, left, right, dl, dr) in enumerate(self.client.sent):
            self.assertAlmostEqual(timestamp, 10.0 + 0.02 * index)
            np.testing.assert_allclose(left, np.full(7, 0.12))
            np.testing.assert_allclose(right, [0.01 if index == 0 else 0.02, 0, 0, 0, 0, 0, 0])
            if index < 2:
                np.testing.assert_array_equal(dl, np.zeros(7))
                np.testing.assert_allclose(dr, [0.5, 0, 0, 0, 0, 0, 0])
            else:
                self.assertIsNone(dl)
                self.assertIsNone(dr)
        self.assertAlmostEqual(result["pos_err"], 0.004)
        self.assertEqual(set(result), {
            "target", "reached", "stale", "pos_err", "pos_err_xyz", "rot_err",
            "q_track_err", "ik_fails", "clipped", "q_cmd", "q_meas", "step_violations",
            "step_rejects", "branch_rejects", "implicit_fallbacks", "peak_raw_dq",
            "peak_accepted_dq", "max_joint_step_rad", "trajectory_valid", "aborted",
            "branch", "sent_frames"})
        with mock.patch.object(ros, "_fresh_hold", side_effect=AssertionError("opt-in only")):
            self.run_move(converge=0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ros._print_convergence(result)
        self.assertEqual(output.getvalue(), "")

    def test_static_bias_converges_and_keeps_original_target_separate(self):
        result = self.run_move(converge=3)
        self.assertTrue(result["converged"])
        self.assertEqual(result["converge_reason"], "tolerance")
        self.assertEqual(result["converge_iterations"], 1)
        np.testing.assert_allclose(result["converge_hist"], [0.004, 0.0], atol=1e-15)
        self.assertAlmostEqual(result["q_cmd"][0], 0.02)
        self.assertAlmostEqual(result["q_hold"][0], 0.024)
        self.assertEqual(self.client.fresh_calls, 0)
        self.assertTrue(np.all(np.diff(result["converge_state_counts"]) > 0))
        self.assert_cadence()

    def test_left_side_preserves_other_arm(self):
        self.client = Client(self.clock, side="left")
        result = self.run_move(converge=3)
        self.assertTrue(result["converged"])
        for frame in self.client.sent:
            np.testing.assert_array_equal(frame[2], np.full(7, 0.12))

    def test_nonfinite_feedback_stops_without_correction(self):
        self.client.response = lambda q: np.full(7, np.nan)
        result = self.run_move(converge=3)
        self.assertEqual(result["converge_reason"], "invalid_feedback")
        self.assertFalse(result["converged"])
        self.assertEqual(result["converge_hist"], [])
        self.assertEqual(result["converge_iterations"], 0)

    def test_initial_incomplete_or_nonfinite_either_arm_rejects_before_send(self):
        for side in ("left", "right"):
            for invalid in ("missing", "nan"):
                with self.subTest(side=side, invalid=invalid):
                    self.client = Client(self.clock)
                    if invalid == "missing":
                        self.client.state_names.remove(f"{side}_{ros.ARM_JOINT_SUFFIX[0]}")
                    else:
                        self.client.measured[side][0] = np.nan
                    with self.assertRaisesRegex(ValueError, "完整双臂"):
                        self.run_move(converge=3)
                    self.assertEqual(self.client.sent, [])

    def test_new_incomplete_frame_cannot_reuse_old_joint_values(self):
        full_response = self.client.response

        def incomplete_response(q):
            self.client.state_names = [f"right_{suffix}" for suffix in ros.ARM_JOINT_SUFFIX]
            return full_response(q)

        self.client.response = incomplete_response
        result = self.run_move(converge=3)
        self.assertGreater(self.client.state_count, 1)
        self.assertEqual(result["converge_reason"], "invalid_feedback")
        self.assertFalse(result["converged"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["converge_hist"], [])
        self.assertEqual(result["converge_iterations"], 0)
        np.testing.assert_array_equal(self.client.q("left"), np.full(7, 0.12))

    def test_default_does_not_require_complete_frame(self):
        self.client.state_names = []
        result = self.run_move()
        self.assertEqual(len(self.client.sent), 4)
        self.assertEqual(self.client.fresh_calls, 1)
        self.assertAlmostEqual(result["pos_err"], 0.004)

    def test_stale_feedback_times_out_while_sending_without_correction(self):
        self.client.feedback_enabled = False
        result = self.run_move(converge=3)
        self.assertEqual(result["converge_reason"], "stale_feedback")
        self.assertFalse(result["converged"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["converge_hist"], [])
        self.assertEqual(result["converge_iterations"], 0)
        self.assertGreaterEqual(len(self.client.sent), 29)
        self.assert_cadence()

    def test_delayed_feedback_does_not_cause_next_settle_to_burst(self):
        self.client.feedback_after = 10.31
        result = self.run_move(converge=3)
        self.assertTrue(result["converged"])
        self.assertGreater(len(self.client.sent), 18)
        self.assert_cadence()

    def test_divergence_stops_even_at_last_allowed_iteration(self):
        def inverted_response(q):
            q[0] = 0.016 - 2 * (q[0] - 0.02)
            return q
        self.client.response = inverted_response
        result = self.run_move(converge=1)
        self.assertEqual(result["converge_reason"], "diverging")
        self.assertEqual(result["converge_iterations"], 1)
        np.testing.assert_allclose(result["converge_hist"], [0.004, 0.012])

    def test_step_and_total_limits_stop_before_exceeding_total(self):
        self.client.response = lambda q: np.zeros(7)
        result = self.run_move(converge=5, converge_step=0.005, converge_total=0.006)
        self.assertEqual(result["converge_reason"], "total_limit")
        self.assertEqual(result["converge_iterations"], 1)
        self.assertGreater(result["converge_step_clips"], 0)
        self.assertAlmostEqual(result["q_correction"][0], 0.005)
        self.assertLessEqual(max(frame[2][0] for frame in self.client.sent), 0.025 + 1e-12)

    def test_joint_limit_rejects_candidate_instead_of_sending_clamped_step(self):
        self.client.models["right"].q_max[0] = 0.023
        result = self.run_move(converge=3)
        self.assertEqual(result["converge_reason"], "joint_limit")
        self.assertEqual(result["converge_iterations"], 0)
        self.assertLessEqual(max(frame[2][0] for frame in self.client.sent), 0.02 + 1e-12)

    def test_max_rounds_and_early_tolerance(self):
        self.client.response = lambda q: np.zeros(7)
        result = self.run_move(converge=2)
        self.assertEqual(result["converge_reason"], "max_iterations")
        self.assertEqual(len(result["converge_hist"]), 3)
        self.assertEqual(result["converge_iterations"], 2)
        self.client = Client(self.clock, bias=0.0005)
        result = self.run_move(converge=3)
        self.assertEqual(result["converge_reason"], "tolerance")
        self.assertEqual(result["converge_iterations"], 0)

    def test_rejected_trajectory_cannot_be_corrected(self):
        self.client.iks["right"].track = lambda *args, **kwargs: None
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.run_move(converge=3, duration=0.12)
        self.assertEqual(result["converge_reason"], "invalid_trajectory")
        self.assertEqual(result["converge_iterations"], 0)
        self.assertFalse(result["trajectory_valid"])

    def test_angle_wrap_uses_small_joint_correction(self):
        model = self.client.models["right"]
        model.forward_kinematics = lambda q: (np.array([math.sin(q[0]), math.cos(q[0]), 0]), np.eye(3))
        q_target = np.array([math.pi - 0.002, 0, 0, 0, 0, 0, 0])
        self.client.measured["right"] = q_target.copy()
        ik = self.client.iks["right"]
        solution = ik.track(np.zeros(3), np.eye(3))
        solution.q = q_target
        ik.track = lambda *args, **kwargs: solution
        self.client.response = lambda q: q - np.array([2 * math.pi + 0.004, 0, 0, 0, 0, 0, 0])
        result = ros.goto_cartesian(self.client, "right", model.forward_kinematics(q_target)[0],
                                    np.eye(3), duration=0.04, settle=0.04, converge=3)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["q_correction"][0], 0.004)

    def test_invalid_api_parameters_fail_before_any_send(self):
        for options in ({"converge": -1}, {"converge": 1.5}, {"converge": True},
                        {"converge_tol": math.nan}, {"converge_step": math.inf},
                        {"converge_total": -1}, {"converge_step": 0},
                        {"converge_step": 0.1, "converge_total": 0.01},
                        {"converge": 1, "settle": -1},
                        {"converge": 1, "duration": math.inf}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_move(**options)
        self.assertEqual(self.client.sent, [])


class CliTests(unittest.TestCase):
    def test_cartesian_failure_stops_before_return_home_or_next_target(self):
        cli = Client(Clock())
        cli.wait_state = lambda: True
        cli.check_contention = lambda: None
        cli.enter_control = lambda: True
        args = SimpleNamespace(target=None, side="right", duration=3.0, settle=2.0,
                               converge=3, converge_tol=1.0, converge_step=0.5,
                               converge_total=3.0)
        result = dict(pos_err=0.004, pos_err_xyz=np.array([0.004, 0, 0]), rot_err=0.0,
                      q_track_err=0.004, ik_fails=0, clipped=0, stale=True,
                      converged=False, converge_hist=[], converge_iterations=0,
                      converge_reason="stale_feedback", q_correction=np.zeros(7))
        targets = [("first", np.zeros(3), np.eye(3)), ("second", np.ones(3), np.eye(3))]
        with mock.patch.object(ros, "goto_joint") as move_joint:
            with mock.patch.object(ros, "goto_cartesian", return_value=result) as move_cartesian:
                with mock.patch.object(ros, "default_targets", return_value=targets):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(ros.cmd_cartesian(cli, args), 1)
        self.assertEqual(move_joint.call_count, 1)  # 只允许最开始的 HOME。
        self.assertEqual(move_cartesian.call_count, 1)

    def test_cli_initial_incomplete_feedback_does_not_send_home_or_pose(self):
        cli = Client(Clock())
        cli.wait_state = lambda: True
        cli.state_names = []
        args = SimpleNamespace(converge=3)
        with mock.patch.object(ros, "goto_joint") as move_joint:
            with mock.patch.object(ros, "goto_cartesian") as move_cartesian:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(ros.cmd_cartesian(cli, args), 1)
                    self.assertEqual(ros.cmd_pose(cli, args), 1)
        move_joint.assert_not_called()
        move_cartesian.assert_not_called()

    def test_pose_preserves_default_output_and_forwards_enabled_options(self):
        clock = Clock()
        cli = Client(clock)
        cli.wait_state = lambda: True
        cli.check_contention = lambda: None
        cli.enter_control = lambda: True
        result = dict(q_meas=np.zeros(7), reached=np.zeros(3), pos_err=0.004,
                      pos_err_xyz=np.array([0.004, 0, 0]), rot_err=0.0,
                      ik_fails=0, clipped=0, q_track_err=0.004, step_rejects=0,
                      branch_rejects=0, peak_raw_dq=0.01, peak_accepted_dq=0.01,
                      trajectory_valid=True, stale=False)
        args = SimpleNamespace(target=[0.02, 0, 0], side="right", duration=3.0,
                               settle=2.0, rpy=None, drpy=None)
        with mock.patch.object(ros, "goto_cartesian", return_value=result) as move:
            with mock.patch.object(ros, "preflight"), mock.patch.object(ros, "preflight_line", return_value="预检通过"):
                with mock.patch.object(ros, "publish_target"):
                    old_output = io.StringIO()
                    with contextlib.redirect_stdout(old_output):
                        self.assertEqual(ros.cmd_pose(cli, args), 0)
                    self.assertEqual(move.call_args.kwargs, {})
                    args.converge = 0
                    new_output = io.StringIO()
                    with contextlib.redirect_stdout(new_output):
                        self.assertEqual(ros.cmd_pose(cli, args), 0)
                    self.assertEqual(old_output.getvalue(), new_output.getvalue())
                    args.converge, args.converge_tol = 3, 1.0
                    args.converge_step, args.converge_total = 0.5, 3.0
                    result.update(converge_hist=[0.004, 0.002], converge_iterations=1,
                                  converge_reason="diverging", converged=False,
                                  q_correction=np.zeros(7))
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(ros.cmd_pose(cli, args), 1)
                    self.assertEqual(move.call_args.kwargs["converge"], 3)
                    self.assertIn("4.000 -> 2.000", output.getvalue())
                    self.assertIn("停止原因 diverging", output.getvalue())

    def test_cli_unit_conversion_and_default_opt_out(self):
        parser = argparse.ArgumentParser()
        ros._add_converge_args(parser)
        self.assertEqual(ros._converge_options(parser.parse_args([])), {})
        values = ros._converge_options(parser.parse_args([
            "--converge", "3", "--converge-tol", "0.8", "--converge-step", "0.25",
            "--converge-total", "2"]))
        self.assertEqual(values["converge"], 3)
        self.assertAlmostEqual(values["converge_tol"], 0.0008)
        self.assertAlmostEqual(values["converge_step"], math.radians(0.25))
        self.assertAlmostEqual(values["converge_total"], math.radians(2))

    def test_both_commands_expose_flags_and_reject_bad_values_before_ros(self):
        for command in ("pose", "cartesian"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                ros.main([command, "--help"])
            self.assertEqual(raised.exception.code, 0)
            self.assertIn("--converge-total", output.getvalue())
            prefix = [command, "0.3", "-0.25", "0.05"] if command == "pose" else [command]
            for options in (["--converge", "-1"], ["--converge", "1.2"],
                            ["--converge-tol", "nan"], ["--converge-step", "inf"],
                            ["--converge-total", "-1"], ["--converge-step", "0"],
                            ["--converge", "1", "--settle", "nan"]):
                with self.subTest(command=command, options=options):
                    with mock.patch.object(ros, "X2ArmClient") as factory:
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                            ros.main(prefix + options)
                        self.assertEqual(raised.exception.code, 2)
                        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
