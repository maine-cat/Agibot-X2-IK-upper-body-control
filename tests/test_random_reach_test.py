"""随机到点计划与执行门禁的离线测试；所有在线客户端都是内存模拟。"""
from collections import Counter
from concurrent.futures import Future
import contextlib
import copy
import io
import math
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_sim_ros as ros
from x2_frames import HOME_Q


class FakeClient:
    """本地内存客户端；保留实际发布线程、原始输入和编码器偏差。"""
    def __init__(self, *args, **kwargs):
        self.mode = "upper_body"
        self.rate = 50.
        self.models = {s: ros.ArmModel(s) for s in ("left", "right")}
        self.iks = {s: ros.SrsArmIK(m) for s, m in self.models.items()}
        self.feedback = {s: HOME_Q.copy() for s in self.models}
        self._last_sent = {s: HOME_Q.copy() for s in self.models}
        self.state_count = 1
        self.state_names = list(ros.DEFAULT_ARM_ORDER)
        self.imu_count = dict(chest=1, pelvis=1)
        self.imu_has_orientation = dict(chest=True, pelvis=True)
        self._imu_last = {"pelvis": ((0., 0., 0., 1.), (0., 0., 9.81))}
        self.q_waist = np.zeros(3)
        self.grav = SimpleNamespace(source="pelvis", have_fix=True, last_reject="")
        self.refresh_gravity = Mock(return_value=np.array([0., 0., -9.81]))
        self.k_eff = np.full(7, 40.)
        self.bias_limit = math.radians(12.)
        self.recorder = None
        self.sent = []
        self.ros_threads = []
        self.publishers, self.subscribers = 1, 1
        self.node = SimpleNamespace(count_publishers=lambda _: self.publishers,
                                    count_subscribers=lambda _: self.subscribers)
        self.rclpy = SimpleNamespace(spin_once=self.spin_once, ok=lambda: True, shutdown=Mock())
        self.wait_state = Mock(return_value=True)
        self.fresh_state = Mock(side_effect=self._fresh)
        self.get_action = Mock(return_value="UPPERBODY_REMOTE_SPLIT")
        self.set_action = Mock(side_effect=AssertionError("状态切换禁止"))
        self.enter_control = Mock(side_effect=AssertionError("状态切换禁止"))
        self.close = Mock()

    def spin_once(self, *args, **kwargs):
        self.ros_threads.append(threading.get_ident())
        self.state_count += 1
        self.imu_count["pelvis"] += 1

    def spin(self, seconds):
        self.spin_once()

    def _fresh(self, *args, **kwargs):
        self.spin_once()
        return True

    def q(self, side):
        return self.feedback[side].copy()

    def dq(self, side):
        return np.zeros(7)

    def tau(self, side):
        return np.zeros(7)

    def send(self, left, right, *args):
        self.sent.append(dict(left=np.asarray(left).copy(), right=np.asarray(right).copy(),
                              at=time.monotonic(), thread=threading.get_ident()))
        self.ros_threads.append(threading.get_ident())
        self._last_sent = dict(left=np.asarray(left).copy(), right=np.asarray(right).copy())
        self.feedback = {side: q.copy() for side, q in self._last_sent.items()}
        self.spin_once()


class FakeActionQuery:
    def __init__(self, client, *args, **kwargs):
        self.client = client

    def start(self):
        pass

    def poll(self):
        return True, self.client.get_action()

    def close(self):
        pass


class PlanTests(unittest.TestCase):
    def setUp(self):
        import x2_random_reach_plan
        self.plan = x2_random_reach_plan

    @staticmethod
    def args(**overrides):
        args = dict(side="right", count=2, seed=20260915, duration=8.)
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_candidates_reproducible_and_all_requested_samples_retained(self):
        first = self.plan.random_candidates("right", count=10, seed=14)
        again = self.plan.random_candidates("right", count=10, seed=14)
        changed = self.plan.random_candidates("right", count=10, seed=15)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 10)
        self.assertEqual(len({row["id"] for row in first}), 10)
        self.assertNotEqual([row["pos"] for row in first], [row["pos"] for row in changed])
        counts = Counter(tuple(row["cell"]) for row in first)
        self.assertEqual(len(counts), 8)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for row in first:
            x, y, z = row["pos"]
            self.assertTrue(.08 <= x <= .16)
            self.assertTrue(.23 <= abs(y) <= .30)
            self.assertTrue(-.09 <= z <= -.03)

    def test_candidate_prefix_stable_and_left_right_use_independent_home_rotations(self):
        first = self.plan.random_candidates("right", count=3, seed=7)
        expanded = self.plan.random_candidates("right", count=12, seed=7)
        self.assertEqual(first, expanded[:3])
        left = self.plan.random_candidates("left", count=3, seed=7)
        left_rot = ros.ArmModel("left").forward_kinematics(HOME_Q)[1]
        right_rot = ros.ArmModel("right").forward_kinematics(HOME_Q)[1]
        for lrow, rrow in zip(left, first):
            np.testing.assert_array_equal(lrow["pos"], np.asarray(rrow["pos"]) * [1., -1., 1.])
            np.testing.assert_array_equal(lrow["rot"], left_rot)
            np.testing.assert_array_equal(rrow["rot"], right_rot)

    def test_illegal_sampling_and_duration_values_rejected(self):
        for options in (dict(count=0), dict(count=33), dict(count=True), dict(seed=-1),
                        dict(side="both"), dict(bounds=((.16, .08), (.23, .30), (-.09, -.03))),
                        dict(bounds=((math.nan, .16), (.23, .30), (-.09, -.03)))):
            kwargs = dict(side="right", count=1)
            kwargs.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan.random_candidates(**kwargs)
        for duration in (0., 7.9, math.nan, math.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.plan.make_plan(self.args(count=1, duration=duration), ros)

    def test_small_real_plan_is_reproducible_and_does_not_hide_rejected_candidates(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = self.plan.make_plan(self.args(count=2), ros)
            again = self.plan.make_plan(self.args(count=2), ros)
        self.assertEqual(first["plan_sha256"], again["plan_sha256"])
        self.assertEqual(len(first["candidates"]), 2)
        self.assertEqual([row["id"] for row in first["candidates"]],
                         [row["id"] for row in again["candidates"]])
        self.assertGreater(len(first["accepted_ids"]), 0)
        for row in first["candidates"]:
            self.assertNotIn("trace", row)

    def test_endpoint_and_path_rejections_keep_all_original_candidates_without_resampling(self):
        endpoints = [dict(accepted=False, reason="no_solution"),
                     dict(accepted=True, reason="accepted"),
                     dict(accepted=True, reason="accepted")]
        paths = [dict(accepted=False, reason="tracking_no_solution", q_end_rad=None),
                 dict(accepted=True, reason="accepted", q_end_rad=HOME_Q.tolist()),
                 dict(accepted=False, reason="path_margin_gate", q_end_rad=None)]
        original = self.plan.random_candidates("right", count=3)
        with patch.object(self.plan, "endpoint_check", side_effect=endpoints) as endpoint, \
                patch.object(self.plan, "segment_summary", side_effect=paths) as segment:
            plan = self.plan.make_plan(self.args(count=3), ros)
        self.assertEqual(endpoint.call_count, 3)
        self.assertEqual(segment.call_count, 3)
        self.assertEqual(plan["accepted_ids"], [])
        self.assertEqual(plan["summary"]["total"], 3)
        self.assertEqual(plan["summary"]["rejected"], 3)
        self.assertEqual(plan["summary"]["success_rate"], 0.)
        self.assertEqual(plan["summary"]["outward_evaluated"], 2)
        self.assertEqual(plan["summary"]["return_evaluated"], 1)
        for before, after in zip(original, plan["candidates"]):
            self.assertEqual(before["id"], after["id"])
            self.assertEqual(before["pos"], after["pos"])
        self.assertIsNone(plan["candidates"][0]["outward"])
        self.assertIsNone(plan["candidates"][0]["return_path"])
        self.assertIsNone(plan["candidates"][1]["return_path"])
        self.assertEqual([row["reason"] for row in plan["candidates"]],
                         ["endpoint:no_solution", "outward:tracking_no_solution", "return:path_margin_gate"])

    def test_hash_describes_deterministic_inputs_instead_of_computation_timings(self):
        with patch.object(self.plan, "endpoint_check", return_value=dict(
                accepted=False, reason="no_solution", solve_ms=1.)):
            first = self.plan.make_plan(self.args(count=1))
        with patch.object(self.plan, "endpoint_check", return_value=dict(
                accepted=False, reason="no_solution", solve_ms=900.)):
            again = self.plan.make_plan(self.args(count=1))
            changed = self.plan.make_plan(self.args(count=1, seed=77))
        self.assertEqual(first["plan_sha256"], again["plan_sha256"])
        self.assertNotEqual(first["plan_sha256"], changed["plan_sha256"])
        self.assertEqual(first["plan_sha256"], self.plan.plan_digest(first["hash_input"]))

    def test_endpoint_solver_uses_new_home_seed_and_strict_original_pose(self):
        model = ros.ArmModel("right")
        pos, rot = model.forward_kinematics(HOME_Q)
        seeds = []
        solver = SimpleNamespace(project_to_workspace=Mock(), solve_hold_rotation=Mock())
        def solve(target, rotation, *, q_seed):
            np.testing.assert_array_equal(target, pos)
            np.testing.assert_array_equal(rotation, rot)
            np.testing.assert_array_equal(q_seed, HOME_Q)
            seeds.append(q_seed)
            q_seed[0] = 50.
            return SimpleNamespace(q=HOME_Q.copy())
        solver.solve = Mock(side_effect=solve)
        self.assertTrue(self.plan.endpoint_check(model, solver, pos, rot)["accepted"])
        self.assertTrue(self.plan.endpoint_check(model, solver, pos, rot)["accepted"])
        self.assertIsNot(seeds[0], seeds[1])
        solver.project_to_workspace.assert_not_called()
        solver.solve_hold_rotation.assert_not_called()
        np.testing.assert_array_equal(HOME_Q, [.4, 0., 0., -1.2, 0., 0., 0.])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        import x2_random_reach_test
        self.runner = x2_random_reach_test
        self.client = FakeClient()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run.json"
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)
        self.query = patch.object(self.runner, "ActionQuery", FakeActionQuery)
        self.query.start()
        self.addCleanup(self.query.stop)
        self.real_client_class = ros.X2ArmClient
        self.factory = patch.object(ros, "X2ArmClient", return_value=self.client)
        self.factory_mock = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.calibration = patch.object(self.runner.base, "calibration", return_value=dict(
            sn="TEST_ROBOT", stiffness=40., bias_limit_deg=12., gravity_source="pelvis"))
        self.calibration.start()
        self.addCleanup(self.calibration.stop)

    def args(self, *mode, **overrides):
        args = self.runner.parser().parse_args([*mode, "--count", "1", "--output", str(self.output)])
        for key, value in overrides.items():
            setattr(args, key, value)
        self.runner.validate_args(args)
        return args

    def plan(self, accepted=True):
        pos, rot = self.client.models["right"].forward_kinematics(HOME_Q)
        point = dict(id="right-test", side="right", pos=pos.tolist(), rot=rot.tolist(),
                     accepted=accepted, reason="accepted" if accepted else "endpoint:no_solution")
        return dict(side="right", seed=20260915, home_pos=pos.tolist(), fixed_rot=rot.tolist(),
                    home_q=HOME_Q.tolist(), candidates=[point],
                    accepted_ids=[point["id"]] if accepted else [], plan_sha256="a" * 64)

    @staticmethod
    def result(**overrides):
        result = dict(stale=False, trajectory_valid=True, aborted=False, ik_fails=0,
                      clipped=0, step_rejects=0, branch_rejects=0, implicit_fallbacks=0,
                      pos_err=.0005, rot_err=.001, converged=True, converge_reason="tolerance")
        result.update(overrides)
        return result

    def report(self):
        return json.loads(self.output.read_text())

    def test_default_offline_never_bootstraps_or_creates_ros_client(self):
        with patch.object(self.runner, "make_plan", return_value=self.plan()), \
                patch.object(self.runner, "bootstrap") as bootstrap:
            code = self.runner.main(["--count", "1", "--output", str(self.output)])
        self.assertEqual(code, 0)
        bootstrap.assert_not_called()
        self.factory_mock.assert_not_called()
        self.assertEqual(self.client.sent, [])
        self.assertEqual(self.report()["mode"], "offline")

    def test_preflight_never_publishes_and_releases_original_client(self):
        code = self.runner.run(self.args("--preflight"), ros, self.plan())
        self.assertEqual(code, 0)
        self.assertEqual(self.client.sent, [])
        self.client.close.assert_called_once()
        self.client.rclpy.shutdown.assert_called_once()
        self.assertEqual(self.report()["status"], "preflight_passed")

    def test_non_urs_rejected_before_motion_without_state_changes(self):
        self.client.get_action.return_value = "JOINT_DEFAULT"
        code = self.runner.run(self.args("--execute"), ros, self.plan())
        self.assertNotEqual(code, 0)
        self.assertEqual(self.client.sent, [])
        self.assertIn("URS", self.report()["error"])
        self.client.close.assert_called_once()
        with self.assertRaises(RuntimeError):
            self.client.set_action("UPPERBODY_REMOTE_SPLIT")

    def test_invalid_arguments_and_existing_output_rejected_without_planning(self):
        for key, value in (("count", 0), ("count", 33), ("duration", 7.9),
                           ("duration", math.nan), ("settle", 1.), ("settle", math.inf),
                           ("converge", 9), ("repeats", 0), ("plan_sha256", "bad")):
            args = self.runner.parser().parse_args([])
            setattr(args, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.runner.validate_args(args)
        self.output.write_text("preserve me")
        with patch.object(self.runner, "make_plan") as make_plan:
            code = self.runner.main(["--output", str(self.output)])
        self.assertEqual(code, 2)
        make_plan.assert_not_called()
        self.factory_mock.assert_not_called()
        self.assertEqual(self.output.read_text(), "preserve me")

    def test_all_rejected_candidates_keep_report_denominator_and_do_not_connect(self):
        code = self.runner.run(self.args("--execute"), ros, self.plan(accepted=False))
        self.assertNotEqual(code, 0)
        self.factory_mock.assert_not_called()
        summary = self.report()["summary"]
        self.assertEqual(summary["sampled_candidates"], 1)
        self.assertEqual(summary["candidate_rejected"], 1)
        self.assertEqual(summary["planned_legs"], 0)

    def test_plan_hash_mismatch_refuses_before_connecting(self):
        code = self.runner.run(self.args("--execute", plan_sha256="b" * 64), ros, self.plan())
        self.assertNotEqual(code, 0)
        self.factory_mock.assert_not_called()
        self.assertEqual(self.client.sent, [])
        self.assertIn("SHA-256", self.report()["error"])

    def test_invalid_start_feedback_or_home_pose_rejected_before_any_publish(self):
        for fault in ("missing_joint", "nonfinite", "joint_distance", "wrist_rotation"):
            with self.subTest(fault=fault):
                client = FakeClient()
                if fault == "missing_joint":
                    client.state_names.pop()
                elif fault == "nonfinite":
                    client.feedback["right"][0] = math.nan
                elif fault == "joint_distance":
                    client.feedback["right"][0] += math.radians(3.1)
                else:
                    client.feedback["right"][6] += math.radians(.6)
                with self.assertRaises(RuntimeError):
                    self.runner.check_start(client, ros, "right")
                self.assertEqual(client.sent, [])

    def test_real_imu_callback_nan_quaternion_cannot_pass_read_only_preflight(self):
        msg = SimpleNamespace(orientation=SimpleNamespace(x=math.nan, y=0., z=0., w=1.),
                              linear_acceleration=SimpleNamespace(x=0., y=0., z=9.81),
                              orientation_covariance=[0.] * 9)
        self.real_client_class._on_imu(self.client, "pelvis", msg)
        self.assertTrue(self.client.imu_has_orientation["pelvis"])
        code = self.runner.run(self.args("--preflight"), ros, self.plan())
        self.assertNotEqual(code, 0)
        self.assertIn("四元数", self.report()["error"])
        self.assertEqual(self.client.sent, [])
        self.client.close.assert_called_once()
        self.client.rclpy.shutdown.assert_called_once()

    def test_invalid_gravity_inputs_rejected_before_publish_and_remain_failed(self):
        faults = {
            "quat_nan": lambda c: c._imu_last.update(pelvis=((math.nan, 0., 0., 1.), (0., 0., 9.81))),
            "quat_zero": lambda c: c._imu_last.update(pelvis=((0., 0., 0., 0.), (0., 0., 9.81))),
            "quat_missing": lambda c: c._imu_last.clear(),
            "quat_shape": lambda c: c._imu_last.update(pelvis=((0., 0., 1.), (0., 0., 9.81))),
            "waist_nan": lambda c: setattr(c, "q_waist", np.array([math.nan, 0., 0.])),
            "waist_shape": lambda c: setattr(c, "q_waist", np.zeros(2)),
            "k_nan": lambda c: setattr(c, "k_eff", np.full(7, math.nan)),
            "k_zero": lambda c: setattr(c, "k_eff", np.zeros(7)),
            "k_shape": lambda c: setattr(c, "k_eff", np.full(6, 40.)),
            "bias_nan": lambda c: setattr(c, "bias_limit", math.nan),
            "bias_high": lambda c: setattr(c, "bias_limit", math.radians(12.1)),
            "bias_negative": lambda c: setattr(c, "bias_limit", -.001),
            "gravity_nan": lambda c: setattr(c.refresh_gravity, "return_value", [0., 0., math.nan]),
            "gravity_zero": lambda c: setattr(c.refresh_gravity, "return_value", [0., 0., 0.]),
            "gravity_high": lambda c: setattr(c.refresh_gravity, "return_value", [0., 0., -11.1]),
            "gravity_shape": lambda c: setattr(c.refresh_gravity, "return_value", [0., -9.81]),
            "gravity_no_fix": lambda c: setattr(c.grav, "have_fix", False),
            "gravity_rejected": lambda c: setattr(c.grav, "last_reject", "tilt over limit"),
        }
        for name, inject in faults.items():
            with self.subTest(fault=name):
                client = FakeClient()
                monitor = self.runner.ReachMonitor(client, ros, "right")
                inject(client)
                with self.assertRaises(RuntimeError):
                    self.runner.check_start(client, ros, "right")
                with self.assertRaises(RuntimeError):
                    monitor.send(HOME_Q, HOME_Q)
                self.assertTrue(monitor.failed)
                self.assertEqual(client.sent, [])
                with self.assertRaises(RuntimeError):
                    monitor.hold_tick()
                self.assertEqual(client.sent, [])

    def test_tasks_discard_first_round_only_and_keep_returns_explicit(self):
        plan, args = self.plan(), self.args(repeats=2)
        tasks = self.runner.build_tasks(plan, args)
        self.assertEqual(len(tasks), 6)
        self.assertEqual([task["leg"] for task in tasks], ["target", "return"] * 3)
        self.assertEqual([task["discarded"] for task in tasks], [True, True, False, False, False, False])
        self.assertEqual({task["status"] for task in tasks}, {"not_attempted"})
        trials = [dict(tasks[0], result=self.result(pos_err=.005), precision_pass=False),
                  dict(tasks[1], result=self.result(), precision_pass=True),
                  dict(tasks[2], result=self.result(), precision_pass=True)]
        summary = self.runner.summarize(plan, tasks, trials)
        self.assertEqual(summary["valid_target_attempted"], 1)
        self.assertEqual(summary["valid_target_planned"], 2)
        self.assertEqual(summary["valid_target_unattempted"], 1)
        self.assertEqual(summary["valid_target_pass_rate"], .5)
        self.assertAlmostEqual(summary["position_mm"]["maximum"], .5)

    def test_warmup_precision_failure_stops_without_return_or_home(self):
        failed = (self.result(pos_err=.002, converged=False, converge_reason="max_iterations"), {}, False, None)
        with patch.object(self.runner, "execute_leg", return_value=failed) as execute:
            code = self.runner.run(self.args("--execute"), ros, self.plan())
        self.assertEqual(code, 3)
        execute.assert_called_once()
        report = self.report()
        self.assertTrue(report["trials"][0]["discarded"])
        self.assertEqual(report["trials"][0]["status"], "failed")
        self.assertEqual([task["status"] for task in report["tasks"]],
                         ["failed", "not_attempted", "not_attempted", "not_attempted"])
        self.assertEqual(self.client.sent, [])

    def test_converge_off_reports_low_precision_but_stops_outside_continue_limits(self):
        for error_mm, rotation_deg, expected_calls in ((2., .1, 4), (10.1, .1, 1), (.5, .6, 1)):
            with self.subTest(error_mm=error_mm, rotation_deg=rotation_deg):
                self.output = Path(self.directory.name) / f"off-{error_mm}-{rotation_deg}.json"
                result = self.result(pos_err=error_mm / 1000., rot_err=math.radians(rotation_deg),
                                     converged=False)
                with patch.object(self.runner, "execute_leg", return_value=(result, {}, False, None)) as execute:
                    code = self.runner.run(self.args("--execute", converge=0), ros, self.plan())
                self.assertEqual(code, 3)
                self.assertEqual(execute.call_count, expected_calls)
                self.assertEqual(self.report()["status"], "precision_not_met" if expected_calls == 4 else "aborted")
                self.assertEqual(self.client.sent, [])

    def test_exception_and_ctrl_c_finalize_report_without_running_recovery_leg(self):
        for error, expected in ((RuntimeError("motion failed"), 1), (KeyboardInterrupt(), 130)):
            self.output = Path(self.directory.name) / f"{expected}.json"
            self.client.close.reset_mock()
            self.client.rclpy.shutdown.reset_mock()
            with self.subTest(error=type(error).__name__), \
                    patch.object(self.runner, "execute_leg", side_effect=error) as execute:
                code = self.runner.run(self.args("--execute"), ros, self.plan())
            self.assertEqual(code, expected)
            execute.assert_called_once()
            self.assertEqual(self.report()["status"], "aborted")
            self.client.close.assert_called_once()
            self.client.rclpy.shutdown.assert_called_once()

    def test_short_move_failure_retains_actual_path_in_report(self):
        import x2_random_path_audit
        path = dict(accepted=True, reason="accepted", start_q_rad=HOME_Q.tolist(), trace=[])
        with patch.object(x2_random_path_audit, "audit_segment", return_value=path), \
                patch.object(ros, "goto_cartesian", side_effect=RuntimeError("short move failed")) as goto:
            code = self.runner.run(self.args("--execute"), ros, self.plan())
        self.assertEqual(code, 1)
        goto.assert_called_once()
        report = self.report()
        self.assertEqual(len(report["trials"]), 1)
        self.assertEqual(report["trials"][0]["actual_path"], {k: v for k, v in path.items() if k != "trace"})
        self.assertFalse(report["trials"][0]["precision_pass"])
        self.assertEqual(report["tasks"][1]["status"], "not_attempted")

    def test_action_failure_before_new_path_clears_previous_leg_summary(self):
        monitor = self.runner.ReachMonitor(self.client, ros, "right")
        monitor.last_path_summary = dict(accepted=True, reason="previous leg")
        monitor.confirm_action = Mock(side_effect=RuntimeError("lost URS"))
        monitor.plan_from_feedback = Mock()
        args = self.args("--execute")
        task = self.runner.build_tasks(self.plan(), args)[1]
        with self.assertRaisesRegex(RuntimeError, "lost URS"):
            self.runner.execute_leg(monitor, task, args, ros)
        monitor.plan_from_feedback.assert_not_called()
        self.assertIsNone(monitor.last_path_summary)
        self.assertEqual(self.client.sent, [])

    def test_cleanup_failure_still_shuts_down_ros_and_saves_report(self):
        self.client.close.side_effect = OSError("destroy failed")
        code = self.runner.run(self.args("--preflight"), ros, self.plan())
        self.assertNotEqual(code, 0)
        self.client.rclpy.shutdown.assert_called_once()
        self.assertIn("client", self.report()["cleanup_errors"])

    def test_monitor_keeps_peer_original_input_despite_feedback_drift(self):
        monitor = self.runner.ReachMonitor(self.client, ros, "right")
        initial = self.client.q("left")
        monitor.hold_tick()
        self.client.feedback["left"][0] += .025
        monitor.hold_tick()
        for frame in self.client.sent:
            np.testing.assert_array_equal(frame["left"], initial)
        self.assertTrue(all(frame["thread"] == threading.get_ident() for frame in self.client.sent))

    def test_send_gap_invalid_input_and_uncertain_send_make_monitor_sticky_failed(self):
        for fault in ("gap", "nan", "limit", "step", "tracking", "partial"):
            with self.subTest(fault=fault):
                client = FakeClient()
                if fault == "partial":
                    publish = client.send
                    def partial_send(*args):
                        publish(*args)
                        raise OSError("recorder failed after publish")
                    client.send = partial_send
                monitor = self.runner.ReachMonitor(client, ros, "right")
                target = HOME_Q.copy()
                if fault == "gap":
                    monitor.last_send = time.monotonic() - .21
                elif fault == "nan":
                    target[0] = math.nan
                elif fault == "limit":
                    target[0] = client.models["right"].q_max[0] + .1
                elif fault == "step":
                    target[0] += .051
                elif fault == "tracking":
                    client.feedback["right"][0] += math.radians(6.)
                with self.assertRaises((RuntimeError, OSError)):
                    monitor.send(HOME_Q, target)
                self.assertTrue(monitor.failed)
                count = len(client.sent)
                with self.assertRaises(RuntimeError):
                    monitor.hold_tick()
                self.assertEqual(len(client.sent), count)

    def test_slow_parent_graph_check_cannot_publish_after_command_deadline(self):
        monitor = self.runner.ReachMonitor(self.client, ros, "right")
        now = [10.001]
        monitor.last_send = monitor.last_fresh_at = 10.
        monitor.last_graph_at = 9.

        def slow_graph(*args):
            now[0] += .21
            return dict(publishers=1, subscribers=1)

        with patch.object(self.runner.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(self.runner.base, "check_graph", side_effect=slow_graph) as graph:
            with self.assertRaises(RuntimeError):
                monitor.send(HOME_Q, HOME_Q)
            graph.assert_called_once()
            self.assertEqual(self.client.sent, [])
            self.assertTrue(monitor.failed)
            with self.assertRaises(RuntimeError):
                monitor.hold_tick()
            self.assertEqual(self.client.sent, [])

    def test_action_query_keeps_holding_until_result_then_rejects_unknown_state(self):
        monitor = self.runner.ReachMonitor(self.client, ros, "right")
        monitor.query = SimpleNamespace(start=Mock(), poll=Mock(side_effect=[
            (False, None), (False, None), (True, "UNKNOWN")]), close=Mock())
        with self.assertRaisesRegex(RuntimeError, "URS"):
            monitor.confirm_action()
        self.assertEqual(len(self.client.sent), 2)
        for frame in self.client.sent:
            np.testing.assert_array_equal(frame["left"], HOME_Q)
            np.testing.assert_array_equal(frame["right"], HOME_Q)
        self.client.set_action.assert_not_called()
        self.client.enter_control.assert_not_called()
        monitor.close_query()
        self.assertIsNone(monitor.query)

    def test_actual_feedback_path_planning_runs_only_pure_code_off_main_thread_with_hold(self):
        import x2_random_path_audit
        monitor = self.runner.ReachMonitor(self.client, ros, "right")
        observed = []
        def delayed_path(model, ik, q, pos, rot, duration):
            observed.append((threading.get_ident(), q.copy()))
            time.sleep(.07)
            return dict(accepted=True, reason="accepted", trace=[])
        pos, rot = self.client.models["right"].forward_kinematics(HOME_Q)
        with patch.object(x2_random_path_audit, "audit_segment", side_effect=delayed_path):
            result = monitor.plan_from_feedback(pos, rot, 8.)
        self.assertTrue(result["accepted"])
        self.assertGreaterEqual(len(self.client.sent), 4)
        self.assertNotEqual(observed[0][0], threading.get_ident())
        np.testing.assert_array_equal(observed[0][1], HOME_Q)
        self.assertEqual(set(self.client.ros_threads), {threading.get_ident()})
        gaps = np.diff([frame["at"] for frame in self.client.sent])
        self.assertLess(float(gaps.max()), .2)
        self.assertGreater(float(gaps.min()), .005)

    def test_actual_feedback_path_rejection_stops_before_replaying_any_trajectory(self):
        import x2_random_path_audit
        monitor = self.runner.ReachMonitor(self.client, ros, "right")
        pos, rot = self.client.models["right"].forward_kinematics(HOME_Q)
        with patch.object(x2_random_path_audit, "audit_segment", return_value=dict(
                accepted=False, reason="path_pose_tolerance", trace=[])):
            with self.assertRaises(RuntimeError):
                monitor.plan_from_feedback(pos, rot, 8.)
        self.assertEqual(monitor.last_path_summary, dict(accepted=False, reason="path_pose_tolerance"))
        for frame in self.client.sent:
            np.testing.assert_array_equal(frame["right"], HOME_Q)

    def test_actual_feedback_drift_invalidates_completed_path(self):
        monitor = self.runner.ReachMonitor(self.client, ros, "right")
        monitor.fresh_state = Mock(return_value=True)
        future = Future()
        future.set_result(dict(accepted=True, trace=[]))
        job = SimpleNamespace(future=future, close=Mock())

        def finish_after_drift(*args):
            self.client.feedback["right"][0] += .011
            return job

        pos, rot = self.client.models["right"].forward_kinematics(HOME_Q)
        with patch.object(self.runner, "PlanningJob", side_effect=finish_after_drift):
            with self.assertRaisesRegex(RuntimeError, "0.01 rad"):
                monitor.plan_from_feedback(pos, rot, 8.)
        job.close.assert_called_once()
        self.assertEqual(self.client.sent, [])

    def test_execute_leg_rejects_invalid_or_nonfinite_trajectory_and_pose(self):
        args = self.args("--execute")
        for overrides in (dict(stale=True), dict(trajectory_valid=False), dict(implicit_fallbacks=1),
                          dict(pos_err=math.nan), dict(rot_err=math.nan), dict(rot_err=math.radians(.6))):
            with self.subTest(overrides=overrides):
                monitor = self.runner.ReachMonitor(self.client, ros, "right")
                monitor.confirm_action = Mock()
                monitor.plan_from_feedback = Mock(return_value=dict(accepted=True, trace=[]))
                task = self.runner.build_tasks(self.plan(), args)[0]
                with patch.object(ros, "goto_cartesian", return_value=self.result(**overrides)):
                    _, _, passed, error = self.runner.execute_leg(monitor, task, args, ros)
                self.assertFalse(passed)
                if not (set(overrides) == {"rot_err"} and math.isfinite(overrides["rot_err"])):
                    self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
