"""API 接线与 URS 门禁的离线测试；不导入 ROS，也不连接硬件。"""
import math
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_api as api
import x2_sim_ros as ros


class FakeClient:
    def __init__(self, mode="upper_body", *args, **kwargs):
        self.mode = mode
        self.options = kwargs
        self.state_count = 1
        self.state_names = list(ros.DEFAULT_ARM_ORDER)
        self.models = {s: api.ArmModel(s) for s in ("left", "right")}
        self.iks = {s: api.SrsArmIK(m) for s, m in self.models.items()}
        self.feedback = {s: api.HOME.copy() for s in self.models}
        self.rate = 50.
        self.recorder = None
        self.sent = []
        self.publishers = 1
        self.subscribers = 1
        self.node = SimpleNamespace(
            count_publishers=lambda _: self.publishers,
            count_subscribers=lambda _: self.subscribers)
        self.rclpy = SimpleNamespace(ok=lambda: True, shutdown=Mock())
        self.imu_count = dict(chest=1, pelvis=1)
        self.imu_has_orientation = dict(chest=True, pelvis=True)
        self.get_action = Mock(return_value="UPPERBODY_REMOTE_SPLIT")
        self.wait_state = Mock(return_value=True)
        self.spin = Mock()
        self.close = Mock()
        self.fresh_state = Mock(side_effect=self._fresh)
        self.set_action = Mock(side_effect=AssertionError("禁止状态切换"))
        self.enter_control = Mock(side_effect=AssertionError("禁止进入状态逻辑"))

    def _fresh(self, *args, **kwargs):
        self.state_count += 1
        return True

    def q(self, side):
        return self.feedback[side].copy()

    def dq(self, side):
        return np.zeros(7)

    def tau(self, side):
        return np.zeros(7)

    def send(self, q_left, q_right, *args):
        self.sent.append((q_left.copy(), q_right.copy()))
        # 模拟补偿后另一臂反馈偏离原始输入，下一次调用不能把它作为新输入。
        self.feedback["left"] = q_left + .001
        self.feedback["right"] = q_right.copy()
        self.state_count += 1


class ApiTests(unittest.TestCase):
    def setUp(self):
        api.X2Arm._connected = None
        # Synthetic per-test calibration, never the developer's field record.
        temporary = tempfile.TemporaryDirectory(prefix="x2ik-api-test-")
        self.addCleanup(temporary.cleanup)
        calibration = Path(temporary.name)
        (calibration / "TEST_ROBOT.json").write_text(json.dumps(dict(
            sn="TEST_ROBOT", stiffness=40., bias_limit_deg=12., gravity_source="pelvis")))
        self.calibration_path = patch.object(ros, "CALIB_DIR", calibration)
        self.calibration_path.start()
        self.addCleanup(self.calibration_path.stop)
        self.env = patch.dict(os.environ, {"X2_ROBOT_SN": "TEST_ROBOT"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.factory = patch.object(ros, "X2ArmClient", side_effect=FakeClient)
        self.factory_mock = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.pace = patch.object(ros, "_pace")
        self.pace.start()
        self.addCleanup(self.pace.stop)

    def arm(self, **kwargs):
        arm = api.X2Arm(verbose=False, **kwargs)
        self.addCleanup(arm.close)
        return arm

    def test_offline_has_no_ros_dependency(self):
        script = """
import sys
sys.modules['rclpy'] = None
from x2_api import X2Arm, HOME
arm = X2Arm(connect=False)
p, r = arm.fk(HOME)
assert arm.ik(p, r) is not None
"""
        subprocess.run([sys.executable, "-c", script], check=True,
                       cwd=Path(__file__).resolve().parents[1])

    def test_configuration_loaded_by_sn_and_no_state_switch(self):
        arm = self.arm()
        self.assertEqual(arm.robot_sn, "TEST_ROBOT")
        np.testing.assert_array_equal(arm.cli.options["joint_stiffness"], np.full(7, 40.))
        self.assertAlmostEqual(arm.cli.options["bias_limit"], math.radians(12.))
        self.assertEqual(arm.cli.options["gravity_source"], "pelvis")
        arm.cli.set_action.assert_not_called()
        arm.cli.enter_control.assert_not_called()
        self.assertEqual(arm.cli.sent, [])

    def test_configuration_explicit_values_win(self):
        arm = self.arm(stiffness=35, bias_limit_deg=0, gravity_source="static")
        np.testing.assert_array_equal(arm.cli.options["joint_stiffness"], np.full(7, 35.))
        self.assertEqual(arm.cli.options["bias_limit"], 0)
        self.assertEqual(arm.cli.options["gravity_source"], "static")

    def test_explicit_sn_wins_over_environment(self):
        with patch.object(ros, "load_calibration", return_value={"sn": "chosen"}) as load:
            arm = self.arm(robot_sn="chosen", gravity_source="static")
        load.assert_called_once_with("chosen")
        self.assertEqual(arm.robot_sn, "chosen")

    def test_missing_or_mismatched_calibration_rejected_before_ros(self):
        for value in (None, {"sn": "wrong-machine"}):
            with self.subTest(value=value), patch.object(ros, "load_calibration", return_value=value):
                with self.assertRaises(ValueError):
                    self.arm()
        self.factory_mock.assert_not_called()

    def test_direct_mode_rejected_before_client_creation(self):
        with self.assertRaises(ValueError):
            self.arm(mode="joint_direct")
        self.factory_mock.assert_not_called()

    def test_connection_failure_releases_client(self):
        client = FakeClient()
        client.wait_state.return_value = False
        self.factory_mock.side_effect = None
        self.factory_mock.return_value = client
        with self.assertRaises(RuntimeError):
            self.arm()
        client.close.assert_called_once()
        client.rclpy.shutdown.assert_called_once()
        self.assertIsNone(api.X2Arm._connected)

    def test_only_one_connected_instance(self):
        arm = self.arm()
        with self.assertRaises(RuntimeError):
            self.arm(side="left")
        self.assertEqual(self.factory_mock.call_count, 1)
        arm.close()
        self.arm(side="left")

    def test_record_flush_error_still_closes_client_and_releases_instance(self):
        arm = self.arm()
        client = arm.cli
        failure = OSError("record flush failed")
        client.recorder = SimpleNamespace(close=Mock(side_effect=failure))
        with self.assertRaises(OSError) as caught:
            arm.close()
        self.assertIs(caught.exception, failure)
        client.recorder.close.assert_called_once()
        client.close.assert_called_once()
        client.rclpy.shutdown.assert_called_once()
        self.assertIsNone(arm.cli)
        self.assertIsNone(api.X2Arm._connected)
        arm.close()  # 已清理的实例不会重复写盘或关闭 ROS。
        client.recorder.close.assert_called_once()
        client.close.assert_called_once()
        self.arm(side="left")

    def test_node_close_error_still_shuts_down_and_releases_instance(self):
        arm = self.arm()
        client = arm.cli
        failure = RuntimeError("destroy node failed")
        client.close.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            arm.close()
        self.assertIs(caught.exception, failure)
        client.rclpy.shutdown.assert_called_once()
        self.assertIsNone(arm.cli)
        self.assertIsNone(api.X2Arm._connected)

    def test_shutdown_error_preserves_flush_error_context_and_releases_instance(self):
        arm = self.arm()
        client = arm.cli
        flush_failure = OSError("record flush failed")
        shutdown_failure = RuntimeError("shutdown failed")
        client.recorder = SimpleNamespace(close=Mock(side_effect=flush_failure))
        client.rclpy.shutdown.side_effect = shutdown_failure
        with self.assertRaises(RuntimeError) as caught:
            arm.close()
        self.assertIs(caught.exception, shutdown_failure)
        self.assertIs(caught.exception.__context__, flush_failure)
        client.close.assert_called_once()
        self.assertIsNone(arm.cli)
        self.assertIsNone(api.X2Arm._connected)

    def test_non_urs_unknown_and_competing_publisher_block_motion(self):
        arm = self.arm()
        for action in (None, "STAND_DEFAULT", "JOINT_DEFAULT", "WHOLE_BODY_TELEOP"):
            arm.cli.get_action.return_value = action
            with self.subTest(action=action), self.assertRaises(RuntimeError):
                arm.move_j(api.HOME)
        arm.cli.get_action.return_value = "URS"
        arm.cli.publishers = 2
        with self.assertRaises(RuntimeError):
            arm.move_j(api.HOME)
        self.assertEqual(arm.cli.sent, [])
        arm.cli.set_action.assert_not_called()
        arm.cli.enter_control.assert_not_called()

    def test_missing_stale_nonfinite_and_out_of_limit_feedback_block_motion(self):
        arm = self.arm()
        names = arm.cli.state_names.copy()
        arm.cli.state_names = names[:-1]
        with self.assertRaises(RuntimeError):
            arm.move_j(api.HOME)
        arm.cli.state_names = names
        arm.cli.fresh_state.side_effect = None
        arm.cli.fresh_state.return_value = False
        with self.assertRaises(RuntimeError):
            arm.move_j(api.HOME)
        arm.cli.fresh_state.side_effect = arm.cli._fresh
        arm.cli.feedback["right"][0] = float("nan")
        with self.assertRaises(RuntimeError):
            arm.move_j(api.HOME)
        arm.cli.feedback["right"] = arm.model.q_max + .1
        with self.assertRaises(ValueError):
            arm.move_j(api.HOME)
        self.assertEqual(arm.cli.sent, [])

    def test_invalid_targets_and_timings_never_publish_or_query_action(self):
        arm = self.arm()
        arm.cli.get_action.reset_mock()
        invalid = [lambda: arm.move_j([0] * 6),
                   lambda: arm.move_j([float("nan")] * 7),
                   lambda: arm.move_j(arm.model.q_max + .1),
                   lambda: arm.move_j(api.HOME, duration=0),
                   lambda: arm.move_j(api.HOME, settle=-1),
                   lambda: arm.move_j_both(api.HOME, [float("inf")] * 7),
                   lambda: arm.home(duration=float("nan")),
                   lambda: arm.move_l([0, 0]),
                   lambda: arm.move_l([0, 0, float("nan")]),
                   lambda: arm.move_l([0, 0, 0], rpy=[0, 0]),
                   lambda: arm.move_l([0, 0, 0], duration=-1),
                   lambda: arm.move_l([0, 0, 0], converge=-1),
                   lambda: arm.move_l([0, 0, 0], converge_tol=float("nan"))]
        for call in invalid:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()
        arm.cli.get_action.assert_not_called()
        self.assertEqual(arm.cli.sent, [])

    def test_single_arm_reuses_raw_other_input_between_moves(self):
        arm = self.arm()
        first = arm.move_j(api.HOME, duration=.04, settle=0)
        arm.cli.feedback["left"] += .05
        second = arm.move_j(api.HOME + .01, duration=.04, settle=0)
        for left, _ in arm.cli.sent:
            np.testing.assert_array_equal(left, api.HOME)
        self.assertFalse(first["stale"])
        self.assertAlmostEqual(second["err_max"], 0)
        # 每次 MoveJ 尾部至少多发一次保持，并以该帧反馈更新结果。
        self.assertEqual(len(arm.cli.sent), 6)

    def test_home_preserves_explicit_two_arm_semantics(self):
        arm = self.arm()
        result = arm.home(duration=.02, settle=0)
        self.assertEqual(set(result), {"left", "right"})
        for left, right in arm.cli.sent:
            np.testing.assert_array_equal(left, api.HOME)
            np.testing.assert_array_equal(right, api.HOME)

    def test_both_move_resets_other_arm_baseline_to_command_not_feedback(self):
        arm = self.arm()
        left_target = api.HOME + .02
        arm.move_j_both(left_target, api.HOME, duration=.02, settle=0)
        arm.move_j(api.HOME, duration=.02, settle=0)
        np.testing.assert_array_equal(arm.cli.sent[-1][0], left_target)

    def test_movel_forwards_converge_units_and_reuses_other_input(self):
        arm = self.arm()
        arm.cli.feedback["left"] += .05
        pos, rpy = arm.fk(api.HOME)
        def execute(client, side, pos, rot, duration, settle, **kwargs):
            client.send(client.q("left"), client.q("right"))
            return dict(converged=False, converge_reason="max_iterations", **kwargs)
        with patch.object(ros, "goto_cartesian", side_effect=execute) as move:
            result = arm.move_l(pos, rpy, converge=8, converge_tol=.0008,
                                converge_step=.005, converge_total=.04)
        self.assertEqual(result["converge"], 8)
        self.assertEqual(result["converge_tol"], .0008)
        self.assertEqual(result["converge_step"], .005)
        self.assertEqual(result["converge_total"], .04)
        self.assertFalse(result["converged"])
        np.testing.assert_array_equal(arm.cli.sent[-1][0], api.HOME)

    def test_movel_default_is_no_convergence(self):
        arm = self.arm()
        pos, rpy = arm.fk(api.HOME)
        with patch.object(ros, "goto_cartesian", return_value={}) as move:
            arm.move_l(pos, rpy)
        self.assertEqual(move.call_args.kwargs["converge"], 0)

    def test_movel_wait_for_feedback_continues_last_input(self):
        arm = self.arm()
        pos, rpy = arm.fk(api.HOME)
        result = arm.move_l(pos, rpy, duration=.1, settle=0)
        self.assertFalse(result["stale"])
        self.assertLess(result["pos_err"], 1e-6)
        self.assertGreaterEqual(len(arm.cli.sent), result["sent_frames"] + 1)
        np.testing.assert_array_equal(arm.cli.sent[-1][1], result["q_cmd"])

    def test_runtime_stale_feedback_and_gap_stop_before_next_publish(self):
        arm = self.arm()
        with patch.object(api.time, "monotonic", return_value=0.):
            client = api._MotionClient(arm, single=True)
            client.send(api.HOME, api.HOME)
        with patch.object(api.time, "monotonic", return_value=.201):
            with self.assertRaisesRegex(RuntimeError, "发送间隔"):
                client.send(api.HOME, api.HOME)
        client = api._MotionClient(arm, single=True)
        client.state_at -= .201
        with self.assertRaisesRegex(RuntimeError, "反馈"):
            client.send(api.HOME, api.HOME)
        self.assertEqual(len(arm.cli.sent), 1)

    def test_runtime_competitor_stops_before_next_publish(self):
        arm = self.arm()
        client = api._MotionClient(arm, single=True)
        client.graph_at -= .11
        arm.cli.publishers = 2
        with self.assertRaisesRegex(RuntimeError, "发布者"):
            client.send(api.HOME, api.HOME)
        self.assertEqual(arm.cli.sent, [])

    def test_parallel_motion_rejected(self):
        arm = self.arm()
        arm._motion_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                arm.move_j(api.HOME)
        finally:
            arm._motion_lock.release()
        self.assertEqual(arm.cli.sent, [])

    def test_explicit_left_right_methods_route_independently_of_default_side(self):
        for default_side in ("left", "right"):
            for method_name, active_side, index in (
                    ("L_move_J", "left", 0), ("R_move_J", "right", 1)):
                with self.subTest(default=default_side, method=method_name), self.arm(side=default_side) as arm:
                    target = api.HOME.copy()
                    target[0] += .08
                    result = getattr(arm, method_name)(target, duration=.04, settle=0)
                    np.testing.assert_array_equal(arm.cli.sent[-1][index], target)
                    for frame in arm.cli.sent:
                        np.testing.assert_array_equal(frame[1 - index], api.HOME)
                    expected = target + (.001 if active_side == "left" else 0.)
                    np.testing.assert_allclose(result["q"], expected)
                    np.testing.assert_allclose(result["err"], expected - target, atol=1e-14)
                    self.assertFalse(result["stale"])
                    self.assertEqual(arm.side, default_side)
                    self.assertEqual(arm.model.side, default_side)
                    arm.cli.set_action.assert_not_called()
                    arm.cli.enter_control.assert_not_called()

    def test_named_methods_default_home_while_move_j_keeps_constructor_side(self):
        for default_side in ("left", "right"):
            with self.subTest(default=default_side), self.arm(side=default_side) as arm:
                target = api.HOME.copy()
                target[0] += .06
                arm.move_j(target, duration=.02, settle=0)
                active = 0 if default_side == "left" else 1
                np.testing.assert_array_equal(arm.cli.sent[-1][active], target)
                named_method = arm.L_move_J if default_side == "left" else arm.R_move_J
                result = named_method(duration=.02, settle=0)
                np.testing.assert_array_equal(arm.cli.sent[-1][active], api.HOME)
                expected = api.HOME + (.001 if default_side == "left" else 0.)
                np.testing.assert_allclose(result["q"], expected)

    def test_right_left_right_sequence_never_reuses_drift_as_peer_input(self):
        for default_side in ("left", "right"):
            with self.subTest(default=default_side), self.arm(side=default_side) as arm:
                right_first, left_goal, right_final = (api.HOME.copy() for _ in range(3))
                right_first[0] += .08
                left_goal[0] += .12
                right_final[0] += .04
                arm.R_move_J(right_first, duration=.04, settle=0)
                after_right = len(arm.cli.sent)
                arm.cli.feedback["right"][0] -= .025
                arm.cli.feedback["left"][0] += .025
                arm.L_move_J(left_goal, duration=.04, settle=0)
                after_left = len(arm.cli.sent)
                arm.cli.feedback["left"][0] -= .018
                arm.R_move_J(right_final, duration=.04, settle=0)
                for left, _ in arm.cli.sent[:after_right]:
                    np.testing.assert_array_equal(left, api.HOME)
                for _, right in arm.cli.sent[after_right:after_left]:
                    np.testing.assert_array_equal(right, right_first)
                for left, _ in arm.cli.sent[after_left:]:
                    np.testing.assert_array_equal(left, left_goal)
                self.assertEqual(arm.side, default_side)

    def test_both_move_has_one_trajectory_with_shared_quintic_progress(self):
        arm = self.arm()
        left_start, right_start = arm.cli.q("left"), arm.cli.q("right")
        left_goal, right_goal = left_start.copy(), right_start.copy()
        left_goal[0] += .08
        right_goal[0] -= .16
        with patch.object(ros, "goto_joint", wraps=ros.goto_joint) as move:
            result = arm.move_j_both(left_goal, right_goal, duration=.1, settle=0)
        move.assert_called_once()
        self.assertEqual(set(result), {"left", "right"})
        self.assertEqual(len(arm.cli.sent), 6)  # 五个运动采样 + 一帧新反馈保持。
        progress = [.05792, .31744, .68256, .94208, 1.]
        for (left, right), expected in zip(arm.cli.sent[:5], progress):
            self.assertAlmostEqual((left[0] - left_start[0]) / .08, expected)
            self.assertAlmostEqual((right[0] - right_start[0]) / -.16, expected)
        np.testing.assert_array_equal(arm.cli.sent[-1][0], left_goal)
        np.testing.assert_array_equal(arm.cli.sent[-1][1], right_goal)
        self.assertEqual(self.factory_mock.call_count, 1)
        arm.cli.set_action.assert_not_called()
        arm.cli.enter_control.assert_not_called()

    def test_failed_named_motion_preserves_last_successfully_sent_input_for_peer(self):
        for failed_side, failed_method, peer_method, failed_index in (
                ("right", "R_move_J", "L_move_J", 1),
                ("left", "L_move_J", "R_move_J", 0)):
            default_side = "left" if failed_side == "right" else "right"
            with self.subTest(failed=failed_side), self.arm(side=default_side) as arm:
                target = api.HOME.copy()
                target[0] += .1
                attempts = []
                def fail_after_second_frame(*args):
                    attempts.append(True)
                    if len(attempts) == 2:
                        raise RuntimeError("injected pacing failure")
                with patch.object(ros, "_pace", side_effect=fail_after_second_frame):
                    with self.assertRaisesRegex(RuntimeError, "injected pacing failure"):
                        getattr(arm, failed_method)(target, duration=.1, settle=0)
                self.assertEqual(len(arm.cli.sent), 2)
                last_input = arm.cli.sent[-1][failed_index].copy()
                self.assertGreater(np.max(np.abs(last_input - api.HOME)), 0)
                self.assertGreater(np.max(np.abs(last_input - target)), 0)
                arm.cli.feedback[failed_side][0] += .02
                peer_target = api.HOME.copy()
                peer_target[0] += .015
                getattr(arm, peer_method)(peer_target, duration=.04, settle=0)
                for frame in arm.cli.sent[2:]:
                    np.testing.assert_array_equal(frame[failed_index], last_input)

    def test_trajectory_setup_failure_does_not_replace_previous_peer_hold(self):
        arm = self.arm()
        old_target, new_target = api.HOME.copy(), api.HOME.copy()
        old_target[0] += .04
        new_target[0] += .1
        arm.R_move_J(old_target, duration=.04, settle=0)
        sent_before = len(arm.cli.sent)
        with patch.object(ros, "goto_joint", side_effect=RuntimeError("trajectory setup failed")):
            with self.assertRaisesRegex(RuntimeError, "trajectory setup failed"):
                arm.R_move_J(new_target, duration=.04, settle=0)
        self.assertEqual(len(arm.cli.sent), sent_before)
        arm.L_move_J(api.HOME, duration=.04, settle=0)
        for _, right in arm.cli.sent[sent_before:]:
            np.testing.assert_array_equal(right, old_target)

    def test_post_publish_recording_error_blocks_all_motion_until_reconnected(self):
        arm = self.arm()
        client = arm.cli
        send = client.send
        failure = OSError("record flush failed after publish")
        def published_then_failed(*args):
            send(*args)
            raise failure
        target = api.HOME.copy()
        target[0] += .1
        with patch.object(client, "send", side_effect=published_then_failed):
            with self.assertRaises(OSError) as caught:
                arm.R_move_J(target, duration=.04, settle=0)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(client.sent), 1)  # 异常帧实际上已发布，不能假定未生效。
        client.get_action.reset_mock()
        client.fresh_state.reset_mock()
        pos, rpy = arm.fk(api.HOME)
        for operation in (
                lambda: arm.R_move_J(api.HOME), lambda: arm.L_move_J(api.HOME),
                lambda: arm.move_j_both(api.HOME, api.HOME),
                lambda: arm.move_j(api.HOME), lambda: arm.move_l(pos, rpy)):
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                operation()
        self.assertEqual(len(client.sent), 1)
        client.get_action.assert_not_called()
        client.fresh_state.assert_not_called()
        client.set_action.assert_not_called()
        arm.close()
        reconnected = self.arm()
        reconnected.R_move_J(api.HOME, duration=.02, settle=0)
        self.assertGreater(len(reconnected.cli.sent), 0)
        self.assertEqual(self.factory_mock.call_count, 2)

    def test_caller_mutating_target_arrays_does_not_change_remembered_peer_inputs(self):
        arm = self.arm()
        right_target, left_target = api.HOME.copy(), api.HOME.copy()
        right_target[0] += .06
        left_target[0] += .08
        right_expected, left_expected = right_target.copy(), left_target.copy()
        arm.R_move_J(right_target, duration=.04, settle=0)
        after_right = len(arm.cli.sent)
        right_target[0] += .2
        arm.L_move_J(left_target, duration=.04, settle=0)
        after_left = len(arm.cli.sent)
        left_target[0] += .2
        arm.R_move_J(api.HOME, duration=.04, settle=0)
        for _, right in arm.cli.sent[after_right:after_left]:
            np.testing.assert_array_equal(right, right_expected)
        for left, _ in arm.cli.sent[after_left:]:
            np.testing.assert_array_equal(left, left_expected)

    def test_named_methods_use_requested_arm_asymmetric_joint_limits(self):
        for default_side in ("left", "right"):
            with self.subTest(default=default_side), self.arm(side=default_side) as arm:
                left_only, right_only = api.HOME.copy(), api.HOME.copy()
                left_only[1], right_only[1] = .3, -.3
                arm.cli.get_action.reset_mock()
                with self.assertRaises(ValueError):
                    arm.R_move_J(left_only, duration=.04, settle=0)
                with self.assertRaises(ValueError):
                    arm.L_move_J(right_only, duration=.04, settle=0)
                self.assertEqual(arm.cli.sent, [])
                arm.cli.get_action.assert_not_called()
                arm.L_move_J(left_only, duration=.04, settle=0)
                arm.R_move_J(right_only, duration=.04, settle=0)
                np.testing.assert_array_equal(arm.cli.sent[-1][0], left_only)
                np.testing.assert_array_equal(arm.cli.sent[-1][1], right_only)

    def test_named_methods_reject_invalid_targets_and_timing_before_feedback_or_send(self):
        arm = self.arm()
        arm.cli.get_action.reset_mock()
        arm.cli.fresh_state.reset_mock()
        for method in (arm.R_move_J, arm.L_move_J):
            for target, options in (
                    ([0.] * 6, {}), ([float("nan")] * 7, {}),
                    ([float("inf")] * 7, {}), (api.HOME, {"duration": 0}),
                    (api.HOME, {"duration": float("nan")}),
                    (api.HOME, {"settle": -1}), (api.HOME, {"settle": float("inf")})):
                with self.subTest(method=method.__name__, options=options):
                    with self.assertRaises(ValueError):
                        method(target, **options)
        self.assertEqual(arm.cli.sent, [])
        arm.cli.get_action.assert_not_called()
        arm.cli.fresh_state.assert_not_called()

    def test_named_methods_enforce_urs_complete_feedback_and_exclusive_publisher(self):
        arm = self.arm()
        names = arm.cli.state_names.copy()
        for method in (arm.R_move_J, arm.L_move_J):
            for action in (None, "JOINT_DEFAULT", "STAND_DEFAULT"):
                arm.cli.get_action.return_value = action
                with self.subTest(method=method.__name__, action=action):
                    with self.assertRaises(RuntimeError):
                        method(api.HOME)
            arm.cli.get_action.return_value = "URS"
            arm.cli.publishers = 2
            with self.assertRaises(RuntimeError):
                method(api.HOME)
            arm.cli.publishers = 1
            arm.cli.state_names = names[:-1]
            with self.assertRaises(RuntimeError):
                method(api.HOME)
            arm.cli.state_names = names
        self.assertEqual(arm.cli.sent, [])
        arm.cli.set_action.assert_not_called()
        arm.cli.enter_control.assert_not_called()

    def test_named_methods_share_motion_lock_and_reject_concurrent_calls(self):
        arm = self.arm()
        arm.cli.get_action.reset_mock()
        arm._motion_lock.acquire()
        try:
            for method in (arm.R_move_J, arm.L_move_J):
                with self.subTest(method=method.__name__), self.assertRaises(RuntimeError):
                    method(api.HOME)
        finally:
            arm._motion_lock.release()
        self.assertEqual(arm.cli.sent, [])
        arm.cli.get_action.assert_not_called()

    def test_named_peer_motion_preserves_previous_cartesian_corrected_input(self):
        arm = self.arm(side="right")
        corrected = api.HOME.copy()
        corrected[0] += .025
        pos, rpy = arm.fk(api.HOME)
        def corrected_move(client, *args, **kwargs):
            client.send(api.HOME, corrected)
            return {"converged": True, "q_hold": corrected.copy()}
        with patch.object(ros, "goto_cartesian", side_effect=corrected_move):
            arm.move_l(pos, rpy, converge=8)
        arm.cli.feedback["right"][0] -= .02
        arm.L_move_J(api.HOME, duration=.04, settle=0)
        for _, right in arm.cli.sent:
            np.testing.assert_array_equal(right, corrected)


if __name__ == "__main__":
    unittest.main()
