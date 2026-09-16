"""MDI 的离线会话与发送门禁测试；不加载 ROS、不连接机器人。"""
import contextlib
from collections import deque
from concurrent.futures import Future
import io
import math
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_api as api
import x2_sim_ros as ros
import x2ik


class FakeClient:
    """只模拟缓存反馈和发布；保存调用线程以检查 IK worker 隔离。"""

    def __init__(self):
        self.mode = "upper_body"
        self.verbose = False
        self.rate = 50.
        self.recorder = None
        self.state_count = 1
        self.state_names = list(ros.DEFAULT_ARM_ORDER)
        self.models = {s: api.ArmModel(s) for s in ("left", "right")}
        self.iks = {s: api.SrsArmIK(m) for s, m in self.models.items()}
        self.feedback = {s: api.HOME.copy() for s in self.models}
        self.imu_count = dict(chest=1, pelvis=1)
        self.imu_has_orientation = dict(chest=True, pelvis=True)
        self.grav = SimpleNamespace(source="pelvis")
        self.k_eff = np.full(7, 40.)
        self.bias_limit = math.radians(12.)
        self.publishers = 1
        self.subscribers = 1
        self.sent = []
        self.ros_threads = []
        self.node = SimpleNamespace(
            count_publishers=lambda _: self.publishers,
            count_subscribers=lambda _: self.subscribers)
        self.rclpy = SimpleNamespace(spin_once=self.spin_once, ok=lambda: False,
                                    shutdown=Mock())
        self.get_action = Mock(return_value="UPPERBODY_REMOTE_SPLIT")
        self.wait_state = Mock(return_value=True)
        self.fresh_state = Mock(side_effect=self._fresh)
        self.set_action = Mock(side_effect=AssertionError("不允许状态切换"))
        self.enter_control = Mock(side_effect=AssertionError("不允许进入状态逻辑"))
        self.close = Mock()

    def _fresh(self, *args, **kwargs):
        self.spin_once()
        return True

    def spin_once(self, *args, **kwargs):
        self.ros_threads.append(threading.get_ident())
        self.state_count += 1
        self.imu_count["pelvis"] += 1
        self.imu_count["chest"] += 1

    def spin(self, seconds):
        self.spin_once()

    def q(self, side):
        return self.feedback[side].copy()

    def dq(self, side):
        return np.zeros(7)

    def tau(self, side):
        return np.zeros(7)

    def send(self, left, right, *args):
        self.sent.append(dict(left=np.asarray(left).copy(), right=np.asarray(right).copy(),
                              thread=threading.get_ident(), at=time.monotonic()))
        self.ros_threads.append(threading.get_ident())
        # 模拟重力补偿后的反馈偏差，保持指令不能反复以此作为新原始输入。
        self.feedback["left"] = np.asarray(left).copy() + .001
        self.feedback["right"] = np.asarray(right).copy() + .001
        self.spin_once()


class RoutingTests(unittest.TestCase):
    def test_top_level_mdi_alias_and_ros_mdi_forward_the_same_arguments(self):
        context = object()
        options = ["--dry", "--side", "left", "--no-home-first"]
        with patch.object(x2ik, "build_context", return_value=context), \
                patch.object(x2ik, "cmd_ros", return_value=17) as dispatch:
            self.assertEqual(x2ik.main(["mdi", *options]), 17)
            dispatch.assert_called_with(["mdi", *options], context)
            self.assertEqual(x2ik.main(["ros", "mdi", *options]), 17)
            dispatch.assert_called_with(["mdi", *options], context)
            self.assertEqual(dispatch.call_count, 2)

    def test_mdi_parser_preserves_defaults_and_accepts_startup_dry(self):
        client = SimpleNamespace(recorder=None, close=Mock(),
                                 rclpy=SimpleNamespace(ok=lambda: False))
        with patch.dict(sys.modules, {"rclpy": ModuleType("rclpy"),
                                      "aimdk_msgs": ModuleType("aimdk_msgs")}), \
                patch.object(ros, "X2ArmClient", return_value=client), \
                patch.object(ros, "load_calibration", return_value=None), \
                patch.object(ros, "cmd_mdi", return_value=0) as dispatch, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ros.main(["mdi"]), 0)
            default = dispatch.call_args.args[1]
            self.assertTrue(default.home_first)
            self.assertFalse(default.dry)
            self.assertEqual(default.side, "right")
            self.assertFalse(getattr(default, "converge", 0))
            self.assertEqual(ros.main(["mdi", "--dry", "--no-home-first", "--side", "left"]), 0)
            selected = dispatch.call_args.args[1]
            self.assertTrue(selected.dry)
            self.assertFalse(selected.home_first)
            self.assertEqual(selected.side, "left")

    def test_mdi_invalid_payload_is_rejected_before_ros_client_creation(self):
        with patch.object(ros, "X2ArmClient") as factory, \
                patch.object(ros, "cmd_mdi") as dispatch, \
                contextlib.redirect_stderr(io.StringIO()):
            for value in ("nan", "inf", "-1"):
                with self.subTest(payload=value), self.assertRaises(SystemExit) as caught:
                    ros.main(["--payload", value, "mdi", "--dry"])
                self.assertEqual(caught.exception.code, 2)
        factory.assert_not_called()
        dispatch.assert_not_called()

    def test_mdi_main_recording_failure_still_releases_client_and_ros_context(self):
        for fail_client_close in (False, True):
            with self.subTest(fail_client_close=fail_client_close):
                recording_error = OSError("record flush failed")
                client_error = RuntimeError("node destruction failed")
                recorder = SimpleNamespace(close=Mock(side_effect=recording_error))
                client = SimpleNamespace(
                    recorder=None, close=Mock(side_effect=client_error if fail_client_close else None),
                    rclpy=SimpleNamespace(ok=lambda: True, shutdown=Mock()))
                def mdi_callback(cli, args):
                    # 模拟会话使用借入客户端开启录制；最终释放归 ros.main。
                    cli.recorder = recorder
                    return 0
                expected = client_error if fail_client_close else recording_error
                with patch.dict(sys.modules, {"rclpy": ModuleType("rclpy"),
                                              "aimdk_msgs": ModuleType("aimdk_msgs")}), \
                        patch.object(ros, "X2ArmClient", return_value=client), \
                        patch.object(ros, "load_calibration", return_value=None), \
                        patch.object(ros, "cmd_mdi", side_effect=mdi_callback), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(type(expected)) as caught:
                        ros.main(["mdi", "--dry"])
                self.assertIs(caught.exception, expected)
                if fail_client_close:
                    self.assertIs(caught.exception.__context__, recording_error)
                recorder.close.assert_called_once()
                client.close.assert_called_once()
                client.rclpy.shutdown.assert_called_once()


class FakeActionQuery:
    def __init__(self, client, service_name=None):
        self.client = client

    def start(self):
        pass

    def poll(self):
        return True, self.client.get_action()

    def close(self):
        pass


class ActionQueryTests(unittest.TestCase):
    def setUp(self):
        import x2_mdi
        self.mdi = x2_mdi
        self.stamp = object()
        self.future = Future()
        self.service = SimpleNamespace(service_is_ready=Mock(return_value=False),
                                       call_async=Mock(return_value=self.future))
        self.request_type = SimpleNamespace(Request=lambda: SimpleNamespace(
            request=SimpleNamespace(header=SimpleNamespace(stamp=None))))
        self.service_module = ModuleType("aimdk_msgs.srv")
        self.service_module.GetMcAction = self.request_type
        self.modules = patch.dict(sys.modules, {"aimdk_msgs.srv": self.service_module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.client = SimpleNamespace(
            node=SimpleNamespace(
                create_client=Mock(return_value=self.service), destroy_client=Mock(),
                get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: self.stamp))),
            send=Mock(side_effect=AssertionError("GetMcAction must not send motion")),
            spin=Mock(side_effect=AssertionError("poll must not spin")),
            rclpy=SimpleNamespace(spin_once=Mock(side_effect=AssertionError("poll must not spin"))),
            _action_name=Mock(return_value="UPPERBODY_REMOTE_SPLIT(19)"))
        self.query = self.mdi.ActionQuery(self.client, "/test/get_mc_action")
        self.addCleanup(self.query.close)

    def assert_no_motion_or_spin(self):
        self.client.send.assert_not_called()
        self.client.spin.assert_not_called()
        self.client.rclpy.spin_once.assert_not_called()

    def test_not_ready_does_not_call_service_and_pending_poll_is_nonblocking(self):
        self.query.start()
        self.assertEqual(self.query.poll(), (False, None))
        self.service.call_async.assert_not_called()
        self.service.service_is_ready.return_value = True
        self.assertEqual(self.query.poll(), (False, None))
        self.assertEqual(self.query.poll(), (False, None))
        self.service.call_async.assert_called_once()
        request = self.service.call_async.call_args.args[0]
        self.assertIs(request.request.header.stamp, self.stamp)
        self.client.node.create_client.assert_called_once_with(self.request_type, "/test/get_mc_action")
        self.assert_no_motion_or_spin()

    def test_ready_response_prefers_action_description(self):
        self.service.service_is_ready.return_value = True
        self.future.set_result(SimpleNamespace(info=SimpleNamespace(
            action_desc="UPPERBODY_REMOTE_SPLIT", current_action=SimpleNamespace(value=19))))
        self.assertEqual(self.query.poll(), (True, "UPPERBODY_REMOTE_SPLIT"))
        self.client._action_name.assert_not_called()
        self.assert_no_motion_or_spin()

    def test_legacy_request_header_and_enum_only_response_are_supported(self):
        self.request_type.Request = lambda: SimpleNamespace(header=SimpleNamespace(stamp=None))
        self.service.service_is_ready.return_value = True
        self.future.set_result(SimpleNamespace(info=SimpleNamespace(
            action_desc="", current_action=SimpleNamespace(value=19))))
        self.assertEqual(self.query.poll(), (True, "UPPERBODY_REMOTE_SPLIT(19)"))
        self.client._action_name.assert_called_once_with(19)
        request = self.service.call_async.call_args.args[0]
        self.assertIs(request.header.stamp, self.stamp)
        self.assert_no_motion_or_spin()

    def test_missing_response_does_not_invent_a_state(self):
        self.service.service_is_ready.return_value = True
        self.future.set_result(None)
        self.assertEqual(self.query.poll(), (True, None))
        self.client._action_name.assert_not_called()

    def test_close_cancels_pending_future_and_only_destroys_its_service_once(self):
        self.service.service_is_ready.return_value = True
        self.assertEqual(self.query.poll(), (False, None))
        self.query.close()
        self.assertTrue(self.future.cancelled())
        self.client.node.destroy_client.assert_called_once_with(self.service)
        self.query.close()
        self.client.node.destroy_client.assert_called_once_with(self.service)
        self.assert_no_motion_or_spin()


class WorkerLifecycleTests(unittest.TestCase):
    @staticmethod
    def submit(worker):
        return worker.submit("right", api.HOME, np.zeros(3), np.eye(3), 0., False)

    def test_close_does_not_wait_for_running_solver_and_cancels_queued_work(self):
        import x2_mdi
        started, release = threading.Event(), threading.Event()
        close_started, closed = threading.Event(), threading.Event()
        def blocked_solve(*args):
            started.set()
            release.wait(3.)
            return "finished"
        worker = x2_mdi.PureIKWorker({"right": object()}, lambda model: object())
        self.addCleanup(worker.close)
        with patch.object(x2_mdi, "_solve_pose", side_effect=blocked_solve):
            first = self.submit(worker)
            self.assertTrue(started.wait(1.), "worker 没开始计算")
            second = self.submit(worker)
            def close_worker():
                close_started.set()
                worker.close()
                closed.set()
            closer = threading.Thread(target=close_worker, daemon=True)
            closer.start()
            try:
                self.assertTrue(close_started.wait(1.))
                self.assertTrue(closed.wait(.5), "close 等待了运行中的 solver")
                self.assertFalse(first.done())
                self.assertTrue(second.cancelled())
                with self.assertRaises(RuntimeError):
                    self.submit(worker)
            finally:
                release.set()
                closer.join(1.)
            self.assertEqual(first.result(timeout=1.), "finished")

    def test_worker_exception_is_returned_through_future(self):
        import x2_mdi
        worker = x2_mdi.PureIKWorker({"right": object()}, lambda model: object())
        self.addCleanup(worker.close)
        failure = ValueError("bad IK model")
        with patch.object(x2_mdi, "_solve_pose", side_effect=failure):
            future = self.submit(worker)
            with self.assertRaises(ValueError) as caught:
                future.result(timeout=1.)
        self.assertIs(caught.exception, failure)

    def test_process_exits_naturally_with_a_stuck_solver_after_worker_close(self):
        script = """
import threading
import numpy as np
import x2_mdi
started = threading.Event()
never_released = threading.Event()
def stuck_solver(*args):
    started.set()
    never_released.wait()
x2_mdi._solve_pose = stuck_solver
worker = x2_mdi.PureIKWorker({'right': object()}, lambda model: object())
worker.submit('right', np.zeros(7), np.zeros(3), np.eye(3), 0., False)
assert started.wait(1.), 'solver did not start'
worker.close()
print('closed-with-running-solver', flush=True)
"""
        result = subprocess.run([sys.executable, "-c", script],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=5., check=True)
        self.assertEqual(result.stdout.strip(), "closed-with-running-solver")


class MdiTests(unittest.TestCase):
    def setUp(self):
        import x2_mdi
        self.mdi = x2_mdi
        self.client = FakeClient()
        self.action_patch = patch.object(self.mdi, "ActionQuery", FakeActionQuery)
        self.action_patch.start()
        self.addCleanup(self.action_patch.stop)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stderr.__enter__()
        self.addCleanup(self.stderr.__exit__, None, None, None)
        self.target_patch = patch.object(ros, "publish_target")
        self.target_patch.start()
        self.addCleanup(self.target_patch.stop)

    @staticmethod
    def args(**overrides):
        options = dict(side="right", duration=.04, settle=0., relax=20.,
                       home_first=False, dry=False)
        options.update(overrides)
        return SimpleNamespace(**options)

    def session(self, **overrides):
        session = self.mdi.Session(self.client, self.args(**overrides), ros)
        self.addCleanup(session.close)
        session.initialize()
        return session

    def run_lines(self, lines, **overrides):
        queue = deque(lines)
        client = self.client
        class ScriptedInput:
            def __init__(self, *args, **kwargs):
                pass
            def start(self):
                pass
            def close(self):
                pass
            def poll(self):
                item = queue.popleft() if queue else "q!"
                return item(client) if callable(item) else item
        with patch.object(self.mdi, "InputLines", ScriptedInput):
            return self.mdi.run(client, self.args(**overrides), ros)

    def test_startup_dry_suppresses_home_joint_quit_and_waiting_commands(self):
        result = self.run_lines([None, "j 30 0 0 -68 0 0 0", None,
                                 "home", None, "q"], dry=True, home_first=True)
        self.assertEqual(result, 0)
        self.assertEqual(self.client.sent, [])
        self.client.set_action.assert_not_called()
        self.client.enter_control.assert_not_called()
        self.client.close.assert_not_called()

    def test_legacy_cmd_mdi_delegates_same_client_and_arguments_to_new_session(self):
        args = self.args(dry=True)
        with patch.object(self.mdi, "run", return_value=7) as run:
            self.assertEqual(ros.cmd_mdi(self.client, args), 7)
        run.assert_called_once_with(self.client, args, ros)

    def test_turning_dry_off_does_not_replay_previewed_joint_or_home(self):
        result = self.run_lines(["j 50 0 0 -68 0 0 0", "home", "dry",
                                 None, None, "q!"], dry=True, home_first=True)
        self.assertEqual(result, 0)
        self.assertGreaterEqual(len(self.client.sent), 2)
        for frame in self.client.sent:
            np.testing.assert_array_equal(frame["left"], api.HOME)
            np.testing.assert_array_equal(frame["right"], api.HOME)

    def test_runtime_dry_disables_all_holds_until_explicitly_resumed(self):
        session = self.session()
        session.hold_tick()
        before = len(self.client.sent)
        session.set_dry(True)
        session.hold_tick()
        session.hold_tick()
        self.assertEqual(len(self.client.sent), before)
        session.set_dry(False)
        session.hold_tick()
        self.assertGreater(len(self.client.sent), before)

    def test_quit_without_home_and_normal_quit_have_distinct_motion_semantics(self):
        with patch.object(self.mdi.Session, "move_joint") as move:
            self.assertEqual(self.run_lines(["q!"], home_first=False), 0)
            move.assert_not_called()
            self.assertEqual(self.run_lines(["q"], home_first=False), 0)
            move.assert_called_once()
            targets = move.call_args.args[0]
            self.assertEqual(set(targets), {"left", "right"})
            np.testing.assert_array_equal(targets["left"], api.HOME)
            np.testing.assert_array_equal(targets["right"], api.HOME)

    def test_keyboard_interrupt_or_command_failure_exits_without_automatic_home(self):
        for error in (KeyboardInterrupt(), RuntimeError("motion rejected")):
            with self.subTest(error=type(error).__name__), \
                    patch.object(self.mdi.Session, "move_joint", side_effect=error) as move:
                result = self.run_lines(["j 30 0 0 -68 0 0 0", "q"])
                self.assertNotEqual(result, 0)
                self.assertEqual(move.call_count, 1)
        self.client.close.assert_not_called()

    def test_side_help_units_and_feedback_queries_preserve_raw_hold_input(self):
        def drift_feedback(client):
            for side in ("left", "right"):
                client.feedback[side][0] += .02
            return None
        result = self.run_lines([None, "side left", drift_feedback, "help",
                                 drift_feedback, "mm", drift_feedback, "where",
                                 None, "q!"])
        self.assertEqual(result, 0)
        self.assertGreaterEqual(len(self.client.sent), 4)
        for frame in self.client.sent:
            np.testing.assert_array_equal(frame["left"], api.HOME)
            np.testing.assert_array_equal(frame["right"], api.HOME)

    def test_joint_command_selects_requested_side_and_preserves_peer(self):
        result = self.run_lines(["j 30 0 0 -68 0 0 0", "side left",
                                 "j 35 0 0 -68 0 0 0", "q!"])
        self.assertEqual(result, 0)
        expected_right = np.radians([30, 0, 0, -68, 0, 0, 0])
        expected_left = np.radians([35, 0, 0, -68, 0, 0, 0])
        np.testing.assert_allclose(self.client.sent[-1]["right"], expected_right)
        np.testing.assert_allclose(self.client.sent[-1]["left"], expected_left)
        self.client.set_action.assert_not_called()
        self.client.enter_control.assert_not_called()

    def test_joint_invalid_limits_or_nonfinite_values_never_execute_trajectory(self):
        with patch.object(ros, "goto_joint") as move:
            self.run_lines(["j 30 90 0 -68 0 0 0", "j nan 0 0 -68 0 0 0",
                            "side left", "j 30 -90 0 -68 0 0 0", "q!"])
        move.assert_not_called()

    def test_zero_argument_commands_reject_extra_tokens_without_motion_or_mode_change(self):
        commands = [f"{command} extra" for command in (
            "q", "quit", "exit", "q!", "quit!", "?", "h", "help",
            "w", "where", "mm", "dry", "home")]
        pos, rot = self.client.models["right"].forward_kinematics(api.HOME)
        with patch.object(self.mdi.Session, "move_joint") as move, \
                patch.object(self.mdi.Session, "set_dry") as change_dry, \
                patch.object(self.mdi.Session, "plan_pose", return_value=self.plan(rot, ok=False)) as plan:
            self.assertEqual(self.run_lines([*commands, "0.1 -0.2 0.3", "q!"]), 0)
        move.assert_not_called()
        change_dry.assert_not_called()
        plan.assert_called_once()
        np.testing.assert_allclose(plan.call_args.args[1], [.1, -.2, .3])

    def test_initial_preflight_rejects_unknown_state_feedback_imu_and_contention(self):
        for fault in ("unknown", "non_urs", "incomplete", "imu", "competition"):
            with self.subTest(fault=fault):
                self.client = FakeClient()
                if fault == "unknown":
                    self.client.get_action.return_value = None
                elif fault == "non_urs":
                    self.client.get_action.return_value = "JOINT_DEFAULT"
                elif fault == "incomplete":
                    self.client.state_names.pop()
                elif fault == "imu":
                    self.client.imu_has_orientation["pelvis"] = False
                else:
                    self.client.publishers = 2
                session = self.mdi.Session(self.client, self.args(), ros)
                self.addCleanup(session.close)
                with self.assertRaises((ValueError, RuntimeError)):
                    session.initialize()
                self.assertEqual(self.client.sent, [])
                self.client.set_action.assert_not_called()
                self.client.enter_control.assert_not_called()

    def test_invalid_compensation_configuration_rejected_before_any_send(self):
        faults = [("k_eff", np.zeros(7)), ("k_eff", np.full(7, math.nan)),
                  ("k_eff", np.ones(6)), ("bias_limit", -.1),
                  ("bias_limit", math.inf), ("bias_limit", math.pi + .01),
                  ("gravity_source", "unknown")]
        for field, value in faults:
            with self.subTest(field=field, value=value):
                self.client = FakeClient()
                if field == "gravity_source":
                    self.client.grav.source = value
                else:
                    setattr(self.client, field, value)
                session = self.mdi.Session(self.client, self.args(), ros)
                self.addCleanup(session.close)
                with self.assertRaises((ValueError, RuntimeError)):
                    session.initialize()
                self.assertEqual(self.client.sent, [])
                self.client.set_action.assert_not_called()
                self.client.enter_control.assert_not_called()

    def test_session_only_borrows_existing_client_and_does_not_create_or_close_another(self):
        with patch.object(ros, "X2ArmClient", side_effect=AssertionError("second online client")):
            session = self.session()
            session.close()
        self.client.close.assert_not_called()
        self.client.rclpy.shutdown.assert_not_called()

    def test_uncertain_publication_cannot_resume_holding(self):
        session = self.session()
        send = self.client.send
        def after_publish(*args):
            send(*args)
            raise OSError("recorder failed after publish")
        with patch.object(self.client, "send", side_effect=after_publish):
            with self.assertRaises(OSError):
                session.hold_tick()
        count = len(self.client.sent)
        with self.assertRaises(RuntimeError):
            session.hold_tick()
        self.assertEqual(len(self.client.sent), count)

    def test_ik_worker_keeps_main_thread_publishing_and_never_calls_ros_itself(self):
        session = self.session()
        pos, rot = self.client.models["right"].forward_kinematics(api.HOME)
        solve = self.mdi._solve_pose
        worker_threads = []
        def delayed_solve(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            time.sleep(.07)
            return solve(*args, **kwargs)
        before = len(self.client.sent)
        with patch.object(self.mdi, "_solve_pose", side_effect=delayed_solve):
            session.plan_pose("right", pos, rot, allow_relax=True)
        frames = self.client.sent[before:]
        self.assertGreaterEqual(len(frames), 3)
        self.assertTrue(worker_threads)
        self.assertNotIn(threading.get_ident(), worker_threads)
        self.assertEqual(set(self.client.ros_threads), {threading.get_ident()})
        for frame in frames:
            np.testing.assert_array_equal(frame["left"], api.HOME)
            np.testing.assert_array_equal(frame["right"], api.HOME)
        gaps = np.diff([frame["at"] for frame in frames])
        self.assertLess(float(gaps.max()), .2)
        self.assertGreater(float(gaps.min()), .005)  # 不能在等待任务时无节拍地突发补帧。

    def test_dry_ik_worker_never_sends_even_while_computation_is_pending(self):
        session = self.session(dry=True)
        pos, rot = self.client.models["right"].forward_kinematics(api.HOME)
        solve = self.mdi._solve_pose
        def delayed_solve(*args, **kwargs):
            time.sleep(.04)
            return solve(*args, **kwargs)
        with patch.object(self.mdi, "_solve_pose", side_effect=delayed_solve):
            session.plan_pose("right", pos, rot, allow_relax=True)
        self.assertEqual(self.client.sent, [])

    def test_expired_send_interval_rejected_before_next_publication(self):
        session = self.session()
        session.hold_tick()
        count = len(self.client.sent)
        later = time.monotonic() + .21
        with patch.object(self.mdi.time, "monotonic", return_value=later):
            with self.assertRaises(RuntimeError):
                session.hold_tick()
        self.assertEqual(len(self.client.sent), count)

    @staticmethod
    def plan(rot, *, ok=True, clipped=False):
        return dict(ok=ok, clipped=clipped, clip_mm=10. if clipped else 0.,
                    q=api.HOME.copy() if ok else None, margin=.1,
                    pos_err=0., rot_err=0., rot=rot.copy(), relaxation=None)

    def test_unsolved_or_clipped_pose_never_starts_a_cartesian_trajectory(self):
        pos, rot = self.client.models["right"].forward_kinematics(api.HOME)
        command = " ".join(str(float(x)) for x in pos)
        for ok, clipped in ((False, False), (True, True)):
            with self.subTest(ok=ok, clipped=clipped), \
                    patch.object(self.mdi.Session, "plan_pose",
                                 return_value=self.plan(rot, ok=ok, clipped=clipped)), \
                    patch.object(ros, "goto_cartesian") as move:
                self.assertEqual(self.run_lines([command, "q!"]), 0)
                move.assert_not_called()
        self.assertEqual(self.client.sent, [])

    def test_invalid_or_stale_trajectory_ends_session_without_quit_home(self):
        pos, rot = self.client.models["right"].forward_kinematics(api.HOME)
        command = " ".join(str(float(x)) for x in pos)
        for failure in (dict(stale=True), dict(trajectory_valid=False),
                        dict(step_rejects=1), dict(clipped=1), dict(pos_err=math.nan)):
            result = dict(stale=False, trajectory_valid=True, aborted=False,
                          ik_fails=0, clipped=0, step_rejects=0,
                          branch_rejects=0, implicit_fallbacks=0, pos_err=0., rot_err=0.)
            result.update(failure)
            with self.subTest(failure=failure), \
                    patch.object(self.mdi.Session, "plan_pose", return_value=self.plan(rot)), \
                    patch.object(ros, "goto_cartesian", return_value=result) as move, \
                    patch.object(self.mdi.Session, "move_joint") as home:
                self.assertNotEqual(self.run_lines([command, "q"]), 0)
                move.assert_called_once()
                home.assert_not_called()
        self.client.set_action.assert_not_called()

    def test_relative_commands_keep_units_rotation_frames_and_relaxation_contract(self):
        pos, rot = self.client.models["right"].forward_kinematics(api.HOME)
        requested = []
        def record_plan(side, target_pos, target_rot, allow_relax=False):
            requested.append((side, target_pos.copy(), target_rot.copy(), allow_relax))
            return self.plan(target_rot, ok=False)
        with patch.object(self.mdi.Session, "plan_pose", side_effect=record_plan):
            result = self.run_lines(["mm", "d 10 20 -5", "R 10 0 0", "t 0 15 0",
                                     "rpy 0 0 20", "100 -200 300",
                                     "100 -200 300 0 0 20", "q!"])
        self.assertEqual(result, 0)
        self.assertEqual(len(requested), 6)
        np.testing.assert_allclose(requested[0][1], pos + [.01, .02, -.005])
        np.testing.assert_allclose(requested[0][2], rot)
        np.testing.assert_allclose(requested[1][2], ros.rpy_to_matrix(np.radians([10, 0, 0])) @ rot)
        np.testing.assert_allclose(requested[2][2], rot @ ros.rpy_to_matrix(np.radians([0, 15, 0])))
        np.testing.assert_allclose(requested[3][2], ros.rpy_to_matrix(np.radians([0, 0, 20])))
        np.testing.assert_allclose(requested[4][1], [.1, -.2, .3])
        np.testing.assert_allclose(requested[5][1], [.1, -.2, .3])
        self.assertEqual([entry[3] for entry in requested], [True, False, False, False, True, False])

    def test_orientation_relaxation_runs_in_worker_while_main_thread_keeps_holding(self):
        session = self.session()
        pos, rot = self.client.models["right"].forward_kinematics(api.HOME)
        ik = session.worker.iks["right"]
        threads = []
        def relaxed_solution(*args, **kwargs):
            threads.append(threading.get_ident())
            time.sleep(.07)
            return SimpleNamespace(rot=rot, axis=1, frame="torso", deviation=.02,
                                   sol=SimpleNamespace(q=api.HOME, pos_error=0., rot_error=0.))
        with patch.object(ik, "solve", return_value=None), \
                patch.object(ik, "solve_hold_rotation", side_effect=relaxed_solution):
            plan = session.plan_pose("right", pos, rot, allow_relax=True)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["relaxation"]["axis"], 1)
        self.assertGreaterEqual(len(self.client.sent), 3)
        self.assertNotIn(threading.get_ident(), threads)
        self.assertEqual(set(self.client.ros_threads), {threading.get_ident()})


if __name__ == "__main__":
    unittest.main()
