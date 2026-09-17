"""TCP selection is validated before connection and does not change joint motion."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np

import x2_api as api
import x2_mdi_bridge as bridge_module
import x2_movej
import x2_sim_ros as ros
from x2_arm_dynamics import ArmDynamics
from x2_arm_model import ArmModel, rpy_to_matrix
from x2_tcp import TCP_MODES, load_tcp_tools
from test_x2_api import FakeClient


class TcpInterfaceTests(unittest.TestCase):
    def setUp(self):
        api.X2Arm._connected = None
        temp = tempfile.TemporaryDirectory(prefix="x2ik-tcp-test-")
        self.addCleanup(temp.cleanup)
        self.custom_file = Path(temp.name) / "tool.json"
        tools = load_tcp_tools()
        for side, sign in (("left", 1), ("right", -1)):
            tools[side].pop("mode")
            tools[side].update(name=f"{side} custom TCP",
                               translation_m=[0.027, sign * 0.032, -0.145],
                               rotation_matrix=rpy_to_matrix([0.2, -0.3, sign * 0.4]).tolist())
        self.custom_file.write_text(json.dumps(dict(schema_version=1, units="m", tools=tools)))

    def robot(self, **kwargs):
        def client_factory(*args, **options):
            client = FakeClient(*args, **options)
            client.dyns = {s: ArmDynamics(m) for s, m in client.models.items()}
            return client
        factory = patch.object(ros, "X2ArmClient", side_effect=client_factory)
        factory.start()
        self.addCleanup(factory.stop)
        conf = patch("x2ik.load_conf")
        conf.start()
        self.addCleanup(conf.stop)
        pace = patch.object(ros, "_pace")
        pace.start()
        self.addCleanup(pace.stop)
        robot = x2_movej.Robot(verbose=False, **kwargs)
        self.addCleanup(robot.close)
        return robot

    @staticmethod
    def goto_joint(client, goals, duration, settle):
        client.send(goals["left"], goals["right"])
        return {s: {} for s in goals}

    def test_public_constructor_rejects_custom_before_creating_client(self):
        invalid = (dict(tcp_mode="custom"), dict(tcp_mode="wrong"),
                   dict(tcp_mode="none", tcp_file=str(self.custom_file)),
                   dict(tcp_mode="custom", tcp_file=str(self.custom_file) + ".missing"))
        with patch.object(x2_movej, "_X2Arm") as factory:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, OSError)):
                    x2_movej.Robot(**kwargs)
            factory.assert_not_called()

    def test_public_tool_changes_report_feedback_tcp_without_changing_joint_target(self):
        robot = self.robot()
        original_dynamics = dict(robot._arm.cli.dyns)
        q = api.HOME.copy()
        with patch.object(ros, "goto_joint", side_effect=self.goto_joint):
            for mode in TCP_MODES:
                options = dict(tcp_mode=mode)
                if mode == "custom":
                    options["tcp_file"] = str(self.custom_file)
                result = robot.moveJ(q, side="right", **options)
                expected_tools = load_tcp_tools(mode, options.get("tcp_file"))
                self.assertEqual(result["tcp"], expected_tools["right"])
                model = robot._arm.models["right"]
                pos, rot = model.forward_kinematics(result["q"])
                np.testing.assert_allclose(result["position_m"], pos, atol=1e-14)
                np.testing.assert_allclose(result["rotation_matrix"], rot, atol=1e-14)
                self.assertEqual(result["pose_frame"], "torso_link")
                np.testing.assert_array_equal(robot._arm.cli.sent[-1][1], q)
                for side in ("left", "right"):
                    self.assertIs(robot._arm.cli.dyns[side], original_dynamics[side])
                    np.testing.assert_array_equal(original_dynamics[side].model.tcp_offset,
                                                  np.zeros(3))
                    np.testing.assert_array_equal(original_dynamics[side].model.tcp_rotation,
                                                  np.eye(3))
            # Selection persists even if the original custom file is later absent.
            self.custom_file.unlink()
            result = robot.moveJ(q_left=q, q_right=q)
            self.assertEqual(result["left"]["tcp"]["mode"], "custom")
            self.assertEqual(result["right"]["tcp"]["mode"], "custom")

    def test_invalid_tool_change_does_not_mutate_models_or_send(self):
        robot = self.robot(tcp_mode="gripper")
        models = robot._arm.models
        with self.assertRaises(ValueError):
            robot.moveJ(tcp_mode="none", tcp_file=str(self.custom_file))
        self.assertIs(robot._arm.models, models)
        self.assertEqual(robot._arm.cli.sent, [])
        invalid = load_tcp_tools()
        invalid["right"]["rotation_matrix"] = np.zeros((3, 3)).tolist()
        with self.assertRaises(ValueError):
            robot._arm._set_tcp_tools(invalid)
        self.assertIs(robot._arm.models, models)

    def test_failed_move_with_valid_tcp_restores_previous_selection(self):
        robot = self.robot(tcp_mode="gripper")
        original = load_tcp_tools("gripper")
        for kwargs in (dict(q=[1., 2.]), dict(duration=-1), dict(q=[99.] * 7)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                robot.moveJ(tcp_mode="custom", tcp_file=str(self.custom_file), **kwargs)
            self.assertEqual(robot._tcp_mode, "gripper")
            self.assertIsNone(robot._tcp_file)
            self.assertEqual(robot._arm._tcp_tools, original)
            self.assertEqual(robot._arm.cli.sent, [])
        robot._arm.cli.get_action.return_value = "STAND"
        with self.assertRaises(RuntimeError):
            robot.moveJ(tcp_mode="none")
        self.assertEqual(robot._arm._tcp_tools, original)
        self.assertEqual(robot._tcp_mode, "gripper")

    def test_mid_motion_failure_restores_tcp_without_recovery_command(self):
        robot = self.robot(tcp_mode="custom", tcp_file=str(self.custom_file))
        original = load_tcp_tools("custom", self.custom_file)
        def partial_failure(client, goals, duration, settle):
            client.send(goals["left"], goals["right"])
            raise RuntimeError("test feedback failure")
        with patch.object(ros, "goto_joint", side_effect=partial_failure):
            with self.assertRaisesRegex(RuntimeError, "test feedback failure"):
                robot.moveJ(tcp_mode="gripper")
        self.assertEqual(len(robot._arm.cli.sent), 1)
        self.assertEqual(robot._arm._tcp_tools, original)
        self.assertEqual(robot._tcp_mode, "custom")
        self.assertEqual(robot._tcp_file, str(self.custom_file))
        # No dependency on rereading the old file when rolling back or reusing it.
        self.custom_file.unlink()
        with patch.object(ros, "goto_joint", side_effect=self.goto_joint):
            result = robot.moveJ()
        self.assertEqual(result["tcp"], original["right"])

    def test_concurrent_movej_or_tcp_change_is_rejected(self):
        robot = self.robot()
        models = robot._arm.models
        with robot._call_lock:
            with self.assertRaisesRegex(RuntimeError, "顺序调用"):
                robot.moveJ(tcp_mode="gripper")
        with robot._arm._motion_lock:
            with self.assertRaisesRegex(RuntimeError, "运动期间"):
                robot._arm._set_tcp_tools(load_tcp_tools("gripper"))
        self.assertIs(robot._arm.models, models)
        self.assertEqual(robot._arm.cli.sent, [])

    def test_bridge_all_modes_publish_tcp_and_physical_link_transforms(self):
        transport = bridge_module.Transport(io.StringIO(), io.StringIO())
        for mode in TCP_MODES:
            path = str(self.custom_file) if mode == "custom" else None
            bridge = bridge_module.Bridge(transport, demo=True, tcp_mode=mode, tcp_file=path)
            snapshot = bridge.snapshot()
            # Desktop Window.tcp_matches gates on this top-level session field,
            # independently of per-arm metadata. The real bridge must supply both.
            self.assertEqual(snapshot["tcp_mode"], mode)
            for side, state in snapshot["arms"].items():
                self.assertEqual(state["tcp"]["mode"], mode)
                self.assertEqual(state["tcp"]["mode"], snapshot["tcp_mode"])
                bare = ArmModel(side)
                positions, rotations = bare.joint_frames(bridge.q(side))
                self.assertEqual(len(state["link_transforms"]), 7)
                for idx, frame in enumerate(state["link_transforms"]):
                    np.testing.assert_allclose(frame["xyz"], positions[idx], atol=1e-14)
                    np.testing.assert_allclose(frame["rotation"], rotations[idx], atol=1e-14)
                expected = bridge.models[side].forward_kinematics(bridge.q(side))[0]
                np.testing.assert_allclose(state["xyz"], expected, atol=1e-14)
                result = bridge.execute(dict(id=1, op="mdi", side=side, mode="xyz",
                                              values=state["xyz"], preview=True))
                self.assertIn("预检通过", result)
            self.assertFalse(bridge.armed)

    def test_bridge_rejects_invalid_tool_before_connect_and_in_session_changes(self):
        transport = bridge_module.Transport(io.StringIO(), io.StringIO())
        with patch.object(bridge_module.Bridge, "_connect") as connect:
            with self.assertRaises(ValueError):
                bridge_module.Bridge(transport, tcp_mode="custom")
            connect.assert_not_called()
        for request in (dict(id=1, op="tcp", mode="gripper"),
                        dict(id=2, op="state", tcp_mode="gripper"),
                        dict(id=3, op="home", tcp_file="tools.json")):
            with self.assertRaises(ValueError):
                bridge_module.validate_request(request)

    def test_bridge_pose_plan_targets_custom_tcp(self):
        bridge = bridge_module.Bridge(bridge_module.Transport(io.StringIO(), io.StringIO()),
                                      demo=True, tcp_mode="custom", tcp_file=self.custom_file)
        for side in ("left", "right"):
            pos, rot = bridge.models[side].forward_kinematics(bridge.q(side))
            plan = bridge.plan(dict(side=side, mode="xyz", values=pos.tolist()))
            actual_pos, actual_rot = bridge.models[side].forward_kinematics(plan["q"])
            np.testing.assert_allclose(actual_pos, pos, atol=5e-4)
            np.testing.assert_allclose(actual_rot, rot, atol=1e-5)

    def test_bridge_home_is_same_joint_target_for_every_tool(self):
        for mode in TCP_MODES:
            path = str(self.custom_file) if mode == "custom" else None
            bridge = bridge_module.Bridge(bridge_module.Transport(io.StringIO(), io.StringIO()),
                                          demo=True, tcp_mode=mode, tcp_file=path)
            for side in bridge.demo_q:
                bridge.demo_q[side][0] -= .08
            before = {s: q.copy() for s, q in bridge.demo_q.items()}
            bridge.execute(dict(id=1, op="home", preview=True))
            for side, q in before.items():
                np.testing.assert_array_equal(bridge.q(side), q)
            bridge.execute(dict(id=2, op="arm"))
            bridge.execute(dict(id=3, op="home"))
            for side, state in bridge.snapshot()["arms"].items():
                np.testing.assert_array_equal(bridge.q(side), api.HOME)
                expected = bridge.models[side].forward_kinematics(api.HOME)[0]
                np.testing.assert_allclose(state["xyz"], expected, atol=1e-14)
                self.assertEqual(state["tcp"]["mode"], mode)

    def test_bridge_connection_retains_physical_dynamics(self):
        def factory(*args, **kwargs):
            client = FakeClient(*args, **kwargs)
            client.dyns = {s: ArmDynamics(m) for s, m in client.models.items()}
            return client
        with patch.object(ros, "X2ArmClient", side_effect=factory), \
                patch.object(bridge_module, "ActionQuery"):
            bridge = bridge_module.Bridge(bridge_module.Transport(io.StringIO(), io.StringIO()),
                                          tcp_mode="custom", tcp_file=self.custom_file)
        for side in ("left", "right"):
            dyn = bridge.cli.dyns[side]
            np.testing.assert_array_equal(dyn.model.tcp_offset, np.zeros(3))
            np.testing.assert_array_equal(dyn.model.tcp_rotation, np.eye(3))
            self.assertIsNot(dyn.model, bridge.models[side])
            self.assertIs(bridge.cli.models[side], bridge.models[side])
        self.assertEqual(bridge.cli.sent, [])
        bridge.cli.close()

    def test_terminal_custom_invalid_rejected_before_ros(self):
        with patch.object(ros, "X2ArmClient") as factory, \
                patch.dict(sys.modules, {"rclpy": None, "aimdk_msgs": None}), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ros.main(["mdi", "--tcp-mode", "custom", "--dry"])
            self.assertEqual(error.exception.code, 2)
            factory.assert_not_called()

    def test_terminal_installs_both_tcp_models_before_session_and_keeps_hand_mode(self):
        clients = []
        def factory(*args, **kwargs):
            client = FakeClient(*args, **kwargs)
            clients.append(client)
            return client
        def mdi(client, args):
            for side, model in client.models.items():
                np.testing.assert_allclose(model.tcp_offset,
                    load_tcp_tools("custom", self.custom_file)[side]["translation_m"])
                self.assertIs(client.iks[side].model, model)
            return 0
        with patch.object(ros, "X2ArmClient", side_effect=factory), \
                patch.object(ros, "cmd_mdi", side_effect=mdi), \
                patch.dict(sys.modules, {"rclpy": ModuleType("rclpy"),
                                         "aimdk_msgs": ModuleType("aimdk_msgs")}):
            self.assertEqual(ros.main(["mdi", "--dry", "--tcp-mode", "custom",
                                       "--tcp-file", str(self.custom_file)]), 0)
        self.assertEqual(clients[0].options["hand_mode"], 1)
        clients[0].close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
