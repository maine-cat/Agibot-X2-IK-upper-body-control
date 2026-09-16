"""Customer paths use one profile regardless of site calibration; no ROS/hardware."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_api as api
from x2_compensation import fixed_compensation
import x2_mdi_bridge as bridge_module
import x2_movej
import x2_sim_ros as ros
import x2ik
from test_x2_api import FakeClient


class FixedCompensationTests(unittest.TestCase):
    def setUp(self):
        api.X2Arm._connected = None
        temporary = tempfile.TemporaryDirectory(prefix="x2ik-fixed-test-")
        self.addCleanup(temporary.cleanup)
        self.calibration = Path(temporary.name)
        for replacement in (
                patch.dict(os.environ, {"X2_ROBOT_SN": "TEST_ROBOT"}),
                patch.object(ros, "CALIB_DIR", self.calibration),
                patch.object(x2ik, "load_conf"),
                patch.dict(sys.modules, {"rclpy": ModuleType("rclpy"),
                                         "aimdk_msgs": ModuleType("aimdk_msgs")})):
            replacement.start()
            self.addCleanup(replacement.stop)

    def configurations(self):
        """Missing, unreadable JSON, and stale but valid calibration files."""
        return (None, "not valid JSON", json.dumps(dict(
            sn="ANOTHER_ROBOT", stiffness=13., bias_limit_deg=2., gravity_source="static")))

    def configure(self, content):
        target = self.calibration / "TEST_ROBOT.json"
        if content is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(content)

    def assert_fixed_client(self, call):
        args, kwargs = call
        stiffness = kwargs.get("joint_stiffness") if "joint_stiffness" in kwargs else args[5]
        source = kwargs.get("gravity_source") if "gravity_source" in kwargs else args[7]
        np.testing.assert_array_equal(stiffness, np.full(7, 40.))
        self.assertEqual(kwargs["bias_limit"], math.radians(12.))
        self.assertEqual(source, "pelvis")

    def test_profile_returns_independent_values(self):
        profile = fixed_compensation()
        profile["gravity_source"] = "static"
        self.assertEqual(fixed_compensation(), dict(
            stiffness=40., bias_limit_deg=12., gravity_source="pelvis"))

    def test_public_robot_never_reads_site_calibration(self):
        for content in self.configurations():
            with self.subTest(content=content):
                self.configure(content)
                with patch.object(ros, "load_calibration", wraps=ros.load_calibration) as load, \
                        patch.object(ros, "X2ArmClient", side_effect=FakeClient) as factory:
                    with x2_movej.Robot(verbose=False) as robot:
                        self.assert_fixed_client(factory.call_args)
                        self.assertEqual(robot._arm.robot_sn, "TEST_ROBOT")
                        robot._arm.cli.set_action.assert_not_called()
                        self.assertEqual(robot._arm.cli.sent, [])
                    load.assert_not_called()

    def test_explicit_robot_sn_is_metadata_without_calibration_requirement(self):
        with patch.object(ros, "load_calibration", side_effect=AssertionError("calibration read")) as load, \
                patch.object(ros, "X2ArmClient", side_effect=FakeClient) as factory:
            with x2_movej.Robot(robot_sn="NO_SUCH_MACHINE", verbose=False) as robot:
                self.assert_fixed_client(factory.call_args)
                self.assertEqual(robot._arm.connection_config["robot_sn"], "NO_SUCH_MACHINE")
        load.assert_not_called()

    def test_desktop_bridge_never_reads_site_calibration(self):
        for content in self.configurations():
            with self.subTest(content=content):
                self.configure(content)
                with patch.object(ros, "load_calibration", wraps=ros.load_calibration) as load, \
                        patch.object(ros, "X2ArmClient", side_effect=FakeClient) as factory, \
                        patch.object(bridge_module, "ActionQuery"):
                    bridge = bridge_module.Bridge(bridge_module.Transport(io.StringIO(), io.StringIO()))
                    self.assert_fixed_client(factory.call_args)
                    self.assertTrue(factory.call_args.kwargs["read_only"])
                    self.assertFalse(bridge.armed)
                    bridge.cli.set_action.assert_not_called()
                    self.assertEqual(bridge.cli.sent, [])
                    load.assert_not_called()

    def test_terminal_mdi_never_reads_site_calibration(self):
        for content in self.configurations():
            with self.subTest(content=content):
                self.configure(content)
                with patch.object(ros, "load_calibration", wraps=ros.load_calibration) as load, \
                        patch.object(ros, "X2ArmClient", side_effect=FakeClient) as factory, \
                        patch.object(ros, "cmd_mdi", return_value=0), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(ros.main(["mdi", "--dry"]), 0)
                    self.assert_fixed_client(factory.call_args)
                    load.assert_not_called()

    def test_terminal_mdi_rejects_conflicting_overrides_before_client_creation(self):
        for option, value in (("--stiffness", "35"), ("--stiffness", "nan"),
                              ("--bias-limit", "8"), ("--bias-limit", "nan"),
                              ("--gravity-source", "chest"), ("--gravity-source", "static")):
            with self.subTest(option=option, value=value), \
                    patch.object(ros, "X2ArmClient") as factory, \
                    patch.object(ros, "load_calibration") as load, \
                    contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as caught:
                    ros.main([option, value, "mdi", "--dry"])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("MDI 使用固定补偿", error.getvalue())
                factory.assert_not_called()
                load.assert_not_called()

    def test_terminal_mdi_accepts_explicit_matching_values(self):
        with patch.object(ros, "X2ArmClient", side_effect=FakeClient) as factory, \
                patch.object(ros, "load_calibration") as load, \
                patch.object(ros, "cmd_mdi", return_value=0):
            self.assertEqual(ros.main([
                "--stiffness", "40", "--bias-limit", "12", "--gravity-source", "pelvis",
                "mdi", "--dry"]), 0)
            self.assert_fixed_client(factory.call_args)
            load.assert_not_called()

    def test_non_mdi_engineering_command_still_loads_calibration(self):
        calibration = dict(sn="TEST_ROBOT", stiffness=35., bias_limit_deg=5., gravity_source="static")
        with patch.object(ros, "load_calibration", return_value=calibration) as load, \
                patch.object(ros, "X2ArmClient", side_effect=FakeClient) as factory, \
                patch.object(ros, "cmd_state", return_value=0), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ros.main(["state"]), 0)
            load.assert_called_once_with("TEST_ROBOT")
            np.testing.assert_array_equal(factory.call_args.args[5], np.full(7, 35.))
            self.assertEqual(factory.call_args.args[7], "static")
            self.assertEqual(factory.call_args.kwargs["bias_limit"], math.radians(5.))

    def test_public_robot_still_requires_urs_and_valid_pelvis_imu(self):
        for failure in ("not-urs", "missing-imu", "invalid-orientation"):
            with self.subTest(failure=failure):
                client = FakeClient()
                if failure == "not-urs":
                    client.get_action.return_value = "STAND_DEFAULT"
                elif failure == "missing-imu":
                    client.imu_count["pelvis"] = 0
                else:
                    client.imu_has_orientation["pelvis"] = False
                with patch.object(ros, "X2ArmClient", return_value=client), \
                        self.assertRaises(RuntimeError):
                    x2_movej.Robot(verbose=False)
                self.assertEqual(client.sent, [])
                client.set_action.assert_not_called()
                client.close.assert_called_once()
                self.assertIsNone(api.X2Arm._connected)


if __name__ == "__main__":
    unittest.main()
