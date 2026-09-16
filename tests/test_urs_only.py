"""URS 点位硬闸：离线证明所有状态请求和 HAL 直控被拒绝。"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_sim_ros as ros


class UrsOnlyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"X2_URS_ONLY": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cli = object.__new__(ros.X2ArmClient)
        self.cli.mode = "upper_body"
        self.cli.verbose = False
        self.cli.node = SimpleNamespace(count_subscribers=lambda _: 1)
        self.cli.get_action = Mock(return_value="UPPERBODY_REMOTE_SPLIT")

    def test_all_state_requests_are_rejected_before_ros_import(self):
        for action in ("JD", "PD", "STAND_DEFAULT", "UPPERBODY_REMOTE_SPLIT"):
            self.assertFalse(self.cli.set_action(action))

    def test_urs_entry_only_reads_state_even_when_reenter_true(self):
        self.cli.reenter = True
        self.cli.set_action = Mock(side_effect=AssertionError("state switch"))
        self.assertTrue(self.cli.enter_control())
        self.cli.set_action.assert_not_called()

    def test_unknown_and_non_urs_do_not_switch(self):
        for action in (None, "STAND_DEFAULT", "WHOLE_BODY_TELEOP"):
            self.cli.get_action.return_value = action
            self.assertFalse(self.cli.enter_control())

    def test_direct_hal_cannot_send(self):
        self.cli.mode = "joint_direct"
        self.cli._send_joint = Mock()
        with self.assertRaises(RuntimeError):
            self.cli.send(None, None)
        self.cli._send_joint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
