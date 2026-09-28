"""Exercise the original motion loop with an in-memory joint feedback client."""
import unittest
from unittest.mock import patch

import numpy as np

from fk_oracle import IndependentFK, errors
from x2ik.x2_api import HOME
from x2ik.x2_arm_model import ArmModel, DEFAULT_URDF
from x2ik.x2_srs_ik import SrsArmIK
from x2ik import x2_sim_ros as motion


class MemoryClient:
    rate = 50
    recorder = None

    def __init__(self):
        self.models = {s: ArmModel(s) for s in ('left', 'right')}
        self.iks = {s: SrsArmIK(m) for s, m in self.models.items()}
        self.joints = {s: HOME.copy() for s in self.models}
        self.sent = []

    def q(self, side):
        return self.joints[side].copy()

    def send(self, left, right, *velocity):
        for side, q in (('left', left), ('right', right)):
            if not self.models[side].within_limits(q, tol=1e-9):
                raise AssertionError('out-of-limit command')
            self.joints[side] = q.copy()
        self.sent.append((left.copy(), right.copy()))

    def fresh_state(self):
        return True


class MotionTests(unittest.TestCase):
    def test_original_cartesian_loop_with_local_ik(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                client = MemoryClient()
                oracle = IndependentFK(DEFAULT_URDF.read_bytes(), side)
                pose = oracle.pose(HOME + np.array([.04, -.02, .03, -.04, .01, .01, -.01]))
                with patch.object(motion, '_pace'), \
                     patch.object(client.iks[side], 'solve', side_effect=AssertionError('global IK in motion')):
                    result = motion.goto_cartesian(client, side, *pose, duration=1., settle=0.)
                self.assertFalse(result['aborted'])
                self.assertTrue(result['trajectory_valid'])
                self.assertEqual(result['ik_fails'], 0)
                self.assertEqual(result['clipped'], 0)
                self.assertEqual(result['implicit_fallbacks'], 0)
                self.assertLessEqual(result['peak_accepted_dq'], .05)
                pe, re = errors(oracle.pose(client.q(side)), pose)
                self.assertLessEqual(pe, .01)
                self.assertLessEqual(re, .001)
                other_index = 1 if side == 'left' else 0
                for values in client.sent:
                    np.testing.assert_array_equal(values[other_index], HOME)

    def test_repeated_ik_failure_holds_then_stops(self):
        client = MemoryClient()
        pose = client.models['left'].forward_kinematics(HOME)
        with patch.object(motion, '_pace'), \
             patch.object(client.iks['left'], 'track', return_value=None):
            result = motion.goto_cartesian(client, 'left', *pose, duration=1., settle=0.)
        self.assertTrue(result['aborted'])
        self.assertFalse(result['trajectory_valid'])
        self.assertEqual(result['ik_fails'], 5)
        self.assertLess(len(client.sent), 50)
        for left, right in client.sent:
            np.testing.assert_array_equal(left, HOME)
            np.testing.assert_array_equal(right, HOME)

    def test_preflight_does_not_replace_unreachable_target(self):
        client = MemoryClient()
        result = motion.preflight(client, 'left', np.array([10., 0, 0]), np.eye(3))
        self.assertFalse(result['ok'])
        self.assertTrue(result['clipped'])
        self.assertIsNone(result['q'])
        self.assertEqual(client.sent, [])

    def test_small_home_motion_does_not_switch_redundant_solution(self):
        client = MemoryClient()
        pose = client.models['left'].forward_kinematics(HOME)
        with patch.object(motion, '_pace'):
            result = motion.goto_cartesian(client, 'left', *pose, duration=.2, settle=0.)
        self.assertFalse(result['aborted'])
        for left, right in client.sent:
            np.testing.assert_array_equal(left, HOME)
            np.testing.assert_array_equal(right, HOME)


if __name__ == '__main__':
    unittest.main()
