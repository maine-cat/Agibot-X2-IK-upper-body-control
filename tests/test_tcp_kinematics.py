"""Rigid wrist-to-TCP transforms across scalar/batch IK and dynamics (offline)."""
import unittest

import numpy as np

from x2_arm_dynamics import ArmDynamics
from x2_arm_model import ArmModel, DEFAULT_URDF, log3, rpy_to_matrix
from x2_srs_ik import SrsArmIK


class TcpKinematicsTests(unittest.TestCase):
    OFFSET = np.array([0.031, -0.024, 0.163])
    ROTATION = rpy_to_matrix([0.33, -0.51, 0.72])

    def models(self):
        for side in ("left", "right"):
            sign = 1.0 if side == "left" else -1.0
            q = np.array([-0.35, sign * 0.6, 0.23, -1.05, 0.24, -0.13, 0.18])
            model = ArmModel(side, tcp_offset=self.OFFSET, tcp_rotation=self.ROTATION)
            yield side, model, q

    def test_identity_default_preserves_positional_constructor(self):
        for side in ("left", "right"):
            old_signature = ArmModel(side, DEFAULT_URDF, self.OFFSET, None)
            explicit = ArmModel(side, tcp_offset=self.OFFSET, tcp_rotation=np.eye(3))
            q = np.zeros(7)
            for actual, expected in zip(old_signature.forward_kinematics(q),
                                        explicit.forward_kinematics(q)):
                np.testing.assert_allclose(actual, expected, atol=0.0)

    def test_rejects_invalid_rigid_transforms(self):
        for offset in ([0, 0], [[0, 0, 0]], [0, 0, np.nan], [np.inf, 0, 0]):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                ArmModel(tcp_offset=offset)
        bad_rotations = (np.zeros((3, 3)), np.eye(2), np.diag([1, 1, -1]),
                         np.eye(3) * 1.001, np.full((3, 3), np.nan))
        for rotation in bad_rotations:
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                ArmModel(tcp_rotation=rotation)

    def test_fk_composes_wrist_transform_for_both_sides(self):
        for side, model, q in self.models():
            with self.subTest(side=side):
                bare = ArmModel(side)
                p_wrist, r_wrist = bare.forward_kinematics(q)
                p_tcp, r_tcp = model.forward_kinematics(q)
                np.testing.assert_allclose(p_tcp, p_wrist + r_wrist @ self.OFFSET,
                                           atol=1e-14)
                np.testing.assert_allclose(r_tcp, r_wrist @ self.ROTATION, atol=1e-14)
                np.testing.assert_allclose(model.wrist_center(q), bare.wrist_center(q),
                                           atol=1e-14)

    def test_fk_and_jacobian_batch_preserve_leading_dimensions(self):
        for side, model, q in self.models():
            with self.subTest(side=side):
                qs = np.stack([q, q + 0.02, q - 0.01, q + 0.04]).reshape(2, 2, 7)
                positions, rotations = model.fk_batch(qs)
                jacobians = model.jacobian_batch(qs)
                for idx in np.ndindex(2, 2):
                    pos, rot = model.forward_kinematics(qs[idx])
                    np.testing.assert_allclose(positions[idx], pos, atol=1e-14)
                    np.testing.assert_allclose(rotations[idx], rot, atol=1e-14)
                    np.testing.assert_allclose(jacobians[idx], model.jacobian(qs[idx]),
                                               atol=1e-14)

    def test_tcp_jacobian_matches_world_frame_finite_difference(self):
        h = 1e-6
        for side, model, q in self.models():
            with self.subTest(side=side):
                finite_difference = np.empty((6, 7))
                for i in range(7):
                    dq = np.eye(7)[i] * h
                    p_plus, r_plus = model.forward_kinematics(q + dq)
                    p_minus, r_minus = model.forward_kinematics(q - dq)
                    finite_difference[:3, i] = (p_plus - p_minus) / (2 * h)
                    finite_difference[3:, i] = log3(r_plus @ r_minus.T) / (2 * h)
                np.testing.assert_allclose(model.jacobian(q), finite_difference,
                                           atol=2e-9, rtol=2e-8)

    def test_target_wrist_and_workspace_projection_use_wrist_orientation(self):
        for side, model, q in self.models():
            with self.subTest(side=side):
                ik = SrsArmIK(model)
                pos, rot = model.forward_kinematics(q)
                np.testing.assert_allclose(ik.target_wrist_center(pos, rot),
                                           model.wrist_center(q), atol=1e-14)
                rot_wrist = model.joint_frames(q)[1][-1]
                direction = np.array([0.3, -0.4, 0.5])
                direction /= np.linalg.norm(direction)
                margin = 0.003
                for distance, expected in ((ik.reach_max_limited + 0.2,
                                            ik.reach_max_limited - margin),
                                           (ik.reach_min_limited * 0.5,
                                            ik.reach_min_limited + margin)):
                    wrist = ik.shoulder + direction * distance
                    target = wrist + rot_wrist @ (self.OFFSET - ik.wrist_local)
                    projected, clipped = ik.project_to_workspace(target, rot, margin)
                    self.assertTrue(clipped)
                    np.testing.assert_allclose(ik.target_wrist_center(projected, rot),
                                               ik.shoulder + direction * expected,
                                               atol=1e-13)

    def test_scalar_and_batch_analytic_solutions_match_with_rotated_tcp(self):
        for side, model, q in self.models():
            with self.subTest(side=side):
                ik = SrsArmIK(model)
                qs = np.stack([q, q + 0.015])
                pos, rot = model.fk_batch(qs)
                psi = ik.sew_angle_batch(qs)
                grid = ik.batch.solve_grid(pos, rot, psi[:, None])
                for target in range(2):
                    scalar = ik.solve_at_psi(pos[target], rot[target], psi[target],
                                             limits_only=False)
                    self.assertTrue(scalar)
                    self.assertEqual(int(grid.ok[target, 0].sum()), len(scalar))
                    for sol in scalar:
                        branch = 4 * sol.elbow_branch + 2 * sol.shoulder_branch + sol.wrist_branch
                        delta = grid.q[target, 0, branch] - sol.q
                        np.testing.assert_allclose(np.arctan2(np.sin(delta), np.cos(delta)),
                                                   np.zeros(7), atol=1e-10)

    def test_fk_ik_roundtrip_analytic_polish_and_batch_solver(self):
        for side, model, q in self.models():
            with self.subTest(side=side):
                ik = SrsArmIK(model)
                pos, rot = model.forward_kinematics(q)
                psi = ik.sew_angle(q)
                candidates = ik.solve_at_psi(pos, rot, psi)
                self.assertTrue(candidates)
                near = min(candidates, key=lambda sol: np.linalg.norm(sol.q - q))
                polished = ik.polish(near, pos, rot)
                self.assertLess(polished.pos_error, 1e-9)
                self.assertLess(polished.rot_error, 1e-9)
                np.testing.assert_allclose(polished.q, q, atol=1e-7)
                solved = ik.solve(pos, rot, q_seed=q, psi_hint=psi,
                                  weight_margin=0, weight_manip=0, fallback_numeric=False)
                self.assertIsNotNone(solved)
                self.assertLess(solved.pos_error, 1e-9)
                self.assertLess(solved.rot_error, 1e-9)

    def test_polish_residuals_and_numeric_fallback_use_tcp_orientation(self):
        for side, model, q in self.models():
            with self.subTest(side=side):
                ik = SrsArmIK(model)
                pos, rot = model.forward_kinematics(q)
                psi = ik.sew_angle(q)
                qs = np.stack([q, q + 0.02, q - 0.01])
                residuals = ik._residual_batch(qs, pos, rot, psi)
                np.testing.assert_allclose(residuals[0], np.zeros(7), atol=1e-14)
                for idx, qq in enumerate(qs):
                    np.testing.assert_allclose(residuals[idx], ik._residual(qq, pos, rot, psi),
                                               atol=1e-14)
                solved = ik.numeric_solve(pos, rot, q_seed=q + 0.02)
                self.assertLess(solved.pos_error, 1e-9)
                self.assertLess(solved.rot_error, 1e-9)

    def test_payload_com_and_gravity_are_invariant_to_tcp_axis_rotation(self):
        for side, model, q in self.models():
            with self.subTest(side=side):
                unrotated = ArmModel(side, tcp_offset=self.OFFSET)
                kwargs = dict(payload_mass=0.43, payload_com=[0.021, 0.013, -0.04],
                              gravity=[0.1, 0.2, -9.807])
                rotated_dyn = ArmDynamics(model, **kwargs)
                plain_dyn = ArmDynamics(unrotated, **kwargs)
                np.testing.assert_allclose(rotated_dyn.total_com(q), plain_dyn.total_com(q),
                                           atol=1e-14)
                np.testing.assert_allclose(rotated_dyn.gravity_torque(q),
                                           plain_dyn.gravity_torque(q), atol=1e-14)
                # Independent potential-energy derivative catches offset double counting
                # and disagreement between total COM and the gravity torque path.
                mass = rotated_dyn.arm_mass + rotated_dyn.payload_mass
                h = 1e-6
                gradient = np.empty(7)
                for i in range(7):
                    dq = np.eye(7)[i] * h
                    delta_com = rotated_dyn.total_com(q + dq) - rotated_dyn.total_com(q - dq)
                    gradient[i] = -mass * rotated_dyn.gravity @ delta_com / (2 * h)
                np.testing.assert_allclose(rotated_dyn.gravity_torque(q), gradient,
                                           atol=1e-8, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
