"""Offline mesh provenance, geometry, kinematics and fallback tests."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from desktop.arm_mesh import ArmMeshModel, ASSET_PATH

ROOT = Path(__file__).resolve().parents[1]


class ArmMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = ArmMeshModel()

    def test_assets_are_only_arm_links_with_metre_units(self):
        data = json.loads(ASSET_PATH.read_text())
        self.assertEqual(data['units'], 'm')
        self.assertEqual(data['base_frame'], 'torso_link')
        self.assertEqual(data['kinematics_urdf_sha256'],
                         hashlib.sha256((ROOT / 'x2_ultra.urdf').read_bytes()).hexdigest())
        for side in ('left', 'right'):
            self.assertEqual(len(data['arms'][side]), 7)
            for link in data['arms'][side]:
                self.assertTrue(link['link'].startswith(side + '_'))
                self.assertTrue(link['source_mesh'].endswith('.STL'))
                report = link['simplification']
                self.assertLessEqual(report['bounds_max_delta_m'], .003)
                self.assertLessEqual(report['boundary_edges'], report['source_boundary_edges'])
                self.assertLess(report['faces'], report['source_faces'])
                self.assertEqual(report['faces'], len(link['faces']))
                self.assertLess(max(max(abs(v) for v in p) for p in link['vertices']), .30)

    def test_all_link_frames_match_motion_model_at_random_poses(self):
        import numpy as np
        from x2_arm_model import ArmModel
        rng = random.Random(216)
        for side in ('left', 'right'):
            arm = ArmModel(side)
            for _ in range(12):
                q = np.array([rng.uniform(float(a), float(b)) for a, b in zip(arm.q_min, arm.q_max)])
                positions, rotations = arm.joint_frames(q)
                actual = self.mesh.link_transforms(side, np.degrees(q).tolist())
                for i, transform in enumerate(actual):
                    np.testing.assert_allclose(transform['xyz'], positions[i], atol=1e-12)
                    np.testing.assert_allclose(transform['rotation'], rotations[i], atol=1e-12)

    def test_live_transforms_take_precedence_and_tcp_cannot_move_links(self):
        transforms = self.mesh.link_transforms('left', [0., 0., 0., -60., 0., 0., 0.])
        arm = {'q_deg': [0.] * 7, 'link_transforms': transforms, 'xyz': [0., 0., 0.]}
        first, source = self.mesh.posed_links('left', arm)
        arm['xyz'] = [5., 6., 7.]
        second, _ = self.mesh.posed_links('left', arm)
        self.assertEqual(first, second)
        self.assertEqual(source, 'feedback')
        fallback, source = self.mesh.posed_links('left', {'q_deg': [0.] * 7})
        self.assertEqual(source, 'packaged_fk')
        self.assertNotEqual(first, fallback)

    def test_invalid_transform_never_silently_uses_joint_fallback(self):
        transforms = self.mesh.link_transforms('left', [0.] * 7)
        for broken in ([], None, [{'xyz': [0., 0., 0.], 'rotation': [[1, 0, 0]]}] * 7):
            with self.assertRaises(ValueError):
                self.mesh.posed_links('left', {'q_deg': [0.] * 7, 'link_transforms': broken})
        reflected = copy.deepcopy(transforms)
        reflected[0]['rotation'] = ((-1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
        with self.assertRaises(ValueError):
            self.mesh.posed_links('left', {'link_transforms': reflected})
        for bad_q in ([0.] * 6, [math.nan] * 7, [True] * 7):
            with self.assertRaises(ValueError):
                self.mesh.link_transforms('left', bad_q)

    def test_corrupt_asset_is_rejected(self):
        data = json.loads(ASSET_PATH.read_text())
        data['arms']['left'][0]['faces'][0][0] = 9000000
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'face indices'):
                ArmMeshModel(path)


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt5 import QtWidgets
    from desktop.arm_view_3d import ArmView
except ModuleNotFoundError:
    QtWidgets = None


@unittest.skipIf(QtWidgets is None, 'Qt is unavailable')
class ArmMeshViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def feedback(self):
        return {'left': {'points': [[0., .15, .24], [.1, .2, -.1]],
                         'xyz': [.1, .2, -.1], 'rpy_deg': [0., 0., 0.], 'q_deg': [0.] * 7}}

    def test_display_modes_and_stale_feedback_clear_geometry(self):
        view = ArmView()
        self.addCleanup(view.close)
        view.set_arms(self.feedback())
        self.assertIn('STL', view.render_status())
        self.assertTrue(view._posed_meshes)
        view.set_render_mode('skeleton')
        self.assertEqual(view.render_status(), '骨架示意')
        with self.assertRaises(ValueError):
            view.set_render_mode('invalid')
        view.set_render_mode('mesh')
        view.set_arms({})
        self.assertEqual(view._posed_meshes, {})
        self.assertFalse(view.grab().isNull())

    def test_missing_asset_and_invalid_feedback_label_skeleton_fallback(self):
        with patch('desktop.arm_view_3d.ArmMeshModel', side_effect=OSError('missing asset')):
            view = ArmView()
        self.addCleanup(view.close)
        view.set_arms(self.feedback())
        self.assertIn('骨架降级', view.render_status())
        self.assertFalse(view.grab().isNull())
        live = ArmView()
        self.addCleanup(live.close)
        arm = self.feedback()
        arm['left']['link_transforms'] = []
        live.set_arms(arm)
        self.assertIn('骨架降级', live.render_status())
        self.assertEqual(live._posed_meshes, {})


if __name__ == '__main__':
    unittest.main()
