"""Offline camera/geometry tests; no robot transport or ROS required."""
import copy
import math
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from desktop.arm_view_3d import ArmView, orientation_axes
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith('PyQt5'):
        raise unittest.SkipTest('3D view tests require PyQt5')
    raise


def feedback():
    return {side: {'q_deg': [0.] * 7,
                   'points': [[0., sign * .14, .24], [0., sign * .19, .24],
                              [.05, sign * .23, .03], [.15, sign * .25, -.1]],
                   'xyz': [.15, sign * .25, -.1], 'rpy_deg': [0., 20., 0.]}
            for side, sign in (('left', 1), ('right', -1))}


class ArmViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.view = ArmView()
        self.view.resize(640, 400)
        self.addCleanup(self.view.close)

    def test_front_axes_match_robot_coordinates(self):
        self.view.set_view('front')
        origin, depth = self.view.project_point((0, 0, 0))
        forward, forward_depth = self.view.project_point((.1, 0, 0))
        left, _ = self.view.project_point((0, .1, 0))
        up, _ = self.view.project_point((0, 0, .1))
        self.assertEqual(forward, origin)
        self.assertGreater(forward_depth, depth)
        self.assertGreater(left.x(), origin.x())
        self.assertLess(up.y(), origin.y())

    def test_camera_frame_preserves_handedness_at_every_angle(self):
        angles = list(self.view.VIEWS.values()) + [(127., -37.), (269., 83.), (180., 0.)]
        for azimuth, elevation in angles:
            self.view.azimuth, self.view.elevation = azimuth, elevation
            right, up, eye = self.view._basis()
            cross = (right[1]*up[2]-right[2]*up[1],
                     right[2]*up[0]-right[0]*up[2],
                     right[0]*up[1]-right[1]*up[0])
            for actual, expected in zip(cross, eye):
                self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(sum(v*v for v in right), 1.)
            self.assertAlmostEqual(sum(v*v for v in up), 1.)
            self.assertAlmostEqual(sum(v*v for v in eye), 1.)

    def test_front_observer_sees_robot_left_on_screen_right(self):
        arms = feedback()
        self.view.set_arms(arms)
        for view in ('front', 'default'):
            self.view.set_view(view)
            left, _ = self.view.project_point(arms['left']['xyz'])
            right, _ = self.view.project_point(arms['right']['xyz'])
            self.assertGreater(left.x(), right.x())
        self.assertEqual(self.view.arms, arms)

    def test_default_camera_represents_all_three_axes(self):
        origin, _ = self.view.project_point((0, 0, 0))
        for endpoint in ((.1, 0, 0), (0, .1, 0), (0, 0, .1)):
            projected, depth = self.view.project_point(endpoint)
            self.assertGreater((projected - origin).manhattanLength(), 1.)
            self.assertTrue(math.isfinite(depth))

    def test_named_views_resolve_side_and_top_depth(self):
        for view, point in (('side', (0., .1, 0.)), ('top', (0., 0., .1))):
            with self.subTest(view=view):
                self.view.set_view(view)
                origin, depth = self.view.project_point((0., 0., 0.))
                projected, next_depth = self.view.project_point(point)
                self.assertAlmostEqual(projected.x(), origin.x())
                self.assertAlmostEqual(projected.y(), origin.y())
                self.assertGreater(next_depth, depth)
        with self.assertRaises(ValueError):
            self.view.set_view('invalid')

    def test_rpy_axes_follow_rz_ry_rx(self):
        axes = orientation_axes((0., 0., 90.))
        for actual, expected in zip(axes, ((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.))):
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)
        # A compound rotation distinguishes fixed-axis composition order.
        axes = orientation_axes((90., 0., 90.))
        for actual, expected in zip(axes, ((0., 1., 0.), (0., 0., 1.), (1., 0., 0.))):
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)

    def test_invalid_or_missing_feedback_clears_previous_limbs(self):
        original = feedback()
        self.view.set_arms(original)
        original['left']['points'][0][0] = 99.
        self.assertEqual(self.view.arms['left']['points'][0][0], 0.)
        for invalid in (None, {}, {'left': {}}, {'left': dict(feedback()['left'], xyz=[float('nan'), 0, 0])},
                        {'left': dict(feedback()['left'], points=[[0, 0, 0], [0, 1, float('inf')]])}):
            self.view.set_arms(invalid)
            self.assertEqual(self.view.arms, {})
        self.view.set_arms({'right': feedback()['right']})
        self.assertEqual(set(self.view.arms), {'right'})

    def test_mouse_camera_operations_preserve_feedback(self):
        original = feedback()
        self.view.set_arms(original)
        before = copy.deepcopy(self.view.arms)
        self.view.mousePressEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
            QtCore.QPointF(100, 100), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        self.view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove,
            QtCore.QPointF(150, 120), QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        self.view.mouseReleaseEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease,
            QtCore.QPointF(150, 120), QtCore.Qt.LeftButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
        self.assertNotEqual((self.view.azimuth, self.view.elevation), self.view.VIEWS['default'])
        self.view.mousePressEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
            QtCore.QPointF(100, 100), QtCore.Qt.RightButton, QtCore.Qt.RightButton, QtCore.Qt.NoModifier))
        self.view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove,
            QtCore.QPointF(120, 130), QtCore.Qt.NoButton, QtCore.Qt.RightButton, QtCore.Qt.NoModifier))
        self.assertEqual(self.view.pan, QtCore.QPointF(20, 30))
        for delta in (120, 120000, -120000):
            self.view.wheelEvent(QtGui.QWheelEvent(QtCore.QPointF(200, 150), QtCore.QPointF(200, 150),
                QtCore.QPoint(), QtCore.QPoint(0, delta), QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                QtCore.Qt.NoScrollPhase, False))
            self.assertGreaterEqual(self.view.zoom, .45)
            self.assertLessEqual(self.view.zoom, 3.)
        self.assertEqual(self.view.arms, before)
        self.assertEqual(original, before)
        self.view.reset_view()
        self.assertEqual(self.view.zoom, 1.)
        self.assertEqual(self.view.pan, QtCore.QPointF())
        self.assertEqual((self.view.azimuth, self.view.elevation), self.view.VIEWS['default'])

    def test_offscreen_paint_supports_all_views_and_empty_feedback(self):
        self.view.set_arms(feedback())
        self.view.show()
        self.app.processEvents()
        for view in self.view.VIEWS:
            self.view.set_view(view)
            snapshot = self.view.grab().toImage()
            self.assertFalse(snapshot.isNull())
            self.assertGreater(snapshot.width(), 0)
        self.view.set_arms({})
        self.assertFalse(self.view.grab().isNull())


if __name__ == '__main__':
    unittest.main()
