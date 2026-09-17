"""Offline tests of desktop protocol/gates; no SSH and no robot movement."""
import copy
import importlib.util
import os
from pathlib import Path
import shlex
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mdi_desktop', ROOT / 'desktop/x2_mdi_desktop.py')
ui = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ui)
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith('PyQt5'):
        raise unittest.SkipTest('desktop tests require PyQt5')
    raise


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = ui.QtWidgets.QApplication.instance() or ui.QtWidgets.QApplication([])

    def setUp(self):
        self.args = ui.parser().parse_args(['--demo'])
        self.window = ui.Window(self.args)
        self.addCleanup(self.window.close)

    def pump(self):
        for _ in range(5):
            self.app.processEvents()

    def connect(self):
        self.window.connect_button.click()
        self.pump()
        return self.window

    def arm(self):
        self.window.arm_button.click()
        self.pump()

    def test_startup_no_transport_and_demo_readonly(self):
        w = self.window
        self.assertIsNone(w.backend)
        self.assertFalse(w.home_button.isEnabled())
        self.connect()
        self.assertIsInstance(w.backend, ui.DemoBackend)
        self.assertTrue(w.is_fresh())
        self.assertFalse(w.state['armed'])
        self.assertFalse(w.execute_button.isEnabled())
        self.assertTrue(ui.complete_arms(w.arm_view.arms))

    def test_camera_presets_never_send_commands_or_change_targets(self):
        w = self.connect()
        original_state = copy.deepcopy(w.state)
        original_fields = [field.text() for _, field in w.fields]
        original_history = copy.deepcopy(w.backend.history)
        for name, button in w.view_buttons.items():
            button.click()
            self.pump()
            self.assertEqual((w.arm_view.azimuth, w.arm_view.elevation), w.arm_view.VIEWS[name])
        self.assertEqual(w.state, original_state)
        self.assertEqual(w.backend.history, original_history)
        self.assertEqual([field.text() for _, field in w.fields], original_fields)
        self.assertFalse(w.execute_button.isEnabled())

    def test_quote_remote_paths_and_reject_host_option_injection(self):
        path = "/tmp/runtime 'with space' $(touch SHOULD_NOT_RUN)"
        args = ui.ssh_arguments('user@robot', path, '/tmp/config root/x2ik.conf')
        tokens = shlex.split(args[-1])
        self.assertEqual(tokens[:4], ['cd', '--', path, '&&'])
        self.assertIn('X2IK_CONFIG=/tmp/config root/x2ik.conf', tokens)
        self.assertIn('StrictHostKeyChecking=yes', args)
        self.assertIn('BatchMode=yes', args)
        for host in ('-oProxyCommand=bad', 'agi@host bad', 'agi@-host', 'host', 'agi@host;evil'):
            with self.assertRaises(ValueError):
                ui.ssh_arguments(host, '/tmp/runtime', '/tmp/conf')
        with self.assertRaises(ValueError):
            ui.ssh_arguments('agi@host', 'relative', '/tmp/conf')

    def test_all_modes_correct_arity_and_metres_conversion(self):
        w = self.connect()
        for mode, (_, labels) in ui.MODES.items():
            with self.subTest(mode=mode):
                w.mode.setCurrentIndex(w.mode.findData(mode))
                self.assertEqual(len(w.fields), len(labels))
                for i, (_, field) in enumerate(w.fields):
                    field.setText(str(i + 1))
                w.units.setCurrentText('mm')
                w.command(True)
                req = w.backend.history[-1]
                self.assertEqual(req['op'], 'mdi')
                self.assertEqual(req['mode'], mode)
                expected = [float(i + 1) for i in range(len(labels))]
                if mode in ('xyz', 'pose', 'd'):
                    expected[:3] = [v / 1000. for v in expected[:3]]
                self.assertEqual(req['values'], expected)
                self.assertTrue(req['preview'])
                self.pump()

    def test_nonfinite_input_not_sent(self):
        w = self.connect()
        w.fields[0][1].setText('nan')
        before = len(w.backend.history)
        w.command(True)
        self.assertEqual(len(w.backend.history), before)

    def test_preview_does_not_move_and_arm_is_explicit(self):
        w = self.connect()
        original = copy.deepcopy(w.state['arms'])
        w.mode.setCurrentIndex(w.mode.findData('j'))
        w.fill_current()
        w.fields[0][1].setText('30')
        w.preview_button.click()
        self.pump()
        self.assertEqual(w.state['arms'], original)
        self.assertFalse(w.execute_button.isEnabled())
        self.arm()
        self.assertTrue(w.execute_button.isEnabled())
        w.execute_button.click()
        self.assertFalse(w.home_button.isEnabled())
        self.pump()
        self.assertEqual(w.state['arms']['right']['q_deg'][0], 30)
        self.assertNotEqual(w.state['arms']['right']['xyz'], original['right']['xyz'])
        self.assertEqual(w.state['arms']['left'], original['left'])
        w.home_button.click()
        self.pump()
        self.assertEqual(w.state['arms']['right']['q_deg'], ui.HOME_DEG)

    def test_busy_bad_feedback_and_stale_inhibit_control_without_rearming(self):
        w = self.connect()
        self.arm()
        w.state['busy'] = True
        w.refresh_controls()
        self.assertFalse(w.execute_button.isEnabled())
        self.assertFalse(w.arm_button.isEnabled())
        self.assertTrue(w.disarm_button.isEnabled())
        w.state['busy'] = False
        w.last_state -= 3
        w.tick()
        self.assertFalse(w.execute_button.isEnabled())
        self.assertEqual([r['op'] for r in w.backend.history if r['op'] != 'heartbeat'][-1], 'disarm')
        self.pump()
        w.backend.snapshot()
        self.assertTrue(w.is_fresh())
        self.assertFalse(w.execute_button.isEnabled())
        self.arm()
        bad = copy.deepcopy(w.state)
        bad['arms']['left']['q_deg'] = [float('nan')] * 7
        w.on_message(bad)
        self.assertFalse(w.is_fresh())
        self.assertTrue(w.inhibit)
        self.assertFalse(w.home_button.isEnabled())

    def test_no_mode_controls_and_external_mode_change_requires_rearming(self):
        w = self.connect()
        self.assertFalse(hasattr(w, 'stand_button'))
        self.assertFalse(hasattr(w, 'urs_button'))
        self.assertFalse(hasattr(w, 'switch_action'))
        self.assertIn('MDI 不提供模式切换', w.urs_notice.text())
        self.arm()
        w.backend.action = 'STAND_DEFAULT'  # External state feedback only.
        w.backend.snapshot()
        self.pump()
        self.assertEqual(w.state['action'], 'STAND_DEFAULT')
        self.assertFalse(w.state['armed'])
        self.assertTrue(ui.complete_arms(w.arm_view.arms))
        self.assertTrue(w.preview_button.isEnabled())
        self.assertFalse(w.arm_button.isEnabled())
        self.assertFalse(w.home_button.isEnabled())
        w.backend.action = 'UPPERBODY_REMOTE_SPLIT'
        w.backend.snapshot()
        self.pump()
        self.assertTrue(w.arm_button.isEnabled())
        self.assertFalse(w.execute_button.isEnabled())
        self.assertFalse(any(request['op'] == 'action' for request in w.backend.history))
        self.arm()
        self.assertTrue(w.execute_button.isEnabled())

    def test_stale_mode_blocks_arm_despite_fresh_joint_feedback(self):
        w = self.connect()
        w.state['urs_confirmed'] = False
        w.refresh_controls()
        self.assertTrue(w.is_fresh())
        self.assertFalse(w.arm_button.isEnabled())
        self.assertFalse(w.home_button.isEnabled())

    def test_demo_rejects_mode_request_and_arm_outside_urs(self):
        w = self.connect()
        w.backend.send('action', action='STAND_DEFAULT')
        self.pump()
        self.assertEqual(w.backend.action, 'UPPERBODY_REMOTE_SPLIT')
        w.backend.action = 'STAND_DEFAULT'
        w.backend.send('arm')
        self.pump()
        self.assertFalse(w.backend.armed)

    def test_disconnect_clears_values_and_requires_explicit_reconnect(self):
        w = self.connect()
        self.arm()
        backend = w.backend
        w.disconnect_button.click()
        self.assertFalse(backend.active)
        self.assertIsNone(w.backend)
        self.assertEqual(w.state, {})
        self.assertEqual(w.arm_view.arms, {})
        self.pump()
        self.assertIsNone(w.backend)
        self.connect()
        self.assertFalse(w.state['armed'])
        self.assertFalse(w.execute_button.isEnabled())

    def test_demo_forward_kinematics_matches_project_model(self):
        # Changes to the URDF make the bundled demo snapshot discrepancy visible.
        sys.path.insert(0, str(ROOT))
        try:
            import numpy as np
            from x2_arm_model import ArmModel, matrix_to_rpy
        except ImportError:
            self.skipTest('numpy/model unavailable')
        for side in ('left', 'right'):
            for q in (ui.HOME_DEG, [10., 4., 7., -30., 5., 0., 2.]):
                actual = ui.demo_fk(side, q)
                model = ArmModel(side)
                p, r = model.forward_kinematics(np.radians(q))
                points, _ = model.joint_frames(np.radians(q))
                np.testing.assert_allclose(actual['xyz'], p, atol=1e-12)
                np.testing.assert_allclose(actual['rpy_deg'], np.degrees(matrix_to_rpy(r)), atol=1e-10)
                np.testing.assert_allclose(actual['points'][:-1], points, atol=1e-12)

    def test_tcp_demo_presets_and_custom_match_model(self):
        import numpy as np
        from x2_arm_model import ArmModel
        for mode in ui.TCP_MODES:
            filename = ROOT / 'config/tcp_tool.example.json' if mode == 'custom' else None
            tools = ui.load_tcp_tools(mode, filename)
            for side in ('left', 'right'):
                tool = tools[side]
                actual = ui.demo_fk(side, ui.HOME_DEG, tool)
                model = ArmModel(side, tcp_offset=tool['translation_m'],
                                 tcp_rotation=tool['rotation_matrix'])
                p, _ = model.forward_kinematics(np.radians(ui.HOME_DEG))
                points, rotations = model.joint_frames(np.radians(ui.HOME_DEG))
                np.testing.assert_allclose(actual['xyz'], p, atol=1e-12)
                np.testing.assert_allclose([x['xyz'] for x in actual['link_transforms']], points, atol=1e-12)
                np.testing.assert_allclose([x['rotation'] for x in actual['link_transforms']], rotations, atol=1e-12)
                self.assertEqual(actual['tcp'], tool)

    def test_tcp_selection_locked_until_disconnect(self):
        w = self.window
        w.tcp_mode.setCurrentIndex(w.tcp_mode.findData('hand'))
        self.connect()
        self.assertEqual(w.state['tcp_mode'], 'hand')
        self.assertFalse(w.tcp_mode.isEnabled())
        self.assertFalse(w.tcp_file.isEnabled())
        self.assertIn('估计值', w.readout.text())
        history = copy.deepcopy(w.backend.history)
        state = copy.deepcopy(w.state)
        for mode in ('skeleton', 'mesh'):
            w.render_mode.setCurrentIndex(w.render_mode.findData(mode))
            self.pump()
            self.assertEqual(w.arm_view.render_mode, mode)
        self.assertEqual(w.backend.history, history)
        self.assertEqual(w.state, state)
        w.disconnect_button.click()
        self.assertTrue(w.tcp_mode.isEnabled())

    def test_tcp_mismatch_or_old_backend_blocks_all_commands(self):
        w = self.connect()
        self.arm()
        for mode in ('hand', None):
            bad = copy.deepcopy(w.state)
            bad['tcp_mode'] = mode
            w.on_message(bad)
            self.assertTrue(w.inhibit)
            for button in (w.arm_button, w.preview_button, w.execute_button, w.home_button, w.fill_button):
                self.assertFalse(button.isEnabled())
        self.assertTrue(w.disarm_button.isEnabled())

    def test_custom_tcp_ssh_quoted_path_and_no_upload(self):
        filename = "/tmp/cup 'tool' $(not-a-command).json"
        args = ui.ssh_arguments('user@robot', '/tmp/runtime', '/tmp/conf',
                                tcp_mode='custom', tcp_file=filename)
        tokens = shlex.split(args[-1])
        self.assertEqual(tokens[tokens.index('--tcp-mode') + 1], 'custom')
        self.assertEqual(tokens[tokens.index('--tcp-file') + 1], filename)
        for mode, path in (('bad', None), ('custom', None), ('custom', 'relative.json'),
                           ('custom', '/tmp/new\nline'), ('none', '/tmp/tool.json')):
            with self.assertRaises(ValueError):
                ui.ssh_arguments('user@robot', '/tmp/runtime', '/tmp/conf', tcp_mode=mode, tcp_file=path)

    def test_invalid_custom_demo_file_fails_before_connect(self):
        w = self.window
        w.tcp_mode.setCurrentIndex(w.tcp_mode.findData('custom'))
        w.tcp_file.setText('/nonexistent/x2-tool.json')
        self.connect()
        self.assertIsNone(w.backend)
        self.assertFalse(w.execute_button.isEnabled())


if __name__ == '__main__':
    unittest.main()
