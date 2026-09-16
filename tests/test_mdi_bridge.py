"""Desktop bridge safety contract, entirely offline."""
import io
import json
import math
import queue
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from x2_mdi_bridge import Bridge, Cancelled, Transport, validate_request
from x2_frames import HOME_Q


class ValidationTests(unittest.TestCase):
    def test_bad_targets_are_rejected_before_dispatch(self):
        good = dict(id=1, op='mdi', mode='j', side='right', values=[0.] * 7)
        for changes in ({'values':[0.] * 6}, {'values':[math.nan] * 7}, {'values':[True] * 7},
                        {'side':'both'}, {'duration':0}, {'duration':math.inf}, {'settle':-1},
                        {'preview':'yes'}, {'mode':'shell'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_request({**good, **changes})
        self.assertEqual(validate_request(good), good)

    def test_all_state_actions_are_rejected(self):
        for action in ('UPPERBODY_REMOTE_SPLIT', 'STAND_DEFAULT', 'DAMPING_DEFAULT',
                       'JOINT_DEFAULT', 'PASSIVE_DEFAULT', 'reboot'):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'MDI 不提供模式切换'):
                validate_request(dict(id=1, op='action', action=action))

    def test_busy_drops_commands_and_eof_cancels(self):
        transport = Transport(io.StringIO('{"id":1,"op":"home"}\n'), io.StringIO())
        transport.busy = True
        transport.read()
        self.assertTrue(transport.commands.empty())
        self.assertFalse(json.loads(transport.output.getvalue())['ok'])
        with self.assertRaises(Cancelled):
            transport.check_lease()

    def test_disarm_interrupts_busy_without_waiting_in_queue(self):
        transport = Transport(io.StringIO('{"id":1,"op":"disarm"}\n'), io.StringIO())
        transport.busy = True
        transport.read()
        self.assertTrue(transport.cancel.is_set())
        self.assertTrue(transport.commands.empty())
        self.assertTrue(json.loads(transport.output.getvalue())['ok'])

    def test_expired_heartbeat_does_not_authorize_motion(self):
        transport = Transport(io.StringIO(), io.StringIO())
        transport.last_heartbeat -= 2.1
        with self.assertRaises(Cancelled):
            transport.check_lease()


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.transport = Transport(io.StringIO(), io.StringIO())
        self.bridge = Bridge(self.transport, demo=True)

    def test_starts_readonly_and_does_not_load_ros(self):
        self.assertFalse(self.bridge.armed)
        self.assertIsNone(self.bridge.cli)
        self.assertEqual(set(self.bridge.snapshot()['arms']), {'left', 'right'})
        with self.assertRaises(ValueError):
            self.bridge.execute(dict(id=1, op='home'))

    def test_preview_preserves_feedback_and_does_not_enable(self):
        target = np.degrees(HOME_Q).tolist()
        target[0] += 2.
        self.bridge.execute(dict(id=1, op='mdi', mode='j', side='right', values=target, preview=True))
        np.testing.assert_array_equal(self.bridge.q('right'), HOME_Q)
        self.assertFalse(self.bridge.armed)

    def test_single_side_and_home_are_explicit(self):
        self.bridge.execute(dict(id=1, op='arm'))
        target = np.degrees(HOME_Q).tolist()
        target[0] += 2.
        self.bridge.execute(dict(id=2, op='mdi', mode='j', side='right', values=target))
        np.testing.assert_array_equal(self.bridge.q('left'), HOME_Q)
        self.assertNotEqual(self.bridge.q('right')[0], HOME_Q[0])
        self.bridge.execute(dict(id=3, op='home'))
        np.testing.assert_array_equal(self.bridge.q('right'), HOME_Q)

    def test_mode_switch_cannot_be_enabled_by_environment_or_crafted_request(self):
        with patch.dict('os.environ', {'X2_URS_ONLY': '0', 'X2_MDI_ALLOW_STATE_SWITCH': '1'}):
            bridge = Bridge(self.transport, demo=True)
            bridge.armed = True
            bridge.cli = Mock()
            for action in ('STAND_DEFAULT', 'UPPERBODY_REMOTE_SPLIT'):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'MDI 不提供模式切换'):
                    bridge.execute(dict(id=1, op='action', action=action, force=True))
            self.assertEqual(bridge.action, 'UPPERBODY_REMOTE_SPLIT')
            bridge.cli.node.create_client.assert_not_called()
            bridge.cli.set_action.assert_not_called()

    def test_non_urs_and_stale_mode_allow_feedback_but_reject_arm_and_motion(self):
        for action, age in (('STAND_DEFAULT', 0), (None, 0), ('UPPERBODY_REMOTE_SPLIT', 4)):
            with self.subTest(action=action, age=age):
                self.bridge.action = action
                self.bridge.action_at = time.monotonic() - age
                self.bridge.armed = False
                state = self.bridge.snapshot()
                self.assertTrue(state['fresh'])
                self.assertEqual(set(state['arms']), {'left', 'right'})
                self.assertFalse(state['urs_confirmed'])
                with self.assertRaisesRegex(ValueError, '仅支持已处于 URS'):
                    self.bridge.execute(dict(id=1, op='arm'))
                self.assertFalse(self.bridge.armed)
                self.bridge.execute(dict(id=2, op='home', preview=True))
                self.bridge.armed = True  # Cannot bypass the independent motion gate.
                with self.assertRaisesRegex(ValueError, '仅支持已处于 URS'):
                    self.bridge.execute(dict(id=3, op='home'))
                np.testing.assert_array_equal(self.bridge.q('right'), HOME_Q)

    def test_reenable_requires_new_lease_after_timeout(self):
        self.transport.last_heartbeat -= 2.1
        with self.assertRaises(Cancelled):
            self.bridge.execute(dict(id=1, op='arm'))
        self.assertFalse(self.bridge.armed)


class ProtocolProcessTests(unittest.TestCase):
    def test_native_diagnostics_never_enter_jsonl_even_at_exit(self):
        script = '''
import atexit, os
import x2_mdi_bridge as module
class NoisyBridge:
    def __init__(self, transport, demo=False):
        self.transport = transport
    def run(self):
        os.write(1, b'native DDS diagnostic\\n')
        print('Python diagnostic')
        self.transport.emit({'type': 'state', 'armed': False})
        atexit.register(lambda: os.write(1, b'native destructor diagnostic\\n'))
        return 0
module.Bridge = NoisyBridge
raise SystemExit(module.main(['--demo']))
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'type': 'state', 'armed': False})
        self.assertIn('native DDS diagnostic', result.stderr)
        self.assertIn('native destructor diagnostic', result.stderr)
        self.assertIn('Python diagnostic', result.stderr)

    def test_json_stream_disconnect_closes_without_ros(self):
        proc = subprocess.Popen([sys.executable, str(ROOT/'x2_mdi_bridge.py'), '--demo'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, cwd=str(ROOT))
        messages = queue.Queue()
        def read():
            for line in proc.stdout:
                messages.put(json.loads(line))
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        try:
            state = messages.get(timeout=5)
            self.assertEqual(state['type'], 'state')
            self.assertTrue(state['demo'])
            self.assertFalse(state['armed'])
            proc.stdin.write('{"id":1,"op":"home"}\n')
            proc.stdin.flush()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                value = messages.get(timeout=3)
                if value.get('id') == 1:
                    self.assertFalse(value['ok'])
                    break
            else:
                self.fail('no result')
            proc.stdin.close()
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            proc.stderr.close()


if __name__ == '__main__':
    unittest.main()

class OnlinePathWithFakeRosTests(unittest.TestCase):
    """Exercise the real Session/MoveJ loop with fake feedback, never ROS."""
    def test_initialization_keeps_feedback_live_and_cancel_sends_nothing(self):
        import x2_mdi_bridge as bridge_module
        import x2_sim_ros as ros
        from test_x2_mdi import FakeClient
        fake = FakeClient()
        fake.enable_commands = Mock()
        fake.disable_commands = Mock()
        clock = [10.]
        output = io.StringIO()
        with patch.object(bridge_module.time, 'monotonic', side_effect=lambda: clock[0]):
            transport = Transport(io.StringIO(), output)
            def spin(seconds):
                clock[0] += seconds
                fake.spin_once()
                if clock[0] >= 10.5:
                    transport.cancel.set()
            fake.spin = spin
            query = Mock()
            query.poll.return_value = (True, 'UPPERBODY_REMOTE_SPLIT')
            with patch.dict('os.environ', {'X2_ROBOT_SN': ''}), \
                    patch.object(ros, 'X2ArmClient', return_value=fake), \
                    patch.object(bridge_module, 'ActionQuery', return_value=query):
                bridge = Bridge(transport)
                bridge.poll_action()
                bridge.execute(dict(id=1, op='arm'))
                with self.assertRaises(Cancelled):
                    bridge.execute(dict(id=2, op='home'))
                bridge.disarm()
            states = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertGreaterEqual(len(states), 2)
            self.assertTrue(all(s['fresh'] for s in states))
            self.assertEqual(fake.sent, [])
            self.assertLess(clock[0], 11.)
            fake.set_action.assert_not_called()

    def test_cancel_stops_synchronous_move_before_next_frame(self):
        import x2_mdi_bridge as bridge_module
        import x2_mdi
        import x2_sim_ros as ros
        from test_x2_mdi import FakeClient
        fake = FakeClient()
        fake.enable_commands = Mock()
        fake.disable_commands = Mock()
        transport = Transport(io.StringIO(), io.StringIO())
        send = fake.send
        def send_then_cancel(*args, **kwargs):
            send(*args, **kwargs)
            if len(fake.sent) == 3:
                transport.cancel.set()
        fake.send = send_then_cancel
        query = Mock()
        query.poll.return_value = (True, 'UPPERBODY_REMOTE_SPLIT')
        with patch.dict('os.environ', {'X2_ROBOT_SN':''}), \
                patch.object(ros, 'X2ArmClient', return_value=fake) as factory, \
                patch.object(bridge_module, 'ActionQuery', return_value=query), \
                patch.object(x2_mdi, 'ActionQuery', return_value=query):
            bridge = Bridge(transport)
            self.assertTrue(factory.call_args.kwargs['read_only'])
            self.assertEqual(len(fake.sent), 0)
            bridge.poll_action()
            bridge.execute(dict(id=1, op='arm'))
            self.assertEqual(len(fake.sent), 0)
            with self.assertRaises(Cancelled):
                bridge.execute(dict(id=2, op='home', duration=.2, settle=0))
            bridge.disarm()
            self.assertEqual(len(fake.sent), 3)
            self.assertFalse(bridge.armed)
            self.assertIsNone(bridge.session)
            fake.disable_commands.assert_called_once()
            fake.set_action.assert_not_called()
