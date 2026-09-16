#!/usr/bin/env python3
"""X2 native Qt MDI console. SSH stdio only; startup never connects or sends.

--demo is a local UI/forward-kinematics demonstration. It has no robot transport
and intentionally does not pretend to solve Cartesian inverse kinematics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets

# Permit both the standalone script and imports from the project test runner.
if not getattr(sys, 'frozen', False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_view_3d import ArmView

MODES = {
    'xyz': ('绝对位置', ('X', 'Y', 'Z')),
    'pose': ('绝对位姿', ('X', 'Y', 'Z', 'Rx', 'Ry', 'Rz')),
    'd': ('躯干系平移', ('dX', 'dY', 'dZ')),
    'R': ('绕躯干系旋转', ('Rx', 'Ry', 'Rz')),
    't': ('绕 TCP 系旋转', ('Rx', 'Ry', 'Rz')),
    'rpy': ('绝对姿态', ('Roll', 'Pitch', 'Yaw')),
    'j': ('关节目标', ('J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7')),
}
HOME_DEG = [math.degrees(.4), 0., 0., math.degrees(-1.2), 0., 0., 0.]
URS_NOTICE = '仅支持 URS 模式；请先使用机器人原有操作方式切换到 URS，MDI 不提供模式切换。'


def is_urs(action):
    return isinstance(action, str) and re.fullmatch(
        r'(?:URS|US|UPPERBODY_REMOTE_SPLIT)(?:\(\d+\))?', action.strip()) is not None


def ssh_arguments(host, remote, config, python='python3', identity=''):
    """One quoted remote shell command; no locally evaluated shell or options."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+', host) or '@-' in host:
        raise ValueError('SSH 地址应为 user@host，不包含空格或选项')
    if not remote.startswith('/') or not config.startswith('/'):
        raise ValueError('机器人运行目录和配置必须为绝对路径')
    if any('\x00' in value or '\n' in value for value in (remote, config, python, identity)):
        raise ValueError('路径不能包含换行或空字符')
    command = 'cd -- {} && exec env {} {} -m x2ik mdi --stdio'.format(
        shlex.quote(remote), shlex.quote('X2IK_CONFIG=' + config), shlex.quote(python))
    args = ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=2',
            '-o', 'ServerAliveCountMax=2']
    if identity.strip():
        args += ['-i', os.path.expanduser(identity.strip())]
    return args + [host, command]


def complete_arms(arms):
    if not isinstance(arms, dict):
        return False
    try:
        for side in ('left', 'right'):
            arm = arms[side]
            for name, count in (('q_deg', 7), ('xyz', 3), ('rpy_deg', 3)):
                if len(arm[name]) != count or not all(type(v) in (int, float) and math.isfinite(v) for v in arm[name]):
                    return False
            if len(arm['points']) < 2 or len(arm['points']) > 32:
                return False
            if any(len(p) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in arm['points']):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


class SshBackend(QtCore.QObject):
    message = QtCore.pyqtSignal(dict)
    log = QtCore.pyqtSignal(str)
    disconnected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.buffer = b''
        self.closing = False
        self.counter = 0

    def connect_to(self, host, remote, config, python, identity):
        args = ssh_arguments(host, remote, config, python, identity)
        self.closing = False
        self.buffer = b''
        self.process.start('ssh', args)

    def send(self, op, **kw):
        if self.process.state() != QtCore.QProcess.Running:
            raise RuntimeError('SSH 尚未连接')
        self.counter += 1
        payload = json.dumps(dict(id=self.counter, op=op, **kw), ensure_ascii=False, allow_nan=False)
        if self.process.write((payload + '\n').encode()) < 0:
            raise RuntimeError('SSH 写入失败')
        return self.counter

    def _read(self):
        self.buffer += bytes(self.process.readAllStandardOutput())
        if len(self.buffer) > 262144:
            self.log.emit('SSH 协议输出过长；关闭连接')
            self.close()
            return
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or value.get('type') not in ('state', 'result', 'event'):
                    raise ValueError('未知消息')
                self.message.emit(value)
            except (ValueError, UnicodeError) as exc:
                self.log.emit('SSH 协议错误：' + str(exc))
                self.close()
                return

    def _stderr(self):
        data = bytes(self.process.readAllStandardError()).decode('utf-8', 'replace').strip()
        if data:
            self.log.emit(data[-6000:])

    def _error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.disconnected.emit('无法启动 SSH：' + self.process.errorString())

    def _finished(self, code, _status):
        self.disconnected.emit('连接已关闭' if self.closing else f'SSH 已退出（{code}），请检查日志后手动重连')

    def close(self):
        self.closing = True
        if self.process.state() == QtCore.QProcess.Running:
            try:
                self.send('close')
            except RuntimeError:
                pass
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(400):
                self.process.terminate()
                if not self.process.waitForFinished(400):
                    self.process.kill()
                    self.process.waitForFinished(400)
        elif self.process.state() == QtCore.QProcess.Starting:
            self.process.kill()
            self.process.waitForFinished(400)
        self.buffer = b''


class DemoBackend(QtCore.QObject):
    message = QtCore.pyqtSignal(dict)
    log = QtCore.pyqtSignal(str)
    disconnected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.snapshot)
        self.active = self.armed = self.busy = False
        self.action = 'UPPERBODY_REMOTE_SPLIT'
        self.q = {s: HOME_DEG.copy() for s in ('left', 'right')}
        self.counter = 0
        self.history = []

    def connect_to(self, *_args):
        self.active = True
        self.armed = False
        self.timer.start(200)
        QtCore.QTimer.singleShot(0, self.snapshot)

    def snapshot(self):
        if self.active:
            self.message.emit(dict(type='state', demo=True, connected=True,
                armed=self.armed, busy=self.busy, action=self.action, fresh=True,
                urs_confirmed=is_urs(self.action), arms={s: demo_fk(s, q) for s, q in self.q.items()}))

    def send(self, op, **kw):
        if not self.active:
            raise RuntimeError('演示未连接')
        self.counter += 1
        request_id = self.counter
        self.history.append(dict(op=op, **kw))
        if op == 'heartbeat':
            return request_id
        def handle():
            if not self.active:
                return
            ok, text = True, '离线演示；未连接机器人'
            if op == 'arm':
                if not is_urs(self.action):
                    ok, text = False, URS_NOTICE
                else:
                    self.armed = True
            elif op in ('disarm', 'close'):
                self.armed = False
            elif op in ('mdi', 'home'):
                preview = kw.get('preview', False)
                if not preview and (not self.armed or not is_urs(self.action)):
                    ok, text = False, '演示也要求先启用发送并进入 URS'
                elif op == 'mdi' and kw['mode'] != 'j':
                    ok, text = False, '本地演示仅模拟关节模式/HOME；笛卡尔 IK 预检需连接机器人后端'
                elif preview:
                    self.armed = False
                    text = '演示输入有效；未计算 IK/碰撞，不发送运动'
                elif op == 'home':
                    self.q = {s: HOME_DEG.copy() for s in self.q}
                else:
                    self.q[kw['side']] = kw['values'].copy()
            elif op != 'state':
                ok, text = False, URS_NOTICE if op == 'action' else '不支持的请求'
            self.message.emit(dict(type='result', id=request_id, ok=ok, message=text))
            self.snapshot()
        QtCore.QTimer.singleShot(0, handle)
        return request_id

    def close(self):
        self.active = self.armed = self.busy = False
        self.timer.stop()
        self.disconnected.emit('演示已断开')


def input_group(text, editor):
    """Keep the label/unit and its editor within one visible input boundary."""
    group = QtWidgets.QFrame()
    group.setObjectName('inputGroup')
    layout = QtWidgets.QHBoxLayout(group)
    layout.setContentsMargins(10, 2, 4, 2)
    layout.setSpacing(8)
    label = QtWidgets.QLabel(text)
    label.setObjectName('inputPrefix')
    label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
    editor.setMinimumWidth(64)
    editor.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    editor.setMinimumHeight(34)
    layout.addWidget(label)
    layout.addWidget(editor, 1)
    label.setBuddy(editor)
    return group, label


class Window(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backend = None
        self.state = {}
        self.last_state = 0.
        self.pending = None
        self.inhibit = True
        self.connecting = False
        self.setWindowTitle('X2 · MDI 桌面控制台' + (' [离线演示]' if args.demo else ''))
        self.resize(1140, 900)
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(250)
        self.last_heartbeat = 0.
        self.refresh_controls()

    def _build(self):
        self.setStyleSheet('''QWidget { background: #f3f6fa; color: #172b43; font-size: 13px; }
            QGroupBox { font-weight: bold; border: 1px solid #cbd6e2; border-radius: 6px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            QPushButton { background: white; border: 1px solid #bacadb; padding: 8px 12px; border-radius: 4px; }
            QPushButton:hover { border-color: #107d8b; background: #e9f5f6; }
            QPushButton:disabled { color: #97a3b0; background: #e8edf2; }
            QLineEdit, QComboBox, QDoubleSpinBox { background: white; padding: 5px; border: 1px solid #bbc9d8; border-radius: 3px; }
            QPlainTextEdit { background: #162335; color: #cfdeec; border-radius: 4px; }
            QFrame#inputGroup { background: white; border: 1px solid #bacadb; border-radius: 6px; }
            QLabel#inputPrefix { background: transparent; color: #52657a; font-size: 12px; }
            QFrame#inputGroup QLineEdit, QFrame#inputGroup QComboBox,
            QFrame#inputGroup QDoubleSpinBox { border: 0; background: transparent; padding: 4px; }

        ''')
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        root = QtWidgets.QVBoxLayout(content)
        title = QtWidgets.QLabel('X2  /  MDI 桌面控制台')
        title.setStyleSheet('font-size: 23px; font-weight: bold; padding: 2px;')
        root.addWidget(title)
        note = '离线演示 · 本地 FK 示例 · 无 SSH/机器人连接；笛卡尔 IK 需真实后端' if self.args.demo else 'SSH 加密连接 · 连接后默认只读 · 位置 m / mm，角度 °'
        self.banner = QtWidgets.QLabel(note)
        self.banner.setStyleSheet('color: #8a570d;' if self.args.demo else 'color: #41627d;')
        root.addWidget(self.banner)
        self.urs_notice = QtWidgets.QLabel(URS_NOTICE)
        self.urs_notice.setWordWrap(True)
        self.urs_notice.setStyleSheet('color: #41627d;')
        root.addWidget(self.urs_notice)
        connection = QtWidgets.QGroupBox('连接设置')
        grid = QtWidgets.QGridLayout(connection)
        self.host = QtWidgets.QLineEdit(self.args.host)
        self.remote = QtWidgets.QLineEdit(self.args.remote)
        self.config = QtWidgets.QLineEdit(self.args.config)
        self.identity = QtWidgets.QLineEdit(self.args.identity)
        self.identity.setPlaceholderText('可留空使用 SSH agent / ~/.ssh/config')
        for row, label, field in ((0, 'SSH 地址', self.host), (1, '运行目录', self.remote), (2, '机器人配置', self.config), (3, '本机私钥', self.identity)):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(field, row, 1)
        self.connect_button = QtWidgets.QPushButton('连接演示' if self.args.demo else '连接（只读）')
        self.connect_button.clicked.connect(self.connect_backend)
        self.disconnect_button = QtWidgets.QPushButton('断开')
        self.disconnect_button.clicked.connect(self.disconnect_backend)
        grid.addWidget(self.connect_button, 0, 2, 2, 1)
        grid.addWidget(self.disconnect_button, 2, 2, 2, 1)
        root.addWidget(connection)
        self.status = QtWidgets.QLabel('未连接 · 未启用发送')
        self.status.setStyleSheet('font-weight: bold; padding: 4px;')
        root.addWidget(self.status)
        self.arm_view = ArmView()
        view_tools = QtWidgets.QHBoxLayout()
        view_tools.addWidget(QtWidgets.QLabel('观察视角'))
        self.view_buttons = {}
        for name, text in (('default', '默认 3D'), ('front', '正视'), ('side', '侧视'), ('top', '俯视')):
            button = QtWidgets.QPushButton(text)
            button.setToolTip('仅改变观察视角，不发送机器人指令')
            button.clicked.connect(lambda _checked=False, view=name: self.arm_view.set_view(view))
            self.view_buttons[name] = button
            view_tools.addWidget(button)
        view_tools.addStretch(1)
        root.addLayout(view_tools)
        root.addWidget(self.arm_view, 1)
        legend = QtWidgets.QLabel('<span style="color:#168875">● 左臂</span>　<span style="color:#aa651b">● 右臂</span>　圆点：关节 / TCP · RGB 轴：X / Y / Z · 3D 骨架反馈，不含碰撞模型')
        root.addWidget(legend)
        self.readout = QtWidgets.QLabel('左臂 —\n右臂 —')
        self.readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.readout.setFont(QtGui.QFont('monospace', 10))
        root.addWidget(self.readout)
        controls = QtWidgets.QGroupBox('MDI 单条指令')
        controls.setObjectName('mdiControls')
        layout = QtWidgets.QVBoxLayout(controls)
        modes = QtWidgets.QHBoxLayout()
        modes.setSpacing(10)
        self.mode = QtWidgets.QComboBox()
        for name, (label, _) in MODES.items():
            self.mode.addItem(name + ' · ' + label, name)
        self.side = QtWidgets.QComboBox()
        self.side.addItem('右臂', 'right')
        self.side.addItem('左臂', 'left')
        self.units = QtWidgets.QComboBox()
        self.units.addItems(['m', 'mm'])
        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(.2, 120.)
        self.duration.setValue(8.)
        self.duration.setSuffix(' s')
        self.settle = QtWidgets.QDoubleSpinBox()
        self.settle.setRange(0., 120.)
        self.settle.setValue(2.)
        self.settle.setSuffix(' s')
        for label, widget in (('模式', self.mode), ('手臂', self.side), ('位置单位', self.units), ('运动时长', self.duration), ('稳定时间', self.settle)):
            group, _ = input_group(label, widget)
            modes.addWidget(group, 2 if widget is self.mode else 1)
        layout.addLayout(modes)
        self.fields_layout = QtWidgets.QGridLayout()
        self.fields_layout.setHorizontalSpacing(10)
        self.fields_layout.setVerticalSpacing(10)
        self.fields = []
        layout.addLayout(self.fields_layout)
        self.mode.currentIndexChanged.connect(self.update_fields)
        self.units.currentIndexChanged.connect(self.update_field_units)
        self.update_fields()
        row = QtWidgets.QHBoxLayout()
        self.fill_button = QtWidgets.QPushButton('填入当前值')
        self.fill_button.clicked.connect(self.fill_current)
        self.preview_button = QtWidgets.QPushButton('预检（停止保持，不发送）')
        self.preview_button.clicked.connect(lambda: self.command(True))
        self.execute_button = QtWidgets.QPushButton('下发 MDI')
        self.execute_button.clicked.connect(lambda: self.command(False))
        self.home_button = QtWidgets.QPushButton('双臂 HOME')
        self.home_button.clicked.connect(self.home)
        for button in (self.fill_button, self.preview_button, self.execute_button, self.home_button):
            row.addWidget(button)
        layout.addLayout(row)
        root.addWidget(controls)
        actions = QtWidgets.QHBoxLayout()
        self.arm_button = QtWidgets.QPushButton('启用发送')
        self.arm_button.clicked.connect(self.arm)
        self.disarm_button = QtWidgets.QPushButton('停止发送')
        self.disarm_button.setStyleSheet('background: #8e3040; color: white; font-weight: bold;')
        self.disarm_button.clicked.connect(self.disarm)
        for button in (self.arm_button, self.disarm_button):
            actions.addWidget(button)
        outer.addLayout(actions)  # Stop remains visible even on a small display.
        self.reason = QtWidgets.QLabel('停止发送会关闭保持发布，不会自动回 HOME；这不是硬件急停。')
        self.reason.setWordWrap(True)
        outer.addWidget(self.reason)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(110)
        root.addWidget(self.log)

    def update_fields(self):
        while self.fields_layout.count():
            widget = self.fields_layout.takeAt(0).widget()
            if widget:
                widget.hide()
                widget.deleteLater()
        self.fields = []
        labels = MODES[self.mode.currentData()][1]
        columns = 4 if len(labels) == 7 else 3
        for column in range(4):
            self.fields_layout.setColumnStretch(column, 1 if column < columns else 0)
        for index, name in enumerate(labels):
            field = QtWidgets.QLineEdit('0')
            field.setValidator(QtGui.QDoubleValidator(-1.e8, 1.e8, 8, field))
            group, label = input_group(name, field)
            self.fields_layout.addWidget(group, index // columns, index % columns)
            self.fields.append((label, field))
        self.update_field_units()

    def update_field_units(self):
        mode = self.mode.currentData()
        for i, (label, field) in enumerate(self.fields):
            positional = mode in ('xyz', 'pose', 'd') and i < 3
            label.setText(MODES[mode][1][i] + (' [' + self.units.currentText() + ']' if positional else ' [°]'))
        self.units.setEnabled(mode in ('xyz', 'pose', 'd'))

    def append_log(self, message):
        self.log.appendPlainText(time.strftime('%H:%M:%S') + '  ' + str(message))

    def connect_backend(self):
        if self.backend is not None:
            return
        self.state = {}
        self.inhibit = True
        self.connecting = True
        self.backend = DemoBackend(self) if self.args.demo else SshBackend(self)
        self.backend.message.connect(self.on_message)
        self.backend.log.connect(self.append_log)
        self.backend.disconnected.connect(self.on_disconnect)
        try:
            self.backend.connect_to(self.host.text().strip(), self.remote.text().strip(),
                self.config.text().strip(), self.args.python, self.identity.text().strip())
            self.append_log('开始离线演示' if self.args.demo else '等待 SSH 与机器人反馈（主机密钥必须预先验证）')
        except (ValueError, RuntimeError) as exc:
            self.on_disconnect(str(exc))
        self.refresh_controls()

    def disconnect_backend(self):
        backend = self.backend
        if backend:
            backend.close()
            if self.backend is backend:
                self.on_disconnect('已断开')

    def on_disconnect(self, reason):
        backend, self.backend = self.backend, None
        self.state = {}
        self.pending = None
        self.last_state = 0.
        self.inhibit = True
        self.connecting = False
        self.arm_view.set_arms({})
        self.readout.setText('左臂 —\n右臂 —')
        self.append_log(reason)
        self.refresh_controls()
        if backend:
            backend.deleteLater()

    def on_message(self, value):
        if value['type'] == 'state':
            self.state = value
            self.last_state = time.monotonic()
            self.connecting = False
            if not value.get('connected') or not value.get('fresh') or not complete_arms(value.get('arms')):
                self.trip('反馈不完整或过期，停止发送')
            elif not self.urs_confirmed():
                self.trip('未确认 URS 模式，停止发送；请在机器人原有操作界面确认')
            arms = value.get('arms', {}) if self.is_fresh() else {}
            self.arm_view.set_arms(arms)
            lines = []
            for side, label in (('left', '左臂'), ('right', '右臂')):
                a = arms.get(side)
                if not a:
                    lines.append(label + ' — 无有效反馈')
                    continue
                values = lambda key, precision: '  '.join(f'{v:+.{precision}f}' for v in a[key])
                lines.append(f'{label}  XYZ [m] {values("xyz", 4)}    RPY [°] {values("rpy_deg", 1)}\n'
                             f'       J1…J7 [°] {values("q_deg", 1)}')
            self.readout.setText('\n'.join(lines))
        elif value['type'] == 'result':
            if self.pending and value.get('id') == self.pending[0]:
                if not value.get('ok'):
                    self.inhibit = True
                self.pending = None
            self.append_log(('完成：' if value.get('ok') else '拒绝：') + str(value.get('message', value.get('error', value.get('result', '')))))
        else:
            self.append_log('后端：' + str(value.get('message', '')))
        self.refresh_controls()

    def is_fresh(self):
        return bool(self.backend and self.state.get('connected') and self.state.get('fresh')
            and time.monotonic() - self.last_state < 2. and complete_arms(self.state.get('arms')))

    def urs_confirmed(self):
        return is_urs(self.state.get('action')) and self.state.get('urs_confirmed', True) is True

    def refresh_controls(self):
        fresh = self.is_fresh()
        busy = bool(self.state.get('busy') or self.pending)
        urs = self.urs_confirmed()
        armed = fresh and urs and self.state.get('armed', False) and not self.inhibit
        idle = fresh and not busy
        self.connect_button.setEnabled(self.backend is None)
        self.disconnect_button.setEnabled(self.backend is not None)
        for field in (self.host, self.remote, self.config, self.identity):
            field.setEnabled(self.backend is None and not self.args.demo)
        self.arm_button.setEnabled(idle and urs and not armed)
        self.arm_button.setToolTip('确认 URS 且双臂反馈新鲜后才能启用发送。' + URS_NOTICE)
        self.disarm_button.setEnabled(self.backend is not None)
        self.preview_button.setEnabled(idle)
        self.execute_button.setEnabled(idle and armed and urs)
        self.home_button.setEnabled(idle and armed and urs)
        self.fill_button.setEnabled(idle)
        connection = '连接中' if self.connecting else ('反馈正常' if fresh else ('反馈过期 / 不完整' if self.backend else '未连接'))
        self.status.setText(f'{connection}  ·  运控 {self.state.get("action") or "—"}  ·  '
            + ('已启用发送' if armed else '只读 / 发送关闭') + ('  ·  忙：不接受新指令' if busy else ''))
        self.reason.setText(('未确认 URS 模式，当前仅可查看反馈与预检；请在机器人原有操作界面确认。' if fresh and not urs else '')
            + '停止发送会关闭保持发布，不会自动回 HOME；这不是硬件急停。')

    def send(self, op, **kw):
        if not self.backend:
            return False
        try:
            request_id = self.backend.send(op, **kw)
            if op not in ('heartbeat', 'disarm', 'close'):
                self.pending = (request_id, op)
            self.refresh_controls()
            return True
        except (ValueError, RuntimeError) as exc:
            self.append_log(str(exc))
            self.disconnect_backend()
            return False

    def trip(self, reason):
        was_active = not self.inhibit
        self.inhibit = True
        if was_active:
            self.append_log(reason)
            self.send('disarm')

    def tick(self):
        if self.backend and self.state and not self.is_fresh():
            self.trip('反馈超过 2 秒未更新；发送已关闭，恢复后需要重新启用')
            self.arm_view.set_arms({})
            self.readout.setText('左臂 — 反馈过期\n右臂 — 反馈过期')
        if self.backend and self.state and time.monotonic() - self.last_heartbeat >= .5:
            self.last_heartbeat = time.monotonic()
            self.send('heartbeat')
        self.refresh_controls()

    def arm(self):
        if self.arm_button.isEnabled():
            self.inhibit = False
            if not self.send('arm'):
                self.inhibit = True

    def disarm(self):
        self.inhibit = True
        self.send('disarm')
        self.refresh_controls()

    def fill_current(self):
        if not self.is_fresh():
            return
        arm = self.state['arms'][self.side.currentData()]
        mode = self.mode.currentData()
        values = {'xyz': arm['xyz'], 'pose': arm['xyz'] + arm['rpy_deg'],
                  'rpy': arm['rpy_deg'], 'j': arm['q_deg']}.get(mode, [0.] * len(self.fields))
        for i, (_, field) in enumerate(self.fields):
            value = values[i] * (1000. if self.units.currentText() == 'mm' and mode in ('xyz', 'pose', 'd') and i < 3 else 1.)
            field.setText(f'{value:.6f}')

    def command(self, preview):
        if not (self.preview_button if preview else self.execute_button).isEnabled():
            return
        mode = self.mode.currentData()
        try:
            values = [float(field.text()) for _, field in self.fields]
            if not all(math.isfinite(v) for v in values):
                raise ValueError()
        except ValueError:
            self.append_log('输入错误：每个框必须是有限数值')
            return
        if self.units.currentText() == 'mm' and mode in ('xyz', 'pose', 'd'):
            values[:3] = [v / 1000. for v in values[:3]]
        if preview:
            self.inhibit = True
        self.send('mdi', mode=mode, side=self.side.currentData(), values=values,
                  duration=self.duration.value(), settle=self.settle.value(), preview=preview)

    def home(self):
        if self.home_button.isEnabled():
            self.send('home', duration=self.duration.value(), settle=self.settle.value(), preview=False)

    def closeEvent(self, event):
        self.timer.stop()
        self.disconnect_backend()
        event.accept()


def demo_fk(side, q_deg):
    """Pure-Python FK from the shipped URDF snapshot, for offline UI examples."""
    def mm(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    def mv(a, v):
        return [sum(a[i][j] * v[j] for j in range(3)) for i in range(3)]
    rot = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    pos, points = [0., 0., 0.], []
    for angle, geom in zip(q_deg, DEMO_GEOMETRY[side]):
        offset = mv(rot, geom['xyz'])
        pos = [a + b for a, b in zip(pos, offset)]
        points.append(pos.copy())
        x, y, z = geom['axis']
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        a = [[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
             [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
             [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]]
        rot = mm(mm(rot, geom['rot']), a)
    pitch = math.asin(max(-1., min(1., -rot[2][0])))
    roll, yaw = math.atan2(rot[2][1], rot[2][2]), math.atan2(rot[1][0], rot[0][0])
    return dict(q_deg=q_deg.copy(), xyz=pos, rpy_deg=[math.degrees(a) for a in (roll, pitch, yaw)], points=points + [pos.copy()])


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--host', default='user@robot')
    ap.add_argument('--remote', default='/home/agi/x2ik/runtime')
    ap.add_argument('--config', default='/home/agi/x2ik/x2ik.conf')
    ap.add_argument('--identity', default='')
    ap.add_argument('--python', default='python3')
    ap.add_argument('--smoke-test', action='store_true', help='Build real Qt window, exercise local demo and gates; never SSH')
    ap.add_argument('--screenshot', help='Optional PNG from smoke test or demo')
    return ap


def smoke_test(app, args):
    args.demo = True
    w = Window(args)
    w.show()
    def pump():
        for _ in range(5):
            app.processEvents()
    pump()
    assert w.backend is None and not w.execute_button.isEnabled()
    w.connect_button.click()
    pump()
    assert w.is_fresh() and complete_arms(w.state['arms'])
    assert not w.state['armed'] and not w.home_button.isEnabled()
    w.mode.setCurrentIndex(w.mode.findData('j'))
    w.fill_current()
    original = copy.deepcopy(w.state['arms']['right'])
    w.fields[0][1].setText('30')
    w.preview_button.click()
    pump()
    assert w.state['arms']['right'] == original and not w.execute_button.isEnabled()
    w.arm_button.click()
    pump()
    assert w.execute_button.isEnabled() and w.home_button.isEnabled()
    w.execute_button.click()
    assert not w.home_button.isEnabled()  # outstanding request gate
    pump()
    assert w.state['arms']['right']['q_deg'][0] == 30.
    assert w.state['arms']['right']['xyz'] != original['xyz']
    w.home_button.click()
    pump()
    assert w.state['arms']['right']['q_deg'] == HOME_DEG
    w.state['busy'] = True
    w.refresh_controls()
    assert not w.execute_button.isEnabled() and not w.arm_button.isEnabled()
    w.state['busy'] = False
    w.last_state -= 3.
    w.tick()
    pump()
    assert w.inhibit and not w.execute_button.isEnabled()
    w.backend.snapshot()
    pump()
    assert not w.execute_button.isEnabled()  # fresh again never rearms
    if args.screenshot:
        assert w.grab().save(args.screenshot)
    backend = w.backend
    w.close()
    assert not backend.active and not w.timer.isActive() and w.backend is None
    pump()
    print('x2-mdi-desktop smoke-test ok: Qt window, dual-arm FK, preview, explicit arm, MDI/HOME, busy/stale gates, shutdown; no SSH')
    return 0


# Geometry snapshot generated from x2_ultra.urdf; used only by --demo.
DEMO_GEOMETRY = {'left': [{'xyz': [0.00329906676354264, 0.143031506663431, 0.241647721005027],
           'rot': [[1.0, 0.0, 0.0],
                   [0.0, 0.9781476124727735, -0.20791163559024986],
                   [0.0, 0.20791163559024986, 0.9781476124727735]],
           'axis': [0.0, 1.0, 0.0]},
          {'xyz': [-0.000499991072907324, 0.0495, 0.0],
           'rot': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
           'axis': [1.0, 0.0, 0.0]},
          {'xyz': [0.00122423638490546, 0.0, -0.121194848262102],
           'rot': [[0.9999821458151389, 0.0, -0.005975621386120984],
                   [0.0, 1.0, 0.0],
                   [0.005975621386120984, 0.0, 0.9999821458151389]],
           'axis': [0.0, 0.0, 1.0]},
          {'xyz': [0.0140000000488655, 0.000199986464082369, -0.0829500000002385],
           'rot': [[0.9999821458151389, 0.0, 0.005975621386120984],
                   [0.0, 1.0, 0.0],
                   [-0.005975621386120984, 0.0, 0.9999821458151389]],
           'axis': [0.0, 1.0, 0.0]},
          {'xyz': [-0.0132797235062217, -9.75487744990788e-05, -0.120575783591803],
           'rot': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
           'axis': [0.0, 0.0, 1.0]},
          {'xyz': [0.000490001227040848, 0.0, -0.0819985359552045],
           'rot': [[0.9999821457952123, 0.0, -0.005975624720707577],
                   [0.0, 1.0, 0.0],
                   [0.005975624720707577, 0.0, 0.9999821457952123]],
           'axis': [0.0, 1.0, 0.0]},
          {'xyz': [0.0, 0.0, 0.0],
           'rot': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
           'axis': [1.0, 0.0, 0.0]}],
 'right': [{'xyz': [0.00330093831251838, -0.143030971450747, 0.241648201390685],
            'rot': [[1.0, 0.0, 0.0],
                    [0.0, 0.9781469141740645, 0.2079149208011662],
                    [0.0, -0.2079149208011662, 0.9781469141740645]],
            'axis': [0.0, 1.0, 0.0]},
           {'xyz': [-0.000499999994070091, -0.0495000000000002, 0.0],
            'rot': [[0.9999999881415237, 0.0, 0.00015400309231729766],
                    [0.0, 1.0, 0.0],
                    [-0.00015400309231729766, 0.0, 0.9999999881415237]],
            'axis': [1.0, 0.0, 0.0]},
           {'xyz': [0.000499999999999765, 0.0, -0.121199999999934],
            'rot': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'axis': [0.0, 0.0, 1.0]},
           {'xyz': [0.0139999999999662, -0.0002000000000143, -0.0829500000000745],
            'rot': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'axis': [0.0, 1.0, 0.0]},
           {'xyz': [-0.0140000017961392, 9.7526096753231e-05, -0.120694276209686],
            'rot': [[0.9999999994642363, -3.273419062641397e-05, 0.0],
                    [3.273419062641397e-05, 0.9999999994642363, 0.0],
                    [0.0, 0.0, 1.0]],
            'axis': [0.0, 0.0, 1.0]},
           {'xyz': [0.0, 0.0, -0.081799999999999],
            'rot': [[0.9999999994642363, 3.273419324878957e-05, 0.0],
                    [-3.273419324878957e-05, 0.9999999994642363, 0.0],
                    [0.0, 0.0, 1.0]],
            'axis': [0.0, 1.0, 0.0]},
           {'xyz': [0.0, 0.0, 0.0],
            'rot': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'axis': [1.0, 0.0, 0.0]}]}

def main(argv=None):
    args = parser().parse_args(argv)
    if args.smoke_test:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName('X2 MDI Desktop')
    if args.smoke_test:
        return smoke_test(app, args)
    window = Window(args)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
