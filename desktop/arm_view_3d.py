"""Software-rendered 3D feedback view; camera interaction never sends commands.

All points use torso_link metres. RPY is Rz(yaw) @ Ry(pitch) @ Rx(roll).
The wireframe torso and grid are visual references, not measured body geometry.
"""
from __future__ import annotations

import copy
import math

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from .arm_mesh import ArmMeshModel
except ImportError:
    from arm_mesh import ArmMeshModel


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _vector(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def orientation_axes(rpy_deg):
    """Return the TCP X/Y/Z unit vectors expressed in torso coordinates."""
    roll, pitch, yaw = map(math.radians, rpy_deg)
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                             math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return ((cy * cp, sy * cp, -sp),
            (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
            (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr))


class ArmView(QtWidgets.QWidget):
    """Orbitable orthographic 3D display of independent feedback snapshots.

    set_view accepts default/front/side/top; reset_view restores the default
    orbit, zoom and pan. project_point returns (QPointF, depth), where larger
    depth means closer to the camera. These are display operations only.
    """

    VIEWS = {'default': (32., 22.), 'front': (0., 0.),
             'side': (90., 0.), 'top': (0., 90.)}
    COLORS = {'left': '#55dcb7', 'right': '#ffbf69'}
    AXIS_COLORS = ('#f67c83', '#72d49d', '#7daff9')
    GRID_Z = -.35

    def __init__(self, parent=None):
        super().__init__(parent)
        self.arms = {}
        self.render_mode = 'mesh'
        self._posed_meshes = {}
        self._mesh_sources = {}
        self._mesh_failures = {}
        try:
            self._mesh_model = ArmMeshModel()
            self._mesh_error = ''
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._mesh_model = None
            self._mesh_error = str(exc)
        self.azimuth, self.elevation = self.VIEWS['default']
        self.zoom = 1.
        self.pan = QtCore.QPointF()
        self._drag_position = None
        self._drag_button = None
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setToolTip('左键拖动旋转视角；滚轮缩放；右键 / 中键拖动平移；双击恢复默认视角。')

    def set_arms(self, arms):
        """Replace feedback; invalid or missing arms cannot leave stale limbs."""
        valid = {}
        if isinstance(arms, dict):
            for side in self.COLORS:
                arm = arms.get(side)
                if not isinstance(arm, dict):
                    continue
                points = arm.get('points')
                if (not isinstance(points, (list, tuple)) or not 2 <= len(points) <= 32
                        or not all(_vector(p) for p in points)
                        or not _vector(arm.get('xyz')) or not _vector(arm.get('rpy_deg'))):
                    continue
                valid[side] = copy.deepcopy(arm)
        self.arms = valid
        self._posed_meshes = {}
        self._mesh_sources = {}
        self._mesh_failures = {}
        if self._mesh_model is not None:
            for side, arm in valid.items():
                try:
                    links, source = self._mesh_model.posed_links(side, arm)
                    self._posed_meshes[side] = links
                    self._mesh_sources[side] = source
                except (ValueError, KeyError, TypeError) as exc:
                    self._mesh_failures[side] = str(exc)
        self.update()

    def set_render_mode(self, mode):
        """Select a display style only; never changes robot or TCP data."""
        if mode not in ('mesh', 'skeleton'):
            raise ValueError('Unknown arm render mode: ' + str(mode))
        self.render_mode = mode
        self.update()

    def render_status(self):
        if self.render_mode == 'skeleton':
            return '骨架示意'
        if self._mesh_model is None:
            return '骨架降级：实体资源不可用'
        if self._mesh_failures:
            sides = '、'.join('左臂' if side == 'left' else '右臂' for side in self._mesh_failures)
            return sides + '骨架降级：关节位姿无效'
        if 'packaged_fk' in self._mesh_sources.values():
            return 'STL 实体 · 本地模型 FK'
        return 'STL 实体 · 反馈位姿'

    def set_view(self, name):
        if name not in self.VIEWS:
            raise ValueError('Unknown view: ' + str(name))
        self.azimuth, self.elevation = self.VIEWS[name]
        self.zoom = 1.
        self.pan = QtCore.QPointF()
        self.update()

    def reset_view(self):
        self.set_view('default')

    def _basis(self):
        az, el = map(math.radians, (self.azimuth, self.elevation))
        ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
        # Camera is on +X in front view, looking toward -X. Robot +Y (its
        # left arm) is screen-right. right x up = eye: no mirror reflection.
        return ((-sa, ca, 0.), (-se * ca, -se * sa, ce), (ce * ca, ce * sa, se))

    def _scale(self):
        return max(1., min((self.width() - 96.) / 1.05, (self.height() - 98.) / .83)) * self.zoom

    def project_point(self, point):
        right, up, eye = self._basis()
        centered = (point[0], point[1], point[2] + .035)
        scale = self._scale()
        return (QtCore.QPointF(self.width() * .5 + self.pan.x() + _dot(centered, right) * scale,
                              self.height() * .49 + self.pan.y() - _dot(centered, up) * scale),
                _dot(centered, eye))

    def _project_vertices(self, vertices):
        """Batch projection avoids rebuilding the camera per mesh vertex."""
        right, up, eye = self._basis()
        scale = self._scale()
        center_x = self.width() * .5 + self.pan.x()
        center_y = self.height() * .49 + self.pan.y()
        result = []
        for x, y, z in vertices:
            z += .035
            result.append((QtCore.QPointF(center_x + (x*right[0]+y*right[1]+z*right[2])*scale,
                                          center_y - (x*up[0]+y*up[1]+z*up[2])*scale),
                           x*eye[0]+y*eye[1]+z*eye[2]))
        return result

    def mousePressEvent(self, event):
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
            self._drag_position = event.pos()
            self._drag_button = event.button()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_position is None:
            return super().mouseMoveEvent(event)
        delta = event.pos() - self._drag_position
        self._drag_position = event.pos()
        if self._drag_button == QtCore.Qt.LeftButton:
            self.azimuth = (self.azimuth - delta.x() * .45) % 360.
            self.elevation = max(-89., min(89., self.elevation + delta.y() * .4))
        else:
            self.pan += QtCore.QPointF(delta)
            self.pan.setX(max(-self.width() * .6, min(self.width() * .6, self.pan.x())))
            self.pan.setY(max(-self.height() * .6, min(self.height() * .6, self.pan.y())))
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_position = self._drag_button = None
        self.setCursor(QtCore.Qt.OpenHandCursor)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.reset_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        steps = max(-8., min(8., event.angleDelta().y() / 120.))
        self.zoom = max(.45, min(3., self.zoom * 1.12 ** steps))
        self.update()
        event.accept()

    @staticmethod
    def _pen(color, width=1., style=QtCore.Qt.SolidLine):
        return QtGui.QPen(QtGui.QColor(color), width, style, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)

    def _line(self, painter, first, second, color, width=1., style=QtCore.Qt.SolidLine):
        painter.setPen(self._pen(color, width, style))
        painter.drawLine(self.project_point(first)[0], self.project_point(second)[0])

    def _label(self, painter, point, text, color):
        bounds = painter.fontMetrics().boundingRect(text)
        rect = QtCore.QRectF(point.x() + 10, point.y() - 12, bounds.width() + 14, 23)
        rect.moveLeft(max(8., min(self.width() - rect.width() - 8., rect.x())))
        rect.moveTop(max(38., min(self.height() - 53., rect.y())))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor('#172638'))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QtGui.QColor(color))
        painter.drawText(rect, QtCore.Qt.AlignCenter, text)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        background = QtGui.QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0., QtGui.QColor('#152537'))
        background.setColorAt(1., QtGui.QColor('#0e1927'))
        painter.fillRect(self.rect(), background)
        painter.save()
        painter.setClipRect(0, 36, self.width(), self.height() - 80)

        # The XY reference grid sits below the arms, not at an inferred floor.
        for k in range(-5, 6):
            value = k * .1
            color = '#314458' if k == 0 else '#223448'
            self._line(painter, (-.5, value, self.GRID_Z), (.5, value, self.GRID_Z), color)
            self._line(painter, (value, -.5, self.GRID_Z), (value, .5, self.GRID_Z), color)
        for side, arm in self.arms.items():
            tcp = arm['xyz']
            foot = (tcp[0], tcp[1], self.GRID_Z)
            color = QtGui.QColor(self.COLORS[side])
            color.setAlpha(65)
            self._line(painter, tcp, foot, color, 1., QtCore.Qt.DotLine)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(self._pen(color))
            painter.drawEllipse(self.project_point(foot)[0], 4., 4.)

        # Sort all arm links, joints and torso faces together by camera depth.
        primitives = []
        vertices = [(x, y, z) for z in (-.09, .24) for y in (-.115, .115) for x in (-.055, .055)]
        for indices in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                        (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)):
            points = [vertices[i] for i in indices]
            primitives.append((sum(self.project_point(p)[1] for p in points) / 4., 'torso', points, None))
        for side, arm in self.arms.items():
            if self.render_mode == 'mesh' and side in self._posed_meshes:
                eye = self._basis()[2]
                for link in self._posed_meshes[side]:
                    projected = self._project_vertices(link['vertices'])
                    for face, normal in zip(link['faces'], link['normals']):
                        facing = _dot(normal, eye)
                        if facing <= 0.:
                            continue
                        # Directional lighting plus camera fill. Mesh is coloured
                        # by arm so the same L/R convention survives all views.
                        light = max(0., _dot(normal, (.35, -.2, .915)))
                        shade = int(max(58, min(112, 62 + 30*light + 18*facing)))
                        depth = sum(projected[i][1] for i in face) / 3.
                        polygon = QtGui.QPolygonF([projected[i][0] for i in face])
                        primitives.append((depth, 'mesh', (polygon, shade), side))
            else:
                for a, b in zip(arm['points'], arm['points'][1:]):
                    depth = (self.project_point(a)[1] + self.project_point(b)[1]) * .5
                    primitives.append((depth, 'link', (a, b), side))
                # Coincident wrist frames should not stack opaque circles.
                unique = {tuple(point) for point in arm['points']}
                for point in unique:
                    primitives.append((self.project_point(point)[1] + .00001, 'joint', point, side))
            primitives.append((self.project_point(arm['xyz'])[1] + .00002, 'tcp', arm['xyz'], side))
        mesh_brushes = {}
        for depth, kind, points, side in sorted(primitives, key=lambda item: item[0]):
            if kind == 'mesh':
                polygon, shade = points
                key = (side, shade)
                if key not in mesh_brushes:
                    base = QtGui.QColor(self.COLORS[side])
                    mesh_brushes[key] = QtGui.QColor(*(min(255, int(c * shade / 100.))
                        for c in (base.red(), base.green(), base.blue())))
                # Avoid antialias seams between adjacent triangles. The mesh
                # silhouette remains dense enough for software rendering.
                painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(mesh_brushes[key])
                painter.drawPolygon(polygon)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            elif kind == 'torso':
                painter.setPen(self._pen('#466075', 1., QtCore.Qt.DashLine))
                painter.setBrush(QtGui.QColor(32, 50, 65, 90))
                painter.drawPolygon(QtGui.QPolygonF([self.project_point(v)[0] for v in points]))
            elif kind == 'link':
                a, b = (self.project_point(v)[0] for v in points)
                painter.setPen(self._pen('#0c1520', 11.))
                painter.drawLine(a, b)
                color = QtGui.QColor(self.COLORS[side])
                # Subtle brightness cue, independent of geometric scale.
                color = color.darker(int(max(100, min(135, 111 - depth * 35))))
                painter.setPen(self._pen(color, 7.))
                painter.drawLine(a, b)
                highlight = QtGui.QColor(self.COLORS[side]).lighter(125)
                highlight.setAlpha(100)
                painter.setPen(self._pen(highlight, 2.))
                painter.drawLine(a + QtCore.QPointF(-1, -1), b + QtCore.QPointF(-1, -1))
            else:
                center = self.project_point(points)[0]
                painter.setPen(self._pen(self.COLORS[side], 2.))
                painter.setBrush(QtGui.QColor(self.COLORS[side] if kind == 'tcp' else '#142435'))
                radius = 6. if kind == 'tcp' else 4.
                painter.drawEllipse(center, radius, radius)
                if kind == 'tcp':
                    painter.setPen(self._pen('#eaf4fc', 1.))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(center, 8.5, 8.5)

        # The fixed torso origin and TCP local axes share the same RGB convention.
        origin = (0., 0., 0.)
        for i, color in enumerate(self.AXIS_COLORS):
            endpoint = [0., 0., 0.]
            endpoint[i] = .145
            self._line(painter, origin, endpoint, color, 1.5, QtCore.Qt.DashLine)
            p = self.project_point(endpoint)[0]
            painter.drawText(p + QtCore.QPointF(4, -4), 'XYZ'[i])
        for side, arm in self.arms.items():
            tcp = arm['xyz']
            for direction, color in zip(orientation_axes(arm['rpy_deg']), self.AXIS_COLORS):
                endpoint = [tcp[i] + .067 * direction[i] for i in range(3)]
                self._line(painter, tcp, endpoint, color, 2.)
            self._label(painter, self.project_point(tcp)[0], ('L' if side == 'left' else 'R') + ' · TCP', self.COLORS[side])
        painter.restore()

        font = painter.font()
        font.setPixelSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor('#e1ecf6'))
        painter.drawText(17, 25, '3D 双臂反馈 · ' + self.render_status())
        font.setBold(False)
        font.setPixelSize(12)
        painter.setFont(font)
        for side, text, offset in (('left', '左臂', 124), ('right', '右臂', 62)):
            painter.setBrush(QtGui.QColor(self.COLORS[side]))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(self.width() - offset, 20), 3.5, 3.5)
            painter.setPen(QtGui.QColor('#c7d5e4'))
            painter.drawText(self.width() - offset + 10, 25, text)
        painter.setPen(QtGui.QColor('#a4b6c9'))
        painter.drawText(17, self.height() - 27, 'torso_link · 网格 0.1 m · 虚线躯干仅为示意')
        painter.setPen(QtGui.QColor('#8197ad'))
        painter.drawText(17, self.height() - 9, '左键旋转  /  滚轮缩放  /  右键平移  /  双击复位')
        if not self.arms:
            painter.setPen(QtGui.QColor('#c3d2e3'))
            painter.drawText(self.rect().adjusted(0, 10, 0, 0), QtCore.Qt.AlignCenter, '等待双臂实时反馈')
        painter.end()
