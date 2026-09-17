"""Display-only STL arm geometry, independent of motion planning and transport.

Only the standard library is used. Pose feedback is preferred; legacy q_deg
feedback can use the packaged, verified arm chain. Tool/TCP choice never moves
an arm link. The simplified meshes are not collision or metrology geometry.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ASSET_PATH = Path(__file__).resolve().parent / 'assets' / 'arm_meshes.json'
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def _vector(value, count=3):
    return (isinstance(value, (list, tuple)) and len(value) == count
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def rotate(matrix, point):
    x, y, z = point
    return tuple(row[0]*x + row[1]*y + row[2]*z for row in matrix)


def multiply(first, second):
    return tuple(tuple(sum(first[i][k]*second[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                             math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
            (-sp, cp*sr, cp*cr))


def _axis_rotation(axis, angle):
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1. - math.cos(angle)
    return ((t*x*x+c, t*x*y-s*z, t*x*z+s*y),
            (t*x*y+s*z, t*y*y+c, t*y*z-s*x),
            (t*x*z-s*y, t*y*z+s*x, t*z*z+c))


def _normal(a, b, c):
    u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
    n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    size = math.sqrt(sum(x*x for x in n))
    return tuple(x/size for x in n) if size > 1e-15 else (0., 0., 0.)


def _valid_rotation(matrix):
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3 or not all(_vector(r) for r in matrix):
        return False
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[i][k]*matrix[j][k] for k in range(3)) - (i == j)) > 1e-5:
                return False
    a, b, c = matrix
    determinant = (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
                   + a[2]*(b[0]*c[1]-b[1]*c[0]))
    return abs(determinant-1.) <= 1e-5


class ArmMeshModel:
    def __init__(self, path=ASSET_PATH):
        data = json.loads(Path(path).read_text())
        if (not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('units') != 'm'
                or data.get('base_frame') != 'torso_link'):
            raise ValueError('Unsupported arm mesh schema or coordinate frame')
        self.metadata = {k: v for k, v in data.items() if k != 'arms'}
        self.arms = data['arms']
        for side in ('left', 'right'):
            links = self.arms[side]
            if not isinstance(links, list) or len(links) != 7:
                raise ValueError('Arm mesh must contain exactly seven links')
            for link in links:
                vertices, faces = link['vertices'], link['faces']
                if (not 3 <= len(vertices) <= 50000 or not all(_vector(p) for p in vertices)
                        or not 1 <= len(faces) <= 50000):
                    raise ValueError('Invalid arm mesh vertices or face count')
                if not all(isinstance(f, list) and len(f) == 3 and all(
                        type(i) is int and 0 <= i < len(vertices) for i in f) for f in faces):
                    raise ValueError('Invalid mesh face indices')
                for key in ('xyz', 'rpy', 'axis', 'visual_xyz', 'visual_rpy'):
                    if not _vector(link[key]):
                        raise ValueError('Invalid link transform: ' + key)
                if abs(sum(x*x for x in link['axis']) - 1.) > 1e-8:
                    raise ValueError('Joint axis must be a unit vector')
                # Bake URDF visual origin into link-local geometry once.
                rotation, offset = rpy_matrix(link['visual_rpy']), link['visual_xyz']
                link['local_vertices'] = [tuple(x+offset[i] for i, x in enumerate(rotate(rotation, p)))
                                          for p in vertices]
                link['normals'] = [_normal(*(link['local_vertices'][i] for i in face)) for face in faces]

    def link_transforms(self, side, q_deg):
        """FK of the packaged arm-only chain; no tool or TCP transform."""
        if side not in self.arms or not _vector(q_deg, 7):
            raise ValueError('Expected a valid side and seven finite joint angles')
        rotation, position = IDENTITY, (0., 0., 0.)
        transforms = []
        for link, angle in zip(self.arms[side], q_deg):
            offset = rotate(rotation, link['xyz'])
            position = tuple(position[i]+offset[i] for i in range(3))
            rotation = multiply(multiply(rotation, rpy_matrix(link['rpy'])),
                                _axis_rotation(link['axis'], math.radians(angle)))
            transforms.append({'xyz': position, 'rotation': rotation})
        return transforms

    def posed_links(self, side, arm):
        """Return (links, pose_source), refusing malformed transforms.

        If transforms are explicitly supplied but invalid, do not quietly use
        q_deg instead. The view displays a labelled skeleton fallback.
        """
        if side not in self.arms:
            raise ValueError('Unknown arm side')
        if 'link_transforms' in arm:
            transforms, source = arm['link_transforms'], 'feedback'
        else:
            transforms, source = self.link_transforms(side, arm.get('q_deg')), 'packaged_fk'
        if (not isinstance(transforms, (list, tuple)) or len(transforms) != 7
                or not all(isinstance(t, dict) and _vector(t.get('xyz'))
                           and _valid_rotation(t.get('rotation')) for t in transforms)):
            raise ValueError('Invalid link pose feedback')
        posed = []
        for link, transform in zip(self.arms[side], transforms):
            rotation, offset = transform['rotation'], transform['xyz']
            posed.append({'vertices': [tuple(x+offset[i] for i, x in enumerate(rotate(rotation, p)))
                                       for p in link['local_vertices']],
                          'normals': [rotate(rotation, n) for n in link['normals']],
                          'faces': link['faces']})
        return posed, source
