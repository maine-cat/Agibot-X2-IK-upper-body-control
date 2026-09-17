#!/usr/bin/env python3
"""Prepare display-only arm STL meshes. No robot or transport imports.

Build-time requirements: numpy, fast-simplification. The desktop reads only
JSON and needs neither mesh package. QEM collapses edges instead of deleting
arbitrary triangles; original mesh defects are recorded rather than repaired.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


SUFFIXES = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
            'wrist_yaw', 'wrist_pitch', 'wrist_roll')
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(node, attr, default='0 0 0'):
    return [float(x) for x in (node.get(attr, default) if node is not None else default).split()]


def chain(root, side):
    result = []
    parent = 'torso_link'
    for suffix in SUFFIXES:
        joint = root.find(f"joint[@name='{side}_{suffix}_joint']")
        if joint is None or joint.get('type') != 'revolute':
            raise ValueError('Missing revolute arm joint: ' + side + '_' + suffix)
        child = joint.find('child').get('link')
        if joint.find('parent').get('link') != parent:
            raise ValueError('Arm joints are not an ordered torso chain')
        result.append({'link': child, 'xyz': vector(joint.find('origin'), 'xyz'),
                       'rpy': vector(joint.find('origin'), 'rpy'),
                       'axis': vector(joint.find('axis'), 'xyz')})
        parent = child
    return result


def boundary_edges(faces):
    edges = collections.Counter(tuple(sorted((int(x), int(y))))
        for a, b, c in faces for x, y in ((a, b), (b, c), (c, a)))
    return sum(count == 1 for count in edges.values())


def read_stl(path):
    import numpy as np
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError('Expected a binary STL: ' + str(path))
    triangles = np.frombuffer(raw, dtype=np.dtype([
        ('normal', '<f4', (3,)), ('points', '<f4', (3, 3)), ('attr', '<u2')]),
        offset=84, count=count)['points'].reshape(-1, 3)
    vertices, indices = np.unique(triangles, axis=0, return_inverse=True)
    return vertices.astype(float), indices.reshape(-1, 3)


def simplify(vertices, faces):
    import fast_simplification
    import numpy as np
    original_bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    original_open = boundary_edges(faces)
    choices = []
    for aggression in (3, 4, 5, 6, 7):
        points, triangles = fast_simplification.simplify(
            vertices, faces, target_count=min(600, len(faces)-1), agg=aggression)
        error = float(np.abs(np.array([points.min(axis=0), points.max(axis=0)])
                             - original_bounds).max())
        open_edges = boundary_edges(triangles)
        if error <= .003 and open_edges <= original_open:
            choices.append((len(triangles), points, triangles, error, open_edges, aggression))
    if not choices:
        raise ValueError('Cannot simplify without exceeding the 3 mm bounds or source open-edge count')
    _, points, triangles, error, open_edges, aggression = min(choices, key=lambda item: item[0])
    return points, triangles, {'method': 'quadric edge collapse', 'aggression': aggression,
        'source_faces': len(faces), 'faces': len(triangles), 'source_boundary_edges': original_open,
        'boundary_edges': open_edges, 'bounds_max_delta_m': error,
        'source_bounds_m': original_bounds.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', required=True, type=Path)
    parser.add_argument('--kinematics', type=Path, default=ROOT / 'x2_ultra.urdf')
    parser.add_argument('--output', type=Path, default=ROOT / 'desktop/assets/arm_meshes.json')
    args = parser.parse_args()
    source = ET.parse(args.urdf).getroot()
    baseline = ET.parse(args.kinematics).getroot()
    data = {'schema_version': 1, 'units': 'm', 'base_frame': 'torso_link',
            'purpose': 'display_only_not_collision_or_metrology',
            'source_urdf': args.urdf.name, 'source_urdf_sha256': digest(args.urdf),
            'kinematics_urdf': args.kinematics.name,
            'kinematics_urdf_sha256': digest(args.kinematics), 'arms': {}}
    for side in ('left', 'right'):
        joints = chain(source, side)
        reference = chain(baseline, side)
        if joints != reference:
            raise ValueError('Visual arm chain differs from the IK model: ' + side)
        for joint in joints:
            visual = source.find(f"link[@name='{joint['link']}']/visual")
            mesh = visual.find('geometry/mesh')
            scale = vector(mesh, 'scale', '1 1 1')
            if scale != [1., 1., 1.]:
                raise ValueError('Unexpected mesh scale; verify the source units')
            mesh_path = (args.urdf.parent / mesh.get('filename')).resolve()
            vertices, faces = read_stl(mesh_path)
            # These X2 arm links are 35–175 mm; fail on a common STL mm/m mixup.
            largest_extent = max(vertices.max(axis=0) - vertices.min(axis=0))
            if not .02 < largest_extent < .30:
                raise ValueError('Unexpected arm link dimensions: ' + mesh_path.name)
            vertices, faces, report = simplify(vertices, faces)
            joint.update({'visual_xyz': vector(visual.find('origin'), 'xyz'),
                          'visual_rpy': vector(visual.find('origin'), 'rpy'),
                          'source_mesh': mesh_path.name, 'source_sha256': digest(mesh_path),
                          'vertices': [[round(float(x), 8) for x in p] for p in vertices],
                          'faces': faces.tolist(), 'simplification': report})
            print(side, joint['link'], report['source_faces'], '->', report['faces'])
        data['arms'][side] = joints
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False) + '\n')
    print('Wrote', args.output, digest(args.output))


if __name__ == '__main__':
    main()
