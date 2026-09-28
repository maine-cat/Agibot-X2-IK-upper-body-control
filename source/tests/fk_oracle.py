"""Independent URDF chain FK for package regression tests."""
import math
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    """Unit quaternion rotation, independently implemented for the oracle."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    w = math.cos(angle / 2)
    x, y, z = axis * math.sin(angle / 2)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


class IndependentFK:
    def __init__(self, xml, side):
        doc = ET.fromstring(xml)
        parents = {j.find("child").get("link"): j for j in doc.findall("joint")}
        link = f"{side}_wrist_roll_link"
        chain = []
        while link != "torso_link":
            joint = parents[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain = list(reversed(chain))
        active = [j for j in self.chain if j.get("type") != "fixed"]
        assert len(active) == 7
        assert all(j.get("type") == "revolute" for j in active)
        self.names = [j.get("name") for j in active]
        self.low = np.array([float(j.find("limit").get("lower")) for j in active])
        self.high = np.array([float(j.find("limit").get("upper")) for j in active])

    def pose(self, q):
        values = dict(zip(self.names, q))
        result = np.eye(4)
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            fixed = np.eye(4)
            fixed[:3, 3] = xyz
            fixed[:3, :3] = (rotation([0, 0, 1], rpy[2])
                            @ rotation([0, 1, 0], rpy[1])
                            @ rotation([1, 0, 0], rpy[0]))
            motion = np.eye(4)
            if joint.get("type") != "fixed":
                axis = joint.find("axis")
                vector = [1, 0, 0] if axis is None else np.fromstring(axis.get("xyz"), sep=" ")
                motion[:3, :3] = rotation(vector, float(values[joint.get("name")]))
            result = result @ fixed @ motion
        return result[:3, 3], result[:3, :3]


def errors(actual, expected):
    position = float(np.linalg.norm(actual[0] - expected[0]) * 1000)
    delta = actual[1] @ expected[1].T
    skew = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    angle = math.atan2(float(np.linalg.norm(skew)) / 2, float(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
    return position, math.degrees(angle)
