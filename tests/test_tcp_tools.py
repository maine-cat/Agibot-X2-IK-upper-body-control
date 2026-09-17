import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from x2_tcp import load_tcp_tools, read_tcp_json, tcp_to_wrist, validate_tcp_config, wrist_to_tcp
from tools.calibrate_tcp import calibrate_config, fit_pivot


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def custom_config():
    tools = load_tcp_tools("gripper")
    return {"schema_version": 1, "units": "m", "tools": {
        side: {k: v for k, v in tool.items() if k != "mode"} for side, tool in tools.items()}}


def pivot_samples():
    offset = np.array([.025, -.012, -.145])
    fixed = np.array([.3, -.2, .45])
    return [{"rotation_matrix": R.tolist(), "position_m": (fixed - R @ offset).tolist()}
            for a, b in [(-.7, -.6), (-.4, .8), (.2, -.3), (.5, .6), (.8, -.8), (1., .1), (-1., .9), (.7, .4)]
            for R in [rz(a) @ rx(b)]]


class TCPToolsTests(unittest.TestCase):
    def test_none_is_exact_wrist_frame(self):
        for side, tool in load_tcp_tools().items():
            np.testing.assert_array_equal(tool["translation_m"], np.zeros(3))
            np.testing.assert_array_equal(tool["rotation_matrix"], np.eye(3))
            self.assertEqual(tool["frame"], side + "_wrist_roll_link")

    def test_gripper_matches_source_fixed_chain(self):
        # Independently compose the three URDF joint origins, including rotation.
        for side, yaw in (("left", -np.pi / 2), ("right", np.pi / 2)):
            tool = load_tcp_tools("gripper")[side]
            first = rz(yaw) @ rx(np.pi)
            base = first @ rz(np.pi / 2)
            expected_R = base @ rz(-np.pi / 2)
            expected_p = np.array([0., 0., -.0399]) + first @ [0., 0., .0061] + base @ [0., 0., .13008]
            np.testing.assert_allclose(tool["rotation_matrix"], expected_R, atol=1e-14)
            np.testing.assert_allclose(tool["translation_m"], expected_p, atol=1e-14)

    def test_hand_is_marked_estimated_cup_grasp_center(self):
        tool = load_tcp_tools("hand")["left"]
        self.assertEqual(tool["name"], "left_estimated_cup_grasp_center")
        np.testing.assert_allclose(tool["translation_m"], [.035, 0, -.15], atol=1e-14)
        np.testing.assert_allclose(tool["rotation_matrix"], rx(3.14), atol=1e-9)
        self.assertTrue(tool["estimated"])
        self.assertIn("not measured calibration", tool["description"])

    def test_full_pose_round_trip_with_rotated_tool(self):
        p, R = np.array([.3, -.1, .4]), rz(.8) @ rx(-.4)
        for mode in ("none", "hand", "gripper"):
            for tool in load_tcp_tools(mode).values():
                tcp_p, tcp_R = wrist_to_tcp(p, R, tool)
                actual_p, actual_R = tcp_to_wrist(tcp_p, tcp_R, tool)
                np.testing.assert_allclose(actual_p, p, atol=1e-14)
                np.testing.assert_allclose(actual_R, R, atol=1e-14)
                np.testing.assert_allclose(tcp_p - p, R @ tool["translation_m"], atol=1e-14)

    def test_custom_mutation_does_not_escape_loader(self):
        config = custom_config()
        loaded = load_tcp_tools("custom", tcp_config=config)
        loaded["left"]["translation_m"][0] = 55
        self.assertEqual(config["tools"]["left"]["translation_m"][0], 0)

    def test_strict_rotation_and_numeric_validation(self):
        for value in ([[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[1, .1, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]],
                      [[True, 0, 0], [0, 1, 0], [0, 0, 1]]):
            with self.subTest(value=value):
                data = custom_config()
                data["tools"]["left"]["rotation_matrix"] = value
                with self.assertRaises(ValueError):
                    validate_tcp_config(data)
        for value in ([0, 0, float("inf")], ["0", 0, 1], [True, 0, 1], [0, 1]):
            data = custom_config()
            data["tools"]["right"]["translation_m"] = value
            with self.assertRaises(ValueError):
                validate_tcp_config(data)

    def test_wrong_frame_units_and_unknown_fields_rejected(self):
        for edit in (
            lambda d: d.update(units="mm"),
            lambda d: d.update(schema_version=True),
            lambda d: d["tools"]["left"].update(frame="torso_link"),
            lambda d: d["tools"]["left"].update(rotation_rpy=[0, 0, 0]),
            lambda d: d["tools"].pop("right"),
        ):
            data = custom_config()
            edit(data)
            with self.assertRaises(ValueError):
                validate_tcp_config(data)

    def test_mode_and_custom_source_are_unambiguous(self):
        for kwargs in ({"mode": "invalid"}, {"mode": "custom"},
                       {"mode": "none", "tcp_config": custom_config()},
                       {"mode": "custom", "tcp_file": "x", "tcp_config": custom_config()}):
            with self.assertRaises(ValueError):
                load_tcp_tools(**kwargs)

    def test_file_and_transport_config_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tcp.json"
            path.write_text(json.dumps(custom_config()))
            self.assertEqual(load_tcp_tools("custom", tcp_file=path),
                             load_tcp_tools("custom", tcp_config=custom_config()))

    def test_duplicate_nonfinite_and_oversized_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tcp.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}', ' ' * 65537):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_tcp_json(path)


class PivotTests(unittest.TestCase):
    def test_recovers_known_offset_and_pivot(self):
        result = fit_pivot(pivot_samples())
        np.testing.assert_allclose(result["translation_m"], [.025, -.012, -.145], atol=1e-13)
        np.testing.assert_allclose(result["pivot_position_m"], [.3, -.2, .45], atol=1e-13)
        self.assertLess(result["rms_mm"], 1e-9)

    def test_single_axis_samples_cannot_identify_offset(self):
        samples = [{"rotation_matrix": rz(a).tolist(), "position_m": [0, 0, 0]} for a in np.linspace(-1, 1, 8)]
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            fit_pivot(samples)

    def test_poor_orientation_diversity_rejected(self):
        samples = [{"rotation_matrix": (rz(a * 1e-5) @ rx(a * a * 1e-5)).tolist(), "position_m": [0, 0, 0]}
                   for a in np.linspace(-1, 1, 8)]
        with self.assertRaisesRegex(ValueError, "ill-conditioned"):
            fit_pivot(samples)

    def test_bad_touch_is_not_silently_accepted(self):
        samples = pivot_samples()
        samples[0]["position_m"][0] += .03
        with self.assertRaisesRegex(ValueError, "residual too high"):
            fit_pivot(samples)

    def test_calibration_requires_independent_rotation_and_other_arm(self):
        data = {"schema_version": 1, "units": "m", "samples": {"left": pivot_samples()},
                "tcp_rotation_matrix": {"left": np.eye(3).tolist()}}
        with self.assertRaisesRegex(ValueError, "base-file"):
            calibrate_config(data)
        result, report = calibrate_config(data, base_config=custom_config())
        self.assertEqual(result["tools"]["right"], custom_config()["tools"]["right"])
        self.assertEqual(set(report), {"left"})
        load_tcp_tools("custom", tcp_config=result)
        data["tcp_rotation_matrix"] = {}
        with self.assertRaisesRegex(ValueError, "cannot measure orientation"):
            calibrate_config(data, base_config=custom_config())

    def test_rejects_invalid_limits_and_insufficient_samples(self):
        for params in ({"max_rms_mm": float("nan")}, {"max_condition": 0}, {"max_residual_mm": True}):
            with self.assertRaises(ValueError):
                fit_pivot(pivot_samples(), **params)
        with self.assertRaisesRegex(ValueError, "at least 6"):
            fit_pivot(pivot_samples()[:5])


if __name__ == "__main__":
    unittest.main()
