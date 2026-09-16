"""随机 Cartesian 覆盖的独立离线测试；不连接 ROS，不执行机器人运动。"""
from collections import Counter
import contextlib
import copy
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x2_random_points as random_points
from x2_arm_model import ArmModel, rpy_to_matrix
from x2_frames import HOME_Q


class SamplingTests(unittest.TestCase):
    def test_each_cartesian_cell_keeps_exact_count_and_samples_inside_its_bounds(self):
        bounds = ((.08, .30), (.18, .36), (-.10, .10))
        cells = (2, 2, 2)
        for side in ("left", "right"):
            with self.subTest(side=side):
                samples = random_points.generate_samples(side, samples_per_cell=4,
                                                         seed=20260915, bounds=bounds, cells=cells)
                self.assertEqual(len(samples), 32)
                counts = Counter(tuple(row["cell"]) for row in samples)
                self.assertEqual(len(counts), 8)
                self.assertEqual(set(counts.values()), {4})
                self.assertEqual(len({row["id"] for row in samples}), 32)
                for row in samples:
                    position = np.asarray(row["pos"])
                    self.assertGreater(position[1] if side == "left" else -position[1], 0.)
                    unsigned = [position[0], abs(position[1]), position[2]]
                    for axis, cell in enumerate(row["cell"]):
                        edges = np.linspace(*bounds[axis], cells[axis] + 1)
                        self.assertGreaterEqual(unsigned[axis], edges[cell])
                        self.assertLess(unsigned[axis], edges[cell + 1])

    def test_seed_is_reproducible_changes_positions_and_preserves_per_cell_prefix(self):
        first = random_points.generate_samples("right", samples_per_cell=2, seed=17)
        repeat = random_points.generate_samples("right", samples_per_cell=2, seed=17)
        changed = random_points.generate_samples("right", samples_per_cell=2, seed=18)
        expanded = random_points.generate_samples("right", samples_per_cell=4, seed=17)
        self.assertEqual(first, repeat)
        self.assertNotEqual([row["pos"] for row in first], [row["pos"] for row in changed])
        by_id = {row["id"]: row for row in expanded}
        for row in first:
            self.assertEqual(row, by_id[row["id"]])

    def test_left_right_mirror_positions_but_use_their_own_fixed_rotation(self):
        left = random_points.generate_samples("left", samples_per_cell=2, seed=91)
        right = random_points.generate_samples("right", samples_per_cell=2, seed=91)
        left_rot = ArmModel("left").forward_kinematics(HOME_Q)[1]
        right_rot = ArmModel("right").forward_kinematics(HOME_Q)[1]
        self.assertFalse(np.allclose(left_rot, right_rot))
        for lrow, rrow in zip(left, right):
            self.assertEqual(lrow["cell"], rrow["cell"])
            self.assertEqual(lrow["sample_index"], rrow["sample_index"])
            np.testing.assert_array_equal(lrow["pos"], np.asarray(rrow["pos"]) * [1., -1., 1.])
            np.testing.assert_array_equal(lrow["rot"], left_rot)
            np.testing.assert_array_equal(rrow["rot"], right_rot)

    def test_sample_positions_are_not_created_from_fk_reachable_points(self):
        model = ArmModel("right")
        real_fk = model.forward_kinematics
        calls = []
        def sentinel_fk(q):
            calls.append(np.asarray(q).copy())
            _, rotation = real_fk(q)
            return np.array([900., 800., 700.]), rotation
        model.forward_kinematics = sentinel_fk
        with patch.object(random_points, "ArmModel", return_value=model):
            samples = random_points.generate_samples("right", samples_per_cell=2)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], HOME_Q)
        for row in samples:
            self.assertTrue(.08 <= row["pos"][0] <= .30)

    def test_invalid_sampling_arguments_rejected(self):
        invalid = [dict(side="both"), dict(samples_per_cell=0), dict(samples_per_cell=True),
                   dict(samples_per_cell=1.5), dict(seed=-1), dict(seed=True),
                   dict(cells=(2, 0, 2)), dict(cells=(2, 2)), dict(cells=(2, 1.5, 2)),
                   dict(bounds=((0., .3), (.1, .2), (-.1, .1))),
                   dict(bounds=((.1, .3), (0., .2), (-.1, .1))),
                   dict(bounds=((.3, .1), (.1, .2), (-.1, .1))),
                   dict(bounds=((.1, math.inf), (.1, .2), (-.1, .1)))]
        for overrides in invalid:
            options = dict(side="right", samples_per_cell=1)
            options.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                random_points.generate_samples(**options)


class EvaluationTests(unittest.TestCase):
    def controlled_model_and_samples(self):
        model = ArmModel("right")
        original_fk = model.forward_kinematics
        rotation = original_fk(HOME_Q)[1]
        labels = ("accepted", "no_solution", "rotation", "position", "margin",
                  "invalid", "exception")
        samples, solutions, poses, margins = [], {}, {}, {}
        for index, label in enumerate(labels):
            pos = np.array([.1 + index * .01, -.25, 0.])
            q = HOME_Q.copy()
            q[0] += (index + 1) * .02
            samples.append(dict(id=f"case-{label}", side="right", cell=[index % 2, 0, 0],
                                sample_index=index, seed=0, pos=pos.tolist(), rot=rotation.tolist()))
            pose_pos = pos + ([.001, 0., 0.] if label == "position" else np.zeros(3))
            pose_rot = rotation @ rpy_to_matrix([.03, 0., 0.]) if label == "rotation" else rotation
            poses[tuple(q)] = (pose_pos, pose_rot)
            margins[tuple(q)] = .01 if label == "margin" else .1
            if label == "invalid":
                q[2] = math.nan
            solutions[label] = SimpleNamespace(q=q, pos_error=0., rot_error=0.)
        def controlled_fk(q):
            return poses.get(tuple(q), original_fk(q))
        model.forward_kinematics = controlled_fk
        model.limit_margin = lambda q: margins[tuple(q)]
        calls = []
        solver = SimpleNamespace(project_to_workspace=Mock(), solve_hold_rotation=Mock())
        def solve(pos, rot, *, q_seed):
            index = len(calls)
            calls.append(dict(pos=pos.copy(), rot=rot.copy(), seed=q_seed.copy(), seed_object=q_seed))
            np.testing.assert_array_equal(q_seed, HOME_Q)
            q_seed[0] = 99.  # 下一点必须是新的 HOME 副本，不能复用该对象或上一点结果。
            label = labels[index]
            if label == "no_solution":
                return None
            if label == "exception":
                raise RuntimeError("injected IK failure")
            return solutions[label]
        solver.solve = Mock(side_effect=solve)
        return model, samples, solver, calls

    def test_all_gate_failures_remain_in_original_denominator_with_fixed_home_seeds(self):
        model, samples, solver, calls = self.controlled_model_and_samples()
        original = copy.deepcopy(samples)
        progress = Mock()
        with patch.object(random_points, "ArmModel", return_value=model), \
                patch.object(random_points, "SrsArmIK", return_value=solver):
            report = random_points.evaluate_samples(samples, progress=progress)
        self.assertEqual(samples, original)
        self.assertEqual(len(calls), len(samples))
        self.assertEqual(len({id(call["seed_object"]) for call in calls}), len(samples))
        for expected, actual, recorded in zip(samples, calls, report["samples"]):
            np.testing.assert_array_equal(actual["pos"], expected["pos"])
            np.testing.assert_array_equal(actual["rot"], expected["rot"])
            self.assertEqual(recorded["pos"], expected["pos"])
            self.assertEqual(recorded["rot"], expected["rot"])
        solver.project_to_workspace.assert_not_called()
        solver.solve_hold_rotation.assert_not_called()
        summary = report["summary"]["overall"]
        self.assertEqual(summary["total"], 7)
        self.assertEqual(summary["accepted"], 1)
        self.assertAlmostEqual(summary["success_rate"], 1. / 7.)
        self.assertEqual(report["summary"]["by_side"]["right"]["total"], 7)
        self.assertEqual(sum(row["total"] for row in report["summary"]["by_cell"]["right"].values()), 7)
        self.assertEqual([row["reason"] for row in report["samples"]],
                         ["accepted", "no_solution", "pose_tolerance", "pose_tolerance",
                          "margin_gate", "invalid_solution", "solver_error"])
        # IK 自报误差始终为 0；姿态和位置拒绝必须来自独立 FK 的复算。
        self.assertGreater(report["samples"][2]["rot_err_rad"], math.radians(.5))
        self.assertGreater(report["samples"][3]["pos_err_m"], .0005)
        self.assertEqual(progress.call_count, 7)
        self.assertEqual(progress.call_args.args[:2], (7, 7))
        np.testing.assert_array_equal(HOME_Q, [.4, 0., 0., -1.2, 0., 0., 0.])

    def test_arbitrary_rotation_cannot_replace_fixed_home_orientation(self):
        samples = random_points.generate_samples("right", samples_per_cell=1)
        samples[0]["rot"] = np.eye(3).tolist()
        with patch.object(random_points, "SrsArmIK") as solver:
            with self.assertRaises(ValueError):
                random_points.evaluate_samples(samples)
        solver.return_value.solve.assert_not_called()

    def test_invalid_tolerances_empty_or_duplicate_samples_rejected_before_solve(self):
        samples = random_points.generate_samples("right", samples_per_cell=1)
        for kwargs in (dict(pos_tol=0.), dict(pos_tol=math.nan), dict(rot_tol=-1.),
                       dict(rot_tol=math.inf), dict(min_margin=-1.)):
            with self.subTest(kwargs=kwargs), patch.object(random_points, "SrsArmIK") as solver:
                with self.assertRaises(ValueError):
                    random_points.evaluate_samples(samples, **kwargs)
                solver.assert_not_called()
        with self.assertRaises(ValueError):
            random_points.evaluate_samples([])
        with patch.object(random_points, "SrsArmIK") as solver:
            with self.assertRaises(ValueError):
                random_points.evaluate_samples([samples[0], samples[0]])
            solver.return_value.solve.assert_not_called()

    def test_real_models_solve_small_independent_stratified_sample(self):
        samples = [row for side in ("left", "right")
                   for row in random_points.generate_samples(side, samples_per_cell=1, seed=20260915)]
        report = random_points.evaluate_samples(samples)
        self.assertEqual(report["summary"]["overall"]["total"], 16)
        self.assertEqual(len(report["samples"]), 16)
        self.assertTrue(report["offline_only"])
        for side in ("left", "right"):
            summary = report["summary"]["by_side"][side]
            self.assertEqual(summary["total"], 8)
            self.assertGreater(summary["accepted"], 0)
        for row in report["samples"]:
            if row["accepted"]:
                self.assertLessEqual(row["pos_err_m"], .0005)
                self.assertLessEqual(row["rot_err_rad"], math.radians(.5))
                self.assertGreaterEqual(row["min_limit_margin_rad"], math.radians(3.))


class CommandTests(unittest.TestCase):
    def test_invalid_cli_values_rejected_before_generating_or_solving(self):
        for options in (["--samples-per-cell", "0"], ["--seed", "-1"],
                        ["--cells", "1", "0", "2"], ["--x-range", "nan", ".3"],
                        ["--abs-y-range", "0", ".3"]):
            with self.subTest(options=options), \
                    patch.object(random_points, "generate_samples") as generate, \
                    patch.object(random_points, "evaluate_samples") as evaluate, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    random_points.main(options)
                self.assertEqual(caught.exception.code, 2)
                generate.assert_not_called()
                evaluate.assert_not_called()

    def test_existing_report_is_not_overwritten_or_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            original = b'{"preserved": true}\n'
            output.write_bytes(original)
            with patch.object(random_points, "evaluate_samples") as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    random_points.main(["--side", "right", "--samples-per-cell", "1",
                                        "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            evaluate.assert_not_called()
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
