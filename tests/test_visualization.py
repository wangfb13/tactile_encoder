"""Tiny CPU validation of reconstruction rendering; no final-test data or cloud APIs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import build_model
from test_vis import denormalize_signed, output_directory, render_visualizations


NORMALIZATION = {"offset": [0, 30000, 30000], "mean": [10., -4., 6.], "std": [2., 5., 3.],
                 "force_mean": [0.] * 6, "force_std": [1.] * 6, "source_split": "train"}


class TinyViews:
    normalization = NORMALIZATION
    def __len__(self):
        return 2
    def __getitem__(self, index):
        grid = torch.linspace(-1., 1., 48 * 64).reshape(48, 64)
        image = torch.stack((grid, -grid, grid.flip(0)))
        return {"image": image + index * .1, "pair_index": 0, "finger": index}


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = build_model("tiny", image_size=(48, 64))
        model.checkpoint_metadata = {"normalization": copy.deepcopy(NORMALIZATION)}
        return model

    def test_inverse_preserves_signed_deformation_without_storage_bias(self):
        normalized = np.zeros((1, 3, 2, 2), dtype=np.float32)
        restored = denormalize_signed(normalized, NORMALIZATION)
        np.testing.assert_allclose(restored[0, 0], 10.)
        np.testing.assert_allclose(restored[0, 1], -4.)
        np.testing.assert_allclose(restored[0, 2], 6.)

    def test_png_render_excludes_force_and_preserves_mask_rng_and_mode(self):
        model = self.model().train()
        def forbidden(*args):
            raise AssertionError("Force branch must not run during MAE visualization")
        hooks = [model.force_projector.register_forward_hook(forbidden),
                 model.force_head.register_forward_hook(forbidden)]
        before = torch.get_rng_state().clone()
        ratio = model.mask_ratio
        with tempfile.TemporaryDirectory(prefix="visual-unit-", dir=ROOT / "tests") as tmp:
            result = render_visualizations(model, TinyViews(), Path(tmp) / "out", samples=1)
            self.assertFalse(result["force_head_used"])
            self.assertFalse(result["training_performed"])
            self.assertEqual(result["split"], "val")
            path = Path(tmp) / "out" / result["images"][0]["file"]
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 1000)
                self.assertGreater(np.asarray(image).std(), 0)
            self.assertEqual(json.loads((Path(tmp) / "out/visualization.json").read_text())["samples"], 1)
        for hook in hooks:
            hook.remove()
        self.assertEqual(model.mask_ratio, ratio)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_mismatched_training_normalization_is_rejected(self):
        model = self.model()
        model.checkpoint_metadata["normalization"]["std"][1] = 9.
        with tempfile.TemporaryDirectory(prefix="visual-unit-", dir=ROOT / "tests") as tmp:
            with self.assertRaisesRegex(ValueError, "normalization mismatch"):
                render_visualizations(model, TinyViews(), Path(tmp) / "out", samples=1)

    def test_sample_cap_and_output_boundary(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            render_visualizations(self.model(), TinyViews(), ROOT / "tests", samples=9)
        with self.assertRaisesRegex(ValueError, "subdirectory"):
            output_directory("/tmp/outside_tactile_visualization")
        with self.assertRaisesRegex(ValueError, "subdirectory"):
            output_directory(ROOT)

    def test_actual_cli_defaults_to_val_with_tiny_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="visual-cli-", dir=ROOT / "tests") as tmp:
            folder = Path(tmp)
            data = folder / "data";data.mkdir()
            (data / "normalization.json").write_text(json.dumps(NORMALIZATION))
            dd = np.zeros((1, 2, 96, 128, 3), dtype=np.uint16)
            dd[..., 0] = 10
            dd[..., 1] = 29996
            dd[..., 2] = 30006
            np.save(data / "val_dd.npy", dd)
            np.save(data / "val_force.npy", np.zeros((1, 2, 6), dtype=np.float32))
            np.save(data / "val_force_valid.npy", np.ones((1, 2, 6), dtype=bool))
            np.save(data / "val_geometry.npy", np.zeros((1, 2, 7), dtype=np.float32))
            model = self.model()
            checkpoint = folder / "tiny.pt"
            torch.save({"model_config": model.config, "model": model.state_dict(),
                        "normalization": NORMALIZATION}, checkpoint)
            environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, str(ROOT / "test_vis.py"), "--checkpoint", str(checkpoint),
                                     "--data-root", str(data), "--out", str(folder / "out"), "--samples", "1"],
                                    capture_output=True, text=True, env=environment, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((folder / "out/visualization.json").read_text())
            self.assertEqual(report["split"], "val")
            self.assertEqual(report["samples"], 1)
            self.assertFalse(report["force_head_used"])
            self.assertTrue((folder / "out" / report["images"][0]["file"]).is_file())

    def test_launcher_is_a_controller_wrapper_without_local_training(self):
        source = (ROOT / "train.sh").read_text()
        self.assertNotIn("torchrun", source)
        self.assertNotIn("--nproc_per_node", source)
        self.assertIn('cloud/launch.py', source)
        self.assertIn('cloud/control.py', source)
        self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / "train.sh")], capture_output=True).returncode, 0)
        help_result = subprocess.run(["bash", str(ROOT / "train.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("two concurrent", help_result.stdout)
        bad = subprocess.run(["bash", str(ROOT / "train.sh"), "--unknown"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
