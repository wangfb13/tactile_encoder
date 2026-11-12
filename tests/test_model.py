"""CPU tests for rectangular geometry, auxiliary gradients, and resume reproducibility."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import Dataset, DataLoader

from model import build_model, load_model
from engine import train_one_epoch, rng_state, restore_rng, seed_everything, atomic_torch_save


class SyntheticDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(987)
        self.images = torch.randn(8, 3, 32, 48, generator=generator)
        self.force = torch.randn(8, 6, generator=generator)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"image": self.images[index], "force": self.force[index],
                "force_valid": torch.ones(6, dtype=torch.bool)}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self):
        return build_model("tiny", image_size=(32, 48))

    def test_rectangular_patch_round_trip(self):
        model = self.make_model()
        x = torch.randn(2, 3, 32, 48)
        patches = model.patchify(x)
        self.assertEqual(patches.shape, (2, 6, 768))
        self.assertTrue(torch.equal(x, model.unpatchify(patches)))

    def test_unmasked_encode_does_not_mutate_mask_or_rng(self):
        model = self.make_model().eval()
        x = torch.randn(2, 3, 32, 48)
        before = torch.get_rng_state()
        z = model.encode(x)
        self.assertEqual(z.shape, (2, 48))
        self.assertEqual(model.mask_ratio, .75)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(int(model(x)["mask"].sum()), 10)

    def test_force_gradient_reaches_encoder_and_invalid_labels_are_safe(self):
        model = self.make_model()
        x = torch.randn(2, 3, 32, 48)
        force = torch.randn(2, 6)
        force[0, 0] = float("nan")
        valid = torch.ones(2, 6, dtype=torch.bool)
        valid[0, 0] = False
        result = model(x, force, valid, .3)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(result["force_valid_count"], 11)
        result["force_loss"].backward()
        self.assertGreater(float(model.patch_embed.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.force_projector[0].weight.grad.abs().sum()), 0.)

    def test_all_missing_force_is_not_zero_label_supervision(self):
        model = self.make_model()
        result = model(torch.randn(2, 3, 32, 48), torch.full((2, 6), float("nan")),
                       torch.zeros(2, 6, dtype=torch.bool), .5)
        self.assertEqual(float(result["force_loss"].detach()), 0.)
        self.assertEqual(result["force_valid_count"], 0)
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_model_checkpoint_reload(self):
        model = self.make_model().eval()
        image = torch.randn(2, 3, 32, 48)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tests") as tmp:
            path = Path(tmp) / "checkpoint.pt"
            atomic_torch_save(path, {"model_config": model.config, "model": model.state_dict()})
            restored = load_model(path)
            self.assertTrue(torch.equal(model.encode(image), restored.encode(image)))

    def test_optimizer_boundary_resume_is_exact(self):
        args = SimpleNamespace(accum_iter=2, lr=1e-4, warmup_epochs=1., epochs=2, min_lr=0.,
                               lambda_force=.3, gradient_check_freq=0, clip_grad=1.,
                               max_steps=0, log_freq=100)
        dataset = SyntheticDataset()
        def loader():
            return DataLoader(dataset, batch_size=2, shuffle=True,
                              generator=torch.Generator().manual_seed(123))
        seed_everything(17)
        original = self.make_model()
        initial = copy.deepcopy(original.state_dict())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tests") as tmp:
            optimizer = torch.optim.AdamW(original.parameters(), lr=args.lr)
            # Reset RNG after creating the identical initial states.
            seed_everything(42)
            full = train_one_epoch(original, loader(), optimizer, torch.device("cpu"), 0, args, tmp)
            full_weights = copy.deepcopy(original.state_dict())
            interrupted = self.make_model()
            interrupted.load_state_dict(initial)
            interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=args.lr)
            seed_everything(42)
            args.max_steps = 1
            partial = train_one_epoch(interrupted, loader(), interrupted_optimizer, torch.device("cpu"), 0, args, tmp)
            record = {"model": copy.deepcopy(interrupted.state_dict()),
                      "optimizer": copy.deepcopy(interrupted_optimizer.state_dict()), "rng": rng_state()}
            restored = self.make_model()
            restored.load_state_dict(record["model"])
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=args.lr)
            restored_optimizer.load_state_dict(record["optimizer"])
            restore_rng(record["rng"])
            args.max_steps = 0
            resumed = train_one_epoch(restored, loader(), restored_optimizer, torch.device("cpu"), 0,
                                      args, tmp, start_batch=partial["next_batch"], global_step=partial["global_step"])
            self.assertEqual(full["global_step"], resumed["global_step"])
            for key, tensor in full_weights.items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
