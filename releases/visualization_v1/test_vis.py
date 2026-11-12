"""Render MAE reconstruction in sensor units; no force head or cloud API is used.

Default split is val, default device CPU. Nothing is trained or updated.
The signed deformation plots remove the storage offset after reversing train
normalization; these are sensor encoding units, not independently calibrated mm.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

TASK_ROOT = Path("/mnt/workspace/umi/tactile_encoder")
SPLITS = ("train", "val", "test", "ood_funnel", "ood_cups")


def output_directory(path):
    path = Path(path).resolve()
    root = TASK_ROOT.resolve()
    if path == root or root not in path.parents:
        raise ValueError("Visualization output must be a subdirectory of the tactile encoder project")
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalization_arrays(normalization):
    arrays = [np.asarray(normalization[key], dtype=np.float32) for key in ("offset", "mean", "std")]
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Expected finite three-channel training normalization")
    if not np.all(arrays[2] > 0):
        raise ValueError("Normalization standard deviations must be positive")
    return arrays


def denormalize_signed(images, normalization):
    """Undo normalization and remove storage offsets for signed physical channels."""
    offset, mean, std = normalization_arrays(normalization)
    values = np.asarray(images, dtype=np.float32)
    # encoded = values * std + mean + offset. Plotting encoded - offset is
    # algebraically values * std + mean; avoiding add/subtract of 30000 also
    # avoids quantizing small floating-point reconstruction errors.
    shape = (1,) * (values.ndim - 3) + (3, 1, 1)
    return values * std.reshape(shape) + mean.reshape(shape)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@torch.no_grad()
def render_visualizations(model, dataset, out, split="val", samples=4, seed=42, checkpoint=None):
    if split not in SPLITS:
        raise ValueError("Unsupported split")
    if not 1 <= samples <= 8:
        raise ValueError("samples must be between 1 and 8")
    if not len(dataset):
        raise ValueError("Empty visualization split")
    out = output_directory(out)
    normalization = getattr(model, "checkpoint_metadata", {}).get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Checkpoint lacks training normalization; refusing guessed inverse scaling")
    checkpoint_arrays = normalization_arrays(normalization)
    dataset_arrays = normalization_arrays(dataset.normalization)
    for name, trained, current in zip(("offset", "mean", "std"), checkpoint_arrays, dataset_arrays):
        if not np.allclose(trained, current, rtol=1e-6, atol=1e-6):
            raise ValueError("Dataset/checkpoint training normalization mismatch: " + name)
    indices = np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int).tolist()
    examples = [dataset[index] for index in indices]
    device = next(model.parameters()).device
    images = torch.stack([example["image"] for example in examples]).to(device)
    previous_mode = model.training
    previous_ratio = model.mask_ratio
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    model.eval()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            # Explicit lambda=0: force/projector branches are excluded.
            result = model(images, lambda_force=0., mask_ratio=previous_ratio)
        prediction = model.unpatchify(result["pred"]).float()
        mask = F.interpolate(result["mask"].reshape(-1, 1, *model.grid_size).float(),
                             size=model.image_size, mode="nearest")
        reconstruction = images.float() * (1 - mask) + prediction * mask
        original = denormalize_signed(images.float().cpu().numpy(), normalization)
        restored = denormalize_signed(reconstruction.cpu().numpy(), normalization)
        mask_np = mask.cpu().numpy().astype(bool)
    finally:
        model.train(previous_mode)
        if model.mask_ratio != previous_ratio:
            raise RuntimeError("Visualization unexpectedly changed the model mask ratio")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    labels = ("depth (sensor units)", "deformation U (signed sensor units)", "deformation V (signed sensor units)")
    titles = ("Original", "Visible input (gray = masked)", "Reconstruction (visible copied)", "Absolute error (masked region)")
    for k, (index, example) in enumerate(zip(indices, examples)):
        fig, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
        means = []
        for channel, label in enumerate(labels):
            reference = original[k, channel]
            recovered = restored[k, channel]
            missing = mask_np[k, 0]
            visible = reference.copy()
            visible[missing] = np.nan
            error = np.abs(recovered - reference)
            means.append(float(error[missing].mean()) if missing.any() else 0.)
            if channel == 0:
                lower = min(0., float(reference.min()), float(recovered.min()))
                upper = max(lower + 1e-6, float(reference.max()), float(recovered.max()))
                cmap = plt.get_cmap("viridis").copy()
            else:
                limit = max(1e-6, float(np.abs(reference).max()), float(np.abs(recovered).max()))
                lower, upper = -limit, limit
                cmap = plt.get_cmap("RdBu_r").copy()
            cmap = cmap.with_extremes(bad="#d9d9d9")
            mappable = None
            for column, image in enumerate((reference, visible, recovered)):
                mappable = axes[channel, column].imshow(image, cmap=cmap, vmin=lower, vmax=upper,
                                                        origin="upper", interpolation="nearest")
            error_map = axes[channel, 3].imshow(error, cmap="magma", vmin=0.,
                                                vmax=max(1e-6, float(error.max())), origin="upper")
            fig.colorbar(mappable, ax=list(axes[channel, :3]), shrink=.8, label=label)
            fig.colorbar(error_map, ax=axes[channel, 3], shrink=.8, label="absolute sensor-unit error")
            axes[channel, 0].set_ylabel(label + "\nv down; u right")
            for column in range(4):
                if channel == 0:
                    axes[channel, column].set_title(titles[column], fontsize=10)
                axes[channel, column].set_xticks([])
                axes[channel, column].set_yticks([])
        pair = int(example.get("pair_index", index // 2))
        finger = int(example.get("finger", index % 2))
        fig.suptitle(f"{split}: sample {index}, pair {pair}, finger {finger}; mask {previous_ratio:.0%}; SDK units, no force head")
        filename = f"{split}_{k:02d}_index_{index:07d}.png"
        fig.savefig(out / filename, dpi=120)
        plt.close(fig)
        rows.append({"index": index, "pair_index": pair, "finger": finger, "file": filename,
                     "masked_pixel_mae_by_channel": means})
    report = {"split": split, "samples": len(rows), "seed": seed,
              "mask_ratio_before": previous_ratio, "mask_ratio_after": model.mask_ratio,
              "force_head_used": False, "training_performed": False,
              "normalization_source": "checkpoint training statistics",
              "deformation_storage_offsets_removed": list(map(float, checkpoint_arrays[0][1:])),
              "reconstruction": "visible original pixels plus predictions only in masked patches",
              "images": rows}
    if checkpoint is not None:
        report["checkpoint"] = str(Path(checkpoint).resolve())
        report["checkpoint_sha256"] = digest(checkpoint)
    path = out / "visualization.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 1 <= args.samples <= 8:
        parser.error("--samples must be between 1 and 8")
    output_directory(args.out)
    from model import load_model
    from dataset import TactileDataset
    model = load_model(args.checkpoint, device=args.device)
    data = TactileDataset(args.data_root, split=args.split, image_size=model.image_size)
    report = render_visualizations(model, data, args.out, split=args.split,
                                   samples=args.samples, seed=args.seed, checkpoint=args.checkpoint)
    print(json.dumps({"visualization_complete": True, "split": report["split"],
                      "samples": report["samples"], "out": str(Path(args.out).resolve())}), flush=True)


if __name__ == "__main__":
    main()
