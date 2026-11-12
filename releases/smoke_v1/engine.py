"""Training utilities with BF16, explicit finite checks, and deterministic resume."""
import contextlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def learning_rate(epoch_fraction, args):
    if epoch_fraction < args.warmup_epochs:
        return args.lr * (epoch_fraction + 1e-3) / max(args.warmup_epochs, 1e-6)
    progress = (epoch_fraction - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1e-6)
    return args.min_lr + .5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(progress, 1.)))


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def last_block_gradient_ratio(model, result, weight):
    # Diagnostic of a specified encoder block, not a claim about the whole gradient.
    params = tuple(model.encoder_blocks[-1].parameters())
    first = torch.autograd.grad(result["mae_loss"], params, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(result["force_loss"], params, retain_graph=True, allow_unused=True)
    def norm(items):
        return torch.stack([g.detach().float().square().sum() for g in items if g is not None]).sum().sqrt()
    return float((weight * norm(second) / norm(first).clamp_min(1e-12)).item())


def train_one_epoch(model, loader, optimizer, device, epoch, args, out,
                    start_batch=0, global_step=0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    accum = args.accum_iter
    if start_batch % accum:
        raise ValueError("Resume position must be an optimizer-step boundary")
    sums = {"loss": 0., "mae_loss": 0., "force_loss": 0.}
    seen = 0
    start_time = time.perf_counter()
    gradient_ratio = None
    for index, raw in enumerate(loader):
        if index < start_batch:
            continue
        batch = move_batch(raw, device)
        group_start = index - index % accum
        group_batches = min(accum, total_batches - group_start)
        if index % accum == 0:
            lr = learning_rate(epoch + index / total_batches, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
        with autocast(device):
            result = model(batch["image"], batch["force"], batch["force_valid"], args.lambda_force)
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError("Nonfinite training loss")
        if args.lambda_force > 0 and args.gradient_check_freq and index % accum == 0 and global_step % args.gradient_check_freq == 0:
            gradient_ratio = last_block_gradient_ratio(model, result, args.lambda_force)
        batch_count = batch["image"].shape[0]
        group_samples = min(loader.batch_size * group_batches,
                            len(loader.dataset) - group_start * loader.batch_size)
        (result["loss"] * batch_count / group_samples).backward()
        seen += batch_count
        for key in sums:
            sums[key] += float(result[key].detach().item()) * batch_count
        update = (index + 1) % accum == 0 or index + 1 == total_batches
        if not update:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        elapsed = time.perf_counter() - start_time
        row = {"event": "train_step", "epoch": epoch, "batch": index + 1,
               "global_step": global_step, "loss": float(result["loss"].item()),
               "mae_loss": float(result["mae_loss"].item()),
               "force_loss": float(result["force_loss"].item()),
               "valid_force_components": result["force_valid_count"],
               "lr": optimizer.param_groups[0]["lr"],
               "gradient_norm_before_clip": float(grad_norm.item()),
               "last_encoder_block_weighted_force_to_mae_grad_ratio": gradient_ratio,
               "elapsed_seconds": elapsed, "single_finger_images_per_second": seen / max(elapsed, 1e-8)}
        append_json(Path(out) / "train.jsonl", row)
        if global_step % args.log_freq == 0 or global_step == 1:
            print("TACTILE_TRAIN " + json.dumps(row), flush=True)
        if args.max_steps and global_step >= args.max_steps:
            return {"global_step": global_step, "next_batch": index + 1,
                    "epoch_complete": index + 1 == total_batches, "stopped_at_max_steps": True,
                    "metrics": {key: value / max(seen, 1) for key, value in sums.items()},
                    "seen": seen, "elapsed_seconds": elapsed}
    return {"global_step": global_step, "next_batch": total_batches, "epoch_complete": True,
            "stopped_at_max_steps": False,
            "metrics": {key: value / max(seen, 1) for key, value in sums.items()},
            "seen": seen, "elapsed_seconds": time.perf_counter() - start_time}


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    saved_rng = rng_state()
    # Same evaluation masks at every checkpoint and across loss variants.
    seed_everything(91027)
    counts, totals, features = 0, {"mae_loss": 0., "force_loss": 0.}, []
    force_numerator = 0.
    force_components = 0
    try:
        for raw in loader:
            batch = move_batch(raw, device)
            with autocast(device):
                result = model(batch["image"], batch["force"], batch["force_valid"], args.lambda_force)
                if sum(x.shape[0] for x in features) < 256:
                    features.append(model.encode(batch["image"]).float().cpu())
            n = batch["image"].shape[0]
            counts += n
            totals["mae_loss"] += float(result["mae_loss"].item()) * n
            if args.lambda_force > 0:
                force_numerator += float(result["force_loss"].item()) * result["force_valid_count"]
                force_components += result["force_valid_count"]
        z = torch.cat(features)[:256]
        z = z - z.mean(dim=0)
        singular = torch.linalg.svdvals(z).square()
        probability = singular / singular.sum().clamp_min(1e-20)
        effective_rank = (float(torch.exp(-(probability * probability.clamp_min(1e-20).log()).sum()).item())
                          if float(singular.sum()) > 1e-20 else 0.)
        result = {"samples": counts, "mae_loss": totals["mae_loss"] / max(counts, 1),
                  "force_loss": force_numerator / max(force_components, 1),
                  "force_valid_components": force_components,
                  "feature_mean_std": float(z.std(dim=0, unbiased=False).mean().item()),
                  "feature_effective_rank_cov": effective_rank}
        if not all(math.isfinite(v) for v in result.values()):
            raise FloatingPointError("Nonfinite validation metric")
        return result
    finally:
        restore_rng(saved_rng)
