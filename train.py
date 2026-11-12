"""Single-GPU experiment entry point; does not submit or manage cloud jobs."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

from config import get_args_parser
from dataset import TactileDataset
from engine import (append_json, atomic_json, atomic_torch_save, rng_state, restore_rng,
                    seed_everything, train_one_epoch, validate)
from model import TactileMAE, build_model


def data_fingerprint(root):
    paths = [Path(root) / "normalization.json", Path(root) / "train_manifest.jsonl"]
    value = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        value.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
    return value.hexdigest()


def loader_for(dataset, args, epoch, shuffle=True):
    generator = torch.Generator()
    generator.manual_seed(args.seed * 1000003 + epoch + (0 if shuffle else 90000000))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"),
                      generator=generator, persistent_workers=False)


def save_checkpoint(path, model, optimizer, args, dataset, fingerprint, epoch,
                    next_epoch, next_batch, global_step, include_optimizer=True):
    record = {"format_version": 1, "model_config": model.config, "model": model.state_dict(),
              "args": vars(args), "normalization": dataset.normalization,
              "data_fingerprint": fingerprint, "epoch": epoch, "next_epoch": next_epoch,
              "next_batch": next_batch, "global_step": global_step,
              "rng": rng_state(), "amp": {"dtype": "bfloat16", "grad_scaler": None}}
    if include_optimizer:
        record["optimizer"] = optimizer.state_dict()
    atomic_torch_save(path, record)


def main():
    args = get_args_parser().parse_args()
    if args.lambda_force < 0 or args.epochs < 1 or args.accum_iter < 1 or args.batch_size < 1 or min(args.save_freq, args.eval_freq, args.log_freq) < 1:
        raise ValueError("Invalid training arguments")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETED.json").exists():
        print("TACTILE_ALREADY_COMPLETED " + str(out), flush=True)
        return
    if (out / "last.pt").exists() and not args.resume:
        raise RuntimeError("Existing checkpoint: explicitly resume rather than overwrite")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is required for the configured GPU run")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed)
    data = TactileDataset(args.data_root, split="train", image_size=(192, 256))
    validation = TactileDataset(args.data_root, split="val", image_size=(192, 256))
    if args.val_max_samples and len(validation) > args.val_max_samples:
        # Uniform indices across the fixed manifest rather than selecting the easiest prefix.
        indices = torch.linspace(0, len(validation) - 1, args.val_max_samples).long().tolist()
        validation = Subset(validation, indices)
    fingerprint = data_fingerprint(args.data_root)
    model = build_model(args.model_size)
    initialization = None
    start_epoch, start_batch, global_step = 0, 0, 0
    record = None
    resume = out / "last.pt" if args.resume == "auto" else Path(args.resume) if args.resume else None
    if resume and resume.exists():
        record = torch.load(resume, map_location="cpu", weights_only=False)
        if record["data_fingerprint"] != fingerprint:
            raise ValueError("Resume dataset fingerprint mismatch")
        for key in ("lambda_force", "seed", "epochs", "batch_size", "accum_iter",
                    "lr", "min_lr", "warmup_epochs", "weight_decay", "model_size"):
            if record["args"].get(key) != vars(args).get(key):
                raise ValueError("Resume setting changed: " + key)
        if record["model_config"] != model.config:
            raise ValueError("Resume model configuration mismatch")
        model.load_state_dict(record["model"], strict=True)
        start_epoch, start_batch, global_step = record["next_epoch"], record["next_batch"], record["global_step"]
    elif resume and args.resume != "auto":
        raise FileNotFoundError(resume)
    elif args.pretrained:
        if args.model_size != "base":
            raise ValueError("Tiny CPU tests must not use base CLIP weights")
        initialization = model.load_clip_pretrained(args.pretrained)
        atomic_json(out / "initialization.json", initialization)
    elif args.model_size == "base":
        raise ValueError("Base experiments require an explicit common CLIP initialization")
    model.to(device)
    decay = [p for name, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for name, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.}],
                                  lr=args.lr, betas=(.9, .95))
    if record is not None:
        optimizer.load_state_dict(record["optimizer"])
        # load_state_dict casts tensor state to each parameter's device.
        restore_rng(record["rng"])
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in ("model.py", "train.py", "engine.py", "config.py", "dataset.py")}
    atomic_json(out / "run_config.json", {"args": vars(args), "model_config": model.config,
                "data_fingerprint": fingerprint, "normalization": data.normalization,
                "source_sha256": source_hashes, "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "training_samples": len(data), "effective_batch_size": args.batch_size * args.accum_iter,
                "resume_start_epoch": start_epoch, "resume_start_batch": start_batch})
    print("TACTILE_TRAIN_START " + json.dumps({"out": str(out), "lambda_force": args.lambda_force,
          "seed": args.seed, "samples": len(data), "start_epoch": start_epoch,
          "start_step": global_step, "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    start_time = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        loader = loader_for(data, args, epoch, shuffle=True)
        stats = train_one_epoch(model, loader, optimizer, device, epoch, args, out,
                                start_batch=start_batch if epoch == start_epoch else 0,
                                global_step=global_step)
        global_step = stats["global_step"]
        completed_epochs = epoch + int(stats["epoch_complete"])
        next_epoch = epoch + 1 if stats["epoch_complete"] else epoch
        next_batch = 0 if stats["epoch_complete"] else stats["next_batch"]
        validation_stats = None
        if stats["epoch_complete"] and ((epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs):
            validation_stats = validate(model, loader_for(validation, args, epoch, shuffle=False), device, args)
        row = {"event": "epoch", "epoch": completed_epochs, "global_step": global_step,
               "train": stats, "validation": validation_stats,
               "gpu_max_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        append_json(out / "metrics.jsonl", row)
        print("TACTILE_EPOCH " + json.dumps(row), flush=True)
        atomic_json(out / "progress.json", {"epoch": completed_epochs, "next_epoch": next_epoch,
                    "next_batch": next_batch, "global_step": global_step, "lambda_force": args.lambda_force,
                    "seed": args.seed, "elapsed_this_launch_seconds": time.perf_counter() - start_time})
        should_save = stats["stopped_at_max_steps"] or (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs
        if should_save:
            save_checkpoint(out / "last.pt", model, optimizer, args, data, fingerprint,
                            completed_epochs, next_epoch, next_batch, global_step)
            if stats["epoch_complete"]:
                save_checkpoint(out / f"checkpoint-{epoch + 1:04d}.pt", model, optimizer, args, data, fingerprint,
                                completed_epochs, next_epoch, next_batch, global_step, include_optimizer=False)
        if stats["stopped_at_max_steps"]:
            atomic_json(out / "SHORT_RUN.json", {"global_step": global_step, "completed_epochs": completed_epochs,
                                               "training_complete": completed_epochs >= args.epochs})
            if completed_epochs < args.epochs:
                atomic_json(out / "summary.json", {"status": "partial", "epochs": completed_epochs,
                    "target_epochs": args.epochs, "global_step": global_step,
                    "lambda_force": args.lambda_force, "seed": args.seed,
                    "data_fingerprint": fingerprint, "feature_dimension": model.encoder_dim,
                    "seconds_this_launch": time.perf_counter() - start_time,
                    "checkpoint": str(out / "last.pt"), "reason": "max_steps",
                    "training_complete": False})
                return
    save_checkpoint(out / "encoder.pt", model, optimizer, args, data, fingerprint,
                    args.epochs, args.epochs, 0, global_step, include_optimizer=False)
    summary = {"status": "completed", "epochs": args.epochs, "global_step": global_step,
               "lambda_force": args.lambda_force, "seed": args.seed, "data_fingerprint": fingerprint,
               "seconds_this_launch": time.perf_counter() - start_time,
               "feature_dimension": model.encoder_dim, "checkpoint": str(out / "encoder.pt")}
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "COMPLETED.json", summary)
    print("TACTILE_TRAIN_COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
