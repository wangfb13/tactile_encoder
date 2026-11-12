"""Experiment CLI. Full runs are controlled by the task-owned DLC controller."""
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser("Shared single-finger tactile MAE")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--lambda-force", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum-iter", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=0.)
    parser.add_argument("--warmup-epochs", type=float, default=5.)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--clip-grad", type=float, default=1.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-size", choices=["base", "tiny"], default="base")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--gradient-check-freq", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this global optimizer step; 0 means full run")
    parser.add_argument("--resume", default="", help="Checkpoint path or 'auto' for out/last.pt")
    parser.add_argument("--val-max-samples", type=int, default=2048)
    return parser
