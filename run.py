#!/usr/bin/env python3
"""CLI entry point.

    python run.py --env submerged_water --steps 50000 --gpu 0 --render off
    python run.py --list-envs
    python run.py --compare-envs dry_land submerged_water windy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from brain.subset import DEFAULT_SUBSET, SUBSETS
from env.loader import list_presets, load_preset

import os

RUNS_DIR = Path(os.environ.get("FLY_RUNS_DIR", "/data/runs"))
CHECKPOINT_DIR = Path(os.environ.get("FLY_CHECKPOINT_DIR", "/data/checkpoints"))


def _device(gpu: str | None) -> str:
    if gpu in (None, "", "cpu", "-1"):
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        print(f"warning: --gpu {gpu} requested but no CUDA device is visible; using CPU",
              file=sys.stderr)
        return "cpu"
    return f"cuda:{gpu}"


def cmd_list_envs() -> int:
    for name in list_presets():
        p = load_preset(name)
        desc = " ".join(p.description.split())
        print(f"{name:18} {p.temperature_c:>5.1f}C  rho={float(p.physics['density']):.3e}  {desc[:70]}")
    return 0


def _new_session(args, preset: str, render: bool = False):
    from session import Session

    return Session(
        preset=preset,
        connectome=args.connectome,
        subset=args.neuron_subset,
        device=_device(args.gpu),
        seed=args.seed,
        render=render,
        connectome_dir=args.connectome_dir,
        checkpoint=args.checkpoint,
        synthetic_size=args.synthetic_size,
    )


def cmd_run(args) -> int:
    sess = _new_session(args, args.env, render=(args.render == "on"))
    print(f"connectome: {sess.tables.summary()}")
    print(f"groups: {sess.decoder.group_sizes()}")
    print(f"env: {args.env}  device: {sess.net.device}  steps: {args.steps}")

    t0 = time.time()
    summary = sess.run(args.steps, progress_every=max(args.steps // 5, 1))
    summary["wall_seconds"] = round(time.time() - t0, 2)
    summary["steps_per_second"] = round(args.steps / max(time.time() - t0, 1e-9))
    summary["env"] = args.env
    summary["connectome"] = f"{sess.tables.source}:{sess.tables.dataset}"

    print(json.dumps(summary, indent=2))
    _write_run(args, {args.env: summary})

    if args.save_checkpoint:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINT_DIR / args.save_checkpoint
        sess.net.save(path)
        print(f"checkpoint -> {path}")
    return 0


def cmd_compare(args) -> int:
    """Run the SAME brain snapshot across presets and diff the kinematics."""
    presets = args.compare_envs
    unknown = [p for p in presets if p not in list_presets()]
    if unknown:
        print(f"unknown preset(s): {unknown}. Available: {list_presets()}", file=sys.stderr)
        return 2

    # One brain, saved once, reloaded for every preset - otherwise differences
    # between presets are confounded by differences between networks.
    first = _new_session(args, presets[0])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = CHECKPOINT_DIR / f"compare-{args.seed}.pt"
    try:
        first.net.save(snapshot)
    except OSError:
        snapshot = Path(args.connectome_dir) / f"compare-{args.seed}.pt"
        first.net.save(snapshot)
    del first

    results: dict[str, dict] = {}
    for name in presets:
        print(f"\n=== {name} ===", flush=True)
        sess = _new_session(args, name)
        sess.net.load_weights(snapshot)
        results[name] = sess.run(args.steps)
        results[name]["env"] = name
        print(json.dumps(results[name], indent=2))
        sess.body.close()

    _print_diff(results)
    _write_run(args, results)
    return 0


_COMPARE_KEYS = [
    ("mean_speed_mm_s", "mean speed mm/s"),
    ("peak_speed_mm_s", "peak speed mm/s"),
    ("net_displacement_mm", "displacement mm"),
    ("path_straightness", "straightness"),
    ("gait_regularity", "gait regularity"),
    ("joint_excursion", "joint excursion"),
]


def _print_diff(results: dict[str, dict]) -> None:
    names = list(results)
    width = max(len(n) for n in names) + 2
    print("\n" + "=" * 72)
    print("CROSS-ENVIRONMENT COMPARISON (locomotion kinematics)")
    print("=" * 72)
    header = "metric".ljust(22) + "".join(n.ljust(width) for n in names)
    print(header)
    print("-" * len(header))
    for key, label in _COMPARE_KEYS:
        row = label.ljust(22)
        for n in names:
            row += f"{results[n][key]:<{width}.4g}"
        print(row)
    row = "fell over".ljust(22)
    for n in names:
        row += str(results[n]["fell_over"]).ljust(width)
    print(row)
    print("=" * 72)
    print("Sanity check on the physics plumbing, NOT biological validation.")


def _write_run(args, results: dict) -> None:
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / f"run-{int(time.time())}.json"
        out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2, default=str))
        print(f"\nresults -> {out}")
    except OSError as exc:
        print(f"(could not write run log: {exc})", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py", description="Connectome-driven fly simulation across environment presets."
    )
    p.add_argument("--env", default="dry_land", help="environment preset name")
    p.add_argument("--steps", type=int, default=5000, help="physics steps (timestep 1e-4 s)")
    p.add_argument("--gpu", default=None, help="CUDA device id, or omit for CPU")
    p.add_argument("--render", choices=("on", "off"), default="off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list-envs", action="store_true", help="print available presets and exit")
    p.add_argument("--compare-envs", nargs="+", metavar="ENV",
                   help="run one brain snapshot across several presets and diff the outcomes")
    p.add_argument("--connectome", default="neuprint", choices=("neuprint", "feather", "synthetic"))
    p.add_argument("--connectome-dir", default="/data/connectome")
    p.add_argument("--neuron-subset", default=DEFAULT_SUBSET, choices=sorted(SUBSETS))
    p.add_argument("--synthetic-size", type=int, default=20_000)
    p.add_argument("--checkpoint", default=None, help="load synaptic weights from this file")
    p.add_argument("--save-checkpoint", default=None, help="filename under /data/checkpoints")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_envs:
        return cmd_list_envs()
    if args.compare_envs:
        return cmd_compare(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
