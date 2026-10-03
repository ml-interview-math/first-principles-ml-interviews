#!/usr/bin/env python3
"""Attention-scaling experiment for ML Interview Question #001."""

from __future__ import annotations

import argparse
import math
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    h = -(p * np.log(np.clip(p, 1e-30, None))).sum(axis=-1)
    return h / math.log(p.shape[-1])


def run_experiment(dks, num_keys, trials, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for dk in dks:
        q = rng.standard_normal((trials, dk))
        k = rng.standard_normal((trials, num_keys, dk))
        logits = np.einsum("td,tkd->tk", q, k)
        rows.append((
            dk,
            float(normalized_entropy(softmax(logits)).mean()),
            float(normalized_entropy(softmax(logits / math.sqrt(dk))).mean()),
            float(normalized_entropy(softmax(logits / dk)).mean()),
        ))
    return rows


def print_table(rows):
    print(f"{'d_k':>6}  {'no scaling':>12}  {'/ sqrt(d_k)':>14}  {'/ d_k':>12}")
    print("-" * 52)
    for dk, none, sqrt_scaled, d_scaled in rows:
        print(f"{dk:6d}  {none:12.4f}  {sqrt_scaled:14.4f}  {d_scaled:12.4f}")


def maybe_plot(rows, output):
    if output is None:
        return
    import matplotlib.pyplot as plt
    dks = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dks, [r[1] for r in rows], marker="o", label="No scaling")
    ax.plot(dks, [r[2] for r in rows], marker="o", label="Divide by sqrt(d_k)")
    ax.plot(dks, [r[3] for r in rows], marker="o", label="Divide by d_k")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Head dimension d_k")
    ax.set_ylabel("Normalized attention entropy")
    ax.set_title("Effect of attention-logit scaling")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"\nSaved plot to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--num-keys", type=int, default=128)
    parser.add_argument("--trials", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    rows = run_experiment(args.dks, args.num_keys, args.trials, args.seed)
    print_table(rows)
    maybe_plot(rows, args.plot)


if __name__ == "__main__":
    main()
