"""Analyze Adam-LMS step size effect on reconstruction quality."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adam_lms import run_experiment
from config import ADAM_M, ADAM_N


def analyze_adam_hyperparameters(
    alphas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    patients=range(1, 9),
    save_dir="results/adam_lms/hyperparameters",
):
    """Evaluate average Q1 and Q2 for different Adam-LMS step sizes.

    Args:
        alphas: Iterable of step sizes to evaluate.
        patients: Iterable of patient indices.
        save_dir: Directory where results are saved.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    rows = []

    for alpha in alphas:
        q1_vals, q2_vals = [], []

        for p in patients:
            res = run_experiment(p, N=ADAM_N, M=ADAM_M, alpha=alpha)
            if res is None:
                continue
            Q1, Q2, _, _ = res
            q1_vals.append(Q1)
            q2_vals.append(Q2)

        rows.append(
            {
                "alpha": float(alpha),
                "Avg_Q1": float(np.mean(q1_vals)),
                "Avg_Q2": float(np.mean(q2_vals)),
            }
        )

    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 6))
    plt.plot(df["alpha"], df["Avg_Q1"], "o-", label="Avg Q1")
    plt.plot(df["alpha"], df["Avg_Q2"], "s-", label="Avg Q2")
    plt.axhline(0.9, linestyle="--", color="red", alpha=0.5, label="Target Q=0.9")
    plt.xscale("log")
    
    plt.title("Effect of Step Size (α) on Adam-LMS Performance")
    plt.xlabel("Step Size (α)")
    plt.ylabel("Average Quality Score")
    
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "adam_hyperparameters.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    analyze_adam_hyperparameters()