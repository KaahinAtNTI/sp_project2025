"""Analyze RLS filter order effect on reconstruction quality."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RLS_LAMBDA
from rls import run_experiment


def analyze_rls_hyperparameters(
    orders=(2, 5, 10, 15, 20, 30, 50),
    patients=range(1, 9),
    save_dir="results/rls/hyperparameters",
):
    """Evaluate average Q1 and Q2 for different RLS filter orders.

    Args:
        orders: Iterable of filter orders (N=M).
        patients: Iterable of patient indices.
        save_dir: Directory where results are saved.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    rows = []

    for N in orders:
        q1_vals, q2_vals = [], []

        for p in patients:
            res = run_experiment(p, N=N, M=N, lam=RLS_LAMBDA)
            if res is None:
                continue
            Q1, Q2, _, _ = res
            q1_vals.append(Q1)
            q2_vals.append(Q2)

        rows.append(
            {
                "N": N,
                "Avg_Q1": float(np.mean(q1_vals)),
                "Avg_Q2": float(np.mean(q2_vals)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "rls_hyperparameters.csv"), index=False)

    plt.figure(figsize=(9, 6))
    plt.plot(df["N"], df["Avg_Q1"], "o-", label="Avg Q1")
    plt.plot(df["N"], df["Avg_Q2"], "s-", label="Avg Q2")
    plt.axhline(0.9, linestyle="--", color="red", alpha=0.5, label="Target Q=0.9")

    
    plt.title("Effect of Filter Order (N) on RLS Performance")
    plt.xlabel("Filter Order (N)")
    plt.ylabel("Average Quality Score")
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rls_hyperparameters.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    analyze_rls_hyperparameters()