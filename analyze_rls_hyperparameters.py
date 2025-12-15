"""Analyze RLS filter order effect on reconstruction quality."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RLS_LAMBDA, RLS_M
from rls import run_experiment


def analyze_rls_hyperparameters(
    orders=(2, 5, 10, 15, 20, 30, 50),
    patients=range(1, 9),
    fixed_M=RLS_M,
    lam=RLS_LAMBDA,
    save_dir="results/rls/hyperparameters",
):
    """Evaluate average Q1 and Q2 for different RLS filter orders.

    Two sweeps are performed:
      1) N = M = order
      2) M fixed (fixed_M), sweep N = order

    Args:
        orders: Iterable of filter orders to evaluate.
        patients: Iterable of patient indices.
        fixed_M: Fixed M used in the second sweep.
        lam: RLS forgetting factor (lambda).
        save_dir: Directory where results are saved.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    rows = []

    for order in orders:
        q1_vals, q2_vals = [], []
        for p in patients:
            Q1, Q2, _, _ = run_experiment(p, N=order, M=order, lam=lam)
            q1_vals.append(Q1)
            q2_vals.append(Q2)

        rows.append(
            {
                "sweep": "N_equals_M",
                "N": int(order),
                "M": int(order),
                "Avg_Q1": float(np.mean(q1_vals)),
                "Avg_Q2": float(np.mean(q2_vals)),
            }
        )

    for order in orders:
        q1_vals, q2_vals = [], []
        for p in patients:
            Q1, Q2, _, _ = run_experiment(p, N=order, M=fixed_M, lam=lam)
            q1_vals.append(Q1)
            q2_vals.append(Q2)

        rows.append(
            {
                "sweep": "M_fixed_sweep_N",
                "N": int(order),
                "M": int(fixed_M),
                "Avg_Q1": float(np.mean(q1_vals)),
                "Avg_Q2": float(np.mean(q2_vals)),
            }
        )

    df = pd.DataFrame(rows)

    df_eq = df[df["sweep"] == "N_equals_M"].sort_values("N")
    df_fix = df[df["sweep"] == "M_fixed_sweep_N"].sort_values("N")

    plt.figure(figsize=(9, 6))
    plt.plot(df_eq["N"], df_eq["Avg_Q1"], "o-", label="Avg Q1 (N = M)")
    plt.plot(df_eq["N"], df_eq["Avg_Q2"], "s-", label="Avg Q2 (N = M)")

    plt.plot(df_fix["N"], df_fix["Avg_Q1"], "o--", label=f"Avg Q1 (M = {fixed_M})")
    plt.plot(df_fix["N"], df_fix["Avg_Q2"], "s--", label=f"Avg Q2 (M = {fixed_M})")

    plt.axhline(0.9, linestyle="--", color="red", alpha=0.5, label="Target Q=0.9")

    plt.title("Effect of Filter Order (N, M) on RLS Performance")
    plt.xlabel("Filter Order (N)")
    plt.ylabel("Average Quality Score")

    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rls_hyperparameters.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    analyze_rls_hyperparameters()
