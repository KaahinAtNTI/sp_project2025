"""Compute Q1/Q2 metrics for all patients using RLS."""

import os
import pandas as pd

from config import RLS_LAMBDA, RLS_M, RLS_N
from rls import run_experiment


def generate_q_table(
    output_dir="results/rls/q_table",
    basename="q_table_rls",
    N=RLS_N,
    M=RLS_M,
    lam=RLS_LAMBDA,
):
    """Run RLS for patients 1–8 and save Q1/Q2 results to CSV.

    Returns:
        Pandas DataFrame with columns [Patient, Q1, Q2].
    """
    results = []

    for patient_no in range(1, 9):
        try:
            Q1, Q2, _, _ = run_experiment(
                patient_no,
                N,
                M,
                lam,
            )
            results.append({"Patient": patient_no, "Q1": Q1, "Q2": Q2})
        except Exception:
            results.append({"Patient": patient_no, "Q1": 0.0, "Q2": 0.0})

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{basename}.csv")
    df.to_csv(csv_path, index=False)

    return df


if __name__ == "__main__":
    generate_q_table()
