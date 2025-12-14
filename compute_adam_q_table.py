"""Compute Q1/Q2 metrics for all patients using Adam-LMS."""

import os
import pandas as pd

from adam_lms import run_experiment
from config import ADAM_ALPHA, ADAM_M, ADAM_N


def generate_q_table(
    output_dir="results/adam_lms/q_table",
    basename="q_table_adam_lms",
    N=ADAM_N,
    M=ADAM_M,
    alpha=ADAM_ALPHA,
):
    """Run Adam-LMS for patients 1–8 and save Q1/Q2 results to CSV.

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
                alpha,
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
