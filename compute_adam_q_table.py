"""Compute Q1/Q2 metrics for all patients using Adam-LMS."""

import os
import pandas as pd

from config import (
    ADAM_ALPHA,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    ADAM_M,
    ADAM_N,
)
from adam_lms import run_experiment


def generate_q_table(
    output_dir="results/adam/q_table",
    basename="q_table_adam",
    N=ADAM_N,
    M=ADAM_M,
    alpha=ADAM_ALPHA,
    beta1=ADAM_BETA1,
    beta2=ADAM_BETA2,
    eps=ADAM_EPSILON,
):
    """Generate and save a Q1/Q2 table for patients 1–8 using Adam-LMS.

    Args:
        output_dir: Directory where the CSV file is saved.
        basename: Base name of the output CSV file (without extension).
        N: Number of past samples used from ECG V.
        M: Number of past samples used from ECG AVR.
        alpha: Adam step size.
        beta1: Adam first moment decay.
        beta2: Adam second moment decay.
        eps: Adam numerical stability constant.

    Returns:
        pandas.DataFrame with columns: Patient, Q1, Q2.
    """
    results = []

    for patient_no in range(1, 9):
        Q1, Q2, _, _ = run_experiment(
            patient_no,
            N,
            M,
            alpha,
            beta1,
            beta2,
            eps,
        )
        results.append(
            {
                "Patient": patient_no,
                "Q1": Q1,
                "Q2": Q2,
            }
        )

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{basename}.csv")
    df.to_csv(csv_path, index=False)

    return df


if __name__ == "__main__":
    generate_q_table()
