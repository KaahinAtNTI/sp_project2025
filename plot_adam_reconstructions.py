"""Plot Adam-LMS reconstruction."""

import os

import matplotlib.pyplot as plt
import numpy as np

from adam_lms import run_experiment
from config import ADAM_ALPHA, ADAM_M, ADAM_N
from ecg_loader import load_patient_data


def plot_adam_reconstructions(
    patients=(2, 4), save_dir="results/adam_lms/reconstructions", seconds=2
):
    """Plot a symmetric ±seconds window around the split point with Adam-LMS reconstruction.

    Args:
        patients: Iterable of patient indices to plot.
        save_dir: Directory where figures are saved.
        seconds: Seconds shown before and after the split point.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    for patient_no in patients:
        data = load_patient_data(patient_no)
        res = run_experiment(patient_no, N=ADAM_N, M=ADAM_M, alpha=ADAM_ALPHA)
        if data is None or res is None:
            continue

        _, _, d_true_test, d_est_test = res
        d_train = data["d_train"]
        fs = data["fs"]

        split_idx = len(d_train)
        split_t = split_idx / fs

        d_true_full = np.concatenate([d_train, d_true_test])

        half_win = int(float(seconds) * fs)
        start_idx = max(0, split_idx - half_win)
        end_idx = min(len(d_true_full), split_idx + half_win)

        t = np.arange(start_idx, end_idx) / fs
        y_true = d_true_full[start_idx:end_idx]

        test_start = max(split_idx, start_idx)
        test_end = min(split_idx + len(d_est_test), end_idx)
        t_test = np.arange(test_start, test_end) / fs
        y_est = d_est_test[(test_start - split_idx) : (test_end - split_idx)]

        path = os.path.join(save_dir, f"recon_adam_lms_patient_{patient_no:02d}.png")

        plt.figure(figsize=(9, 6))
        plt.plot(t, y_true, label="True signal")
        plt.plot(t_test, y_est, label="Adam-LMS Reconstruction", linestyle="--")

        plt.axvline(split_t, color="gray", linestyle=":", label="Split point")

        plt.title(f"ECG Signal Reconstruction (Patient {patient_no}) - Adam-LMS")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [mV]")

        plt.legend(
            loc="best",
            framealpha=1,
            facecolor="white",
            edgecolor="black",
        )
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    plot_adam_reconstructions()
