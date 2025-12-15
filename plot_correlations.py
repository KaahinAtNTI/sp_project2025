"""Plot cross-correlations between ECG inputs and target."""

import os

import matplotlib.pyplot as plt
import numpy as np

from ecg_loader import load_patient_data


def plot_correlations(patients=(2, 4), lag_window=500, save_dir="results/correlations"):
    """Generate and save cross-correlation plots for selected patients.

    Args:
        patients: Iterable of patient indices to plot.
        lag_window: Half-width of lag window in samples.
        save_dir: Directory where figures are saved.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    for patient_no in patients:
        data = load_patient_data(patient_no)

        x1 = data["x1_train"]
        x2 = data["x2_train"]
        d = data["d_train"]

        corr_1 = np.correlate(x1, d, mode="full")
        corr_2 = np.correlate(x2, d, mode="full")

        norm_1 = np.sqrt(np.sum(x1**2) * np.sum(d**2))
        norm_2 = np.sqrt(np.sum(x2**2) * np.sum(d**2))

        corr_1_norm = corr_1 / norm_1 if norm_1 != 0 else corr_1 * 0.0
        corr_2_norm = corr_2 / norm_2 if norm_2 != 0 else corr_2 * 0.0

        lags = np.arange(-len(x1) + 1, len(x1))

        path = os.path.join(save_dir, f"corr_patient_{patient_no:02d}.png")

        plt.figure(figsize=(9, 6))
        plt.plot(lags, corr_1_norm, label="Corr(V, II)")
        plt.plot(lags, corr_2_norm, label="Corr(AVR, II)", alpha=0.8)

        plt.axvline(0, color="gray", linestyle=":", linewidth=1)

        plt.xlabel("Lag (samples)")
        plt.ylabel("Normalized correlation")
        plt.title(f"Cross-correlation (Training Data) — Patient {patient_no}")
        plt.xlim(-lag_window, lag_window)
        plt.grid(alpha=0.3)

        plt.legend(
            loc="best",
            framealpha=1,
            facecolor="white",
            edgecolor="black",
        )

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    plot_correlations()
