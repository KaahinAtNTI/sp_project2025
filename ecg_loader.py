"""ECG data loading and preprocessing utilities."""

import os
import numpy as np
import scipy.io

from config import FS


def _patient_dir(patient_no):
    """Return dataset directory for a patient.

    Args:
        patient_no: Patient index.

    Returns:
        Path to the patient's dataset directory.
    """
    base = os.path.dirname(__file__)
    return os.path.join(base, "DATASET", f"ECG_{patient_no}")


def _load_vec(path, key):
    """Load and flatten a vector from a .mat file.

    Args:
        path: Path to .mat file.
        key: Variable name inside the .mat file.

    Returns:
        1D numpy array.
    """
    return np.ravel(scipy.io.loadmat(path)[key])


def load_patient_data(patient_no):
    """Load, split, and zero-mean center ECG signals for one patient.

    Args:
        patient_no: Patient index.

    Returns:
        Dict with train/test arrays and means, or None on error.
    """
    pdir = _patient_dir(patient_no)

    spec = {
        "x1": ("V", "ECG_{}_V.mat", "ECG_{}_V"),
        "x2": ("AVR", "ECG_{}_AVR.mat", "ECG_{}_AVR"),
        "d": ("II", "ECG_{}_II.mat", "ECG_{}_II"),
        "d_missing": ("II_missing", "ECG_{}_II_missing.mat", "ECG_{}_II_missing"),
    }

    paths_keys = {}
    for name, (_, fpat, kpat) in spec.items():
        path = os.path.join(pdir, fpat.format(patient_no))
        if not os.path.exists(path):
            print(f"Missing file: {os.path.basename(path)}")
            return None
        paths_keys[name] = (path, kpat.format(patient_no))

    try:
        x1 = _load_vec(*paths_keys["x1"])
        x2 = _load_vec(*paths_keys["x2"])
        d = _load_vec(*paths_keys["d"])
        d_missing = _load_vec(*paths_keys["d_missing"])
    except Exception as exc:
        print(f"Error loading patient {patient_no}: {exc}")
        return None

    split_idx = int(9.5 * 60 * FS)

    mu1 = float(np.mean(x1[:split_idx]))
    mu2 = float(np.mean(x2[:split_idx]))
    muT = float(np.mean(d[:split_idx]))

    x1c, x2c, dc = x1 - mu1, x2 - mu2, d - muT

    return {
        "x1_train": x1c[:split_idx],
        "x2_train": x2c[:split_idx],
        "d_train": dc[:split_idx],
        "x1_test": x1c[split_idx:],
        "x2_test": x2c[split_idx:],
        "d_test_true": d_missing - muT,
        "means": (mu1, mu2, muT),
        "fs": FS,
    }
