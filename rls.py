"""RLS filter and quality metrics for ECG reconstruction."""

import numpy as np

from ecg_loader import load_patient_data
from config import RLS_LAMBDA, RLS_M, RLS_N, RLS_DELTA


class RLSFilter:
    """Recursive Least Squares filter with two inputs."""

    def __init__(self, N=RLS_N, M=RLS_M, lam=RLS_LAMBDA, delta=RLS_DELTA):
        """Initialize the RLS filter.

        Args:
            N: Order for first input signal.
            M: Order for second input signal.
            lam: Forgetting factor.
            delta: Initial scaling for inverse correlation matrix P.
        """
        self.N = N
        self.M = M
        self.lam = lam

        self.L = (N + 1) + (M + 1)
        self.w = np.zeros(self.L)
        self.P = np.eye(self.L) * delta

    def _input_vector(self, x1, x2, n):
        """Construct input vector u[n] from x1 and x2.

        Args:
            x1: First input signal.
            x2: Second input signal.
            n: Sample index.

        Returns:
            1D NumPy array u[n].
        """
        if n >= self.N:
            u1 = x1[n - self.N : n + 1][::-1]
        else:
            history = x1[0 : n + 1][::-1]
            padding = np.zeros(self.N + 1 - len(history))
            u1 = np.concatenate((history, padding))

        if n >= self.M:
            u2 = x2[n - self.M : n + 1][::-1]
        else:
            history = x2[0 : n + 1][::-1]
            padding = np.zeros(self.M + 1 - len(history))
            u2 = np.concatenate((history, padding))

        return np.concatenate((u1, u2))

    def train(self, x1, x2, d):
        """Adapt filter weights over a training set.

        Args:
            x1: First input training signal.
            x2: Second input training signal.
            d: Desired signal.

        Returns:
            w_final: Final weight vector.
            errors: A priori error for each sample.
        """
        num_samples = len(d)
        errors = np.zeros(num_samples)

        for n in range(num_samples):
            u = self._input_vector(x1, x2, n)
            d_hat = np.dot(self.w, u)
            e = d[n] - d_hat
            errors[n] = e

            Pu = np.dot(self.P, u)
            uPu = np.dot(u, Pu)
            gamma = 1.0 / (self.lam + uPu)
            k = gamma * Pu

            self.w = self.w + k * e
            self.P = (1.0 / self.lam) * (self.P - np.outer(k, Pu))

        return self.w, errors

    def predict(self, x1, x2):
        """Generate predictions using frozen weights.

        Args:
            x1: First input test signal.
            x2: Second input test signal.

        Returns:
            1D NumPy array of predicted samples.
        """
        num_samples = len(x1)
        y = np.zeros(num_samples)

        for n in range(num_samples):
            u = self._input_vector(x1, x2, n)
            y[n] = np.dot(self.w, u)

        return y


def calculate_metrics(d_true, d_est):
    """Compute Q1 and Q2 quality measures.

    Args:
        d_true: True target signal.
        d_est: Estimated target signal.

    Returns:
        Q1: Variance-based metric.
        Q2: Correlation-based metric.
    """
    mse = np.mean((d_true - d_est) ** 2)
    var_true = np.var(d_true)
    var_est = np.var(d_est)

    Q1 = 1 - mse / var_true

    if var_est == 0:
        Q2 = 0.0
    else:
        cov = np.cov(d_true, d_est)[0, 1]
        Q2 = cov / np.sqrt(var_true * var_est)

    Q1 = max(0.0, Q1)
    Q2 = max(0.0, Q2)

    return Q1, Q2


def run_experiment(patient_no, N=RLS_N, M=RLS_M, lam=RLS_LAMBDA):
    """Train and evaluate an RLS filter for a given patient.

    Args:
        patient_no: Patient index.
        N: Order for first input.
        M: Order for second input.
        lam: Forgetting factor.

    Returns:
        Q1: Quality measure Q1.
        Q2: Quality measure Q2.
        d_true: True test signal (centered).
        d_est: Estimated test signal (centered).
        Returns None if data could not be loaded.
    """
    data = load_patient_data(patient_no)
    if data is None:
        return None

    rls = RLSFilter(N=N, M=M, lam=lam)
    rls.train(data["x1_train"], data["x2_train"], data["d_train"])

    d_est = rls.predict(data["x1_test"], data["x2_test"])
    d_true = data["d_test_true"]

    Q1, Q2 = calculate_metrics(d_true, d_est)
    return Q1, Q2, d_true, d_est


if __name__ == "__main__":
    run_experiment(2)
