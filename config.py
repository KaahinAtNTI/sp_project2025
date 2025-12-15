"""Shared configuration values for the signal processing project."""

# Sampling frequency
FS = 125  # Hz


# ======================================
# RLS DEFAULT PARAMETERS
# ======================================
RLS_N = 10
RLS_M = 10
RLS_LAMBDA = 1.0
RLS_DELTA = 100.0


# ======================================
# ADAM-LMS DEFAULT PARAMETERS
# ======================================
ADAM_N = 10
ADAM_M = 10

ADAM_ALPHA = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
