"""Calibration metrics."""

import numpy as np


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE for classification (simple version)."""
    probs = np.array(probs)
    labels = np.array(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs > bin_lower) & (probs <= bin_upper)
        if mask.sum() > 0:
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(acc - conf)
    return ece


def prediction_interval_coverage(lower, upper, y):
    lower = np.array(lower)
    upper = np.array(upper)
    y = np.array(y)
    covered = ((y >= lower) & (y <= upper)).mean()
    return covered
