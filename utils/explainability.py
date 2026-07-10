"""Explainability utilities for ECG relevance maps."""
import numpy as np


def normalize_ecg_relevancemap(R, local=False):
    """
    Normalize an ECG relevance map.

    Args:
        R (np.ndarray): The relevance map to be normalized.
        local (bool): If True, normalizes each lead individually. If False, normalizes globally across all leads.

    Returns:
        np.ndarray: The normalized relevance map with values in [-1, 1].
    """
    if local is False:
        # Normalize R to [-1, 1]
        Rn = R / np.max(np.abs(R))
    else:
        # Normalize each lead in R to [-1, 1]
        Rn = np.zeros_like(R)
        for i in range(np.shape(R)[1]):
            Rn[..., i] = R[..., i] / np.max(np.abs(R[..., i]))

    # Replace any nan to 0
    Rn = np.nan_to_num(Rn, nan=0)

    return Rn
