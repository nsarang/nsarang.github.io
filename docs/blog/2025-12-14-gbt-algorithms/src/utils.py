import math

import numba as nb
import numpy as np


@nb.njit
def groupby_sum_2d(
    n_groups: int,
    group_indices: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    sample_weight: np.ndarray,
):
    """
    Compute grouped sums of gradients and hessians for 2D arrays. Uses Numba for speed up.
    Input Shape: `(n_samples, n_outputs)`
    Output Shape: `(n_groups, n_outputs)`
    """
    n_outputs = gradients.shape[1]

    grad_sums = np.empty((n_groups, n_outputs))
    hess_sums = np.empty((n_groups, n_outputs))
    weight_sums = np.bincount(group_indices, weights=sample_weight, minlength=n_groups)

    for j in range(n_outputs):
        grad_sums[:, j] = np.bincount(group_indices, weights=gradients[:, j], minlength=n_groups)
        hess_sums[:, j] = np.bincount(group_indices, weights=hessians[:, j], minlength=n_groups)

    return grad_sums, hess_sums, weight_sums


def map_array(arr: np.ndarray, mapping: dict, missing_value=np.nan) -> np.ndarray:
    """
    Map the values in the input array according to the provided mapping dictionary.
    """
    return np.vectorize(lambda x: mapping.get(x, missing_value))(arr)


def to_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure the input array is 2D. If 1D, reshape to (n_samples, 1)."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError(f"Input array must be 1D or 2D, but got {arr.ndim}D.")
    return arr


@nb.njit
def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""
    Compute the sigmoid function for each element in the input array.
    $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
    """
    return 1 / (1 + np.exp(-x))


def trunc(num, threshold_percent=0.01, truncate_integer=False):
    """
    Adaptively truncate a number based on a percentage of its magnitude. I use this
    pretty much everywhere to make sure numerical outputs don't look ugly.
    """
    if num == 0:
        return 0

    sign = 1 if num > 0 else -1
    num = abs(num)

    min_change = num * (threshold_percent / 100)
    decimal_places = math.ceil(-math.log10(min_change))

    # Check if we need to lose an additional decimal place
    scale_d = 10 ** (decimal_places - 1)
    digit_value_to_lose = ((num * scale_d) % 1) / scale_d
    if min_change >= digit_value_to_lose:
        decimal_places -= 1

    # Don't truncate integer digits unless explicitly allowed
    if not truncate_integer:
        decimal_places = max(decimal_places, 0)

    scale = 10**decimal_places
    truncated = int(num * scale) / scale
    return sign * truncated
