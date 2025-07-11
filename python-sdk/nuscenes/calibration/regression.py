import numpy as np
import pandas as pd
from scipy.special import erfinv
from typing import Literal


def regression_precision_recall_df(
    y_pred: np.ndarray,
    var_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 50,
    n_samples: int | None = None,
) -> pd.DataFrame:
    """
    Compute MAE / MSE / RMSE as the most-uncertain points are progressively
    removed (lower variance = higher “confidence”).
    """
    y_pred = np.asarray(y_pred).ravel()
    var_pred = np.asarray(var_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    # if y_true is a scalar, we assume it is the same for all predictions
    if y_true.ndim == 0:
        y_true = np.full_like(y_pred, y_true)

    if n_samples is not None and n_samples < len(y_pred):
        idx = np.random.permutation(len(y_pred))[:n_samples]
        y_pred, var_pred, y_true = y_pred[idx], var_pred[idx], y_true[idx]

    # sort by predicted variance (ascending == most confident first)
    order   = np.argsort(var_pred, axis=0)
    y_pred  = y_pred[order]
    y_true  = y_true[order]

    diff = y_pred - y_true
    cutoff_percentiles = np.arange(1, n_bins) / (n_bins - 1)
    cutoff_indices = (cutoff_percentiles * len(y_pred)).astype(int)

    mae = np.asarray([np.abs(diff[:k]).mean() for k in cutoff_indices])
    mse = np.asarray([(diff[:k] ** 2).mean() for k in cutoff_indices])

    return pd.DataFrame(
        {
            "percentile": cutoff_percentiles,
            "mae": mae,
            "mse": mse,
            "rmse": np.sqrt(mse),
        }
    )


def regression_calibration_df(
    y_pred: np.ndarray,
    var_pred: np.ndarray,
    y_true: np.ndarray,
    distribution: Literal["normal", "laplace"] = "normal",
    n_bins: int = 50,
    n_samples: int | None = None,
) -> pd.DataFrame:
    r"""
    Compute expected vs. observed coverage for regression uncertainties.

    See §5 of *Kuleshov et al., “Accurate Uncertainties for Deep Learning”,
    ICML 2018* for background.
    """
    y_pred = np.asarray(y_pred).ravel()
    var_pred = np.asarray(var_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    # if y_true is a scalar, we assume it is the same for all predictions
    if y_true.ndim == 0:
        y_true = np.full_like(y_pred, y_true)

    if n_samples is not None and n_samples < len(y_pred):
        idx = np.random.permutation(len(y_pred))[:n_samples]
        y_pred, var_pred, y_true = y_pred[idx], var_pred[idx], y_true[idx]

    if distribution == "normal":
        icdf_fn = _normal_icdf
    elif distribution == "laplace":
        icdf_fn = _laplace_icdf
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    # Vectorised: confidence_levels[:, None] broadcasts across samples
    confidence_levels = np.linspace(0.0, 1.0, n_bins)
    icdf_vals = icdf_fn(confidence_levels[:, None], loc=y_pred, var=var_pred)
    observed_p = (y_true <= icdf_vals).mean(axis=1)

    return pd.DataFrame(
        {
            "expected_p": confidence_levels,
            "observed_p": observed_p,
        }
    )

def _normal_icdf(p: np.ndarray, *, loc: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Inverse CDF (quantile) of N(loc, var).  Uses scipy's erfinv
    """
    scale = np.sqrt(var)
    return loc + scale * np.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def _laplace_icdf(p: np.ndarray, *, loc: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Inverse CDF of the Laplace distribution with variance `var`.
    """
    scale = np.sqrt(var) / np.sqrt(2.0)  # because Var = 2 b²
    return loc - scale * np.sign(p - 0.5) * np.log1p(-2.0 * np.abs(p - 0.5))