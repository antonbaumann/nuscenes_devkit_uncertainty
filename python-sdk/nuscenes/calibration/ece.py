import pandas as pd


def expected_calibration_error(calibration_df: pd.DataFrame) -> float:
    r"""
    Compute the expected calibration error.

    $$ ECE = \sum_{i=1}^{M} w_i \left|p_{obs} - p_{exp}\right| $$,

    where $w_i = \frac{N_i}{N}$ if `count` is present in the calibration dataframe, and $w_i = 1/M$ otherwise.

    Args:
        calibration_df (pd.DataFrame): The calibration dataframe.
    
    Returns:
        `float`: The expected calibration error.
    """

    abs_diff = (calibration_df['observed_p'] - calibration_df['expected_p']).abs()

    if 'count' in calibration_df.columns:
        total_count = calibration_df['count'].sum()
        weight = calibration_df['count'] / total_count
        return (abs_diff * weight).sum()

    return  abs_diff.mean()