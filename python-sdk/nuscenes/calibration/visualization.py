from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd

from .ece import expected_calibration_error


def plot_calibration(
    df: pd.DataFrame,
    label: Optional[str] = None,
    show_ece: bool = True,
    show_ideal: bool = True,
    show_area: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the calibration curve for a model.

    Args:
        df: DataFrame containing the calibration data.
        label: Label for the calibration curve.
        show_ece: Whether to display the Expected Calibration Error (ECE) in the plot.
        show_ideal: Whether to display the ideal calibration line.
        show_area: Whether to color the area between the calibration curve and the ideal line.
        ax: Matplotlib axes to plot on.

    Returns:
        Matplotlib axes with the calibration plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Compute ECE if needed and include it in the legend label
    ece = expected_calibration_error(df) if show_ece else None
    label_with_ece = f"{label} (ECE={ece:.3f})" if show_ece and label else f"ECE={ece:.3f}" if show_ece else label
    ax.plot(df['expected_p'], df['observed_p'], label=label_with_ece, linewidth=1.5)


    if show_ideal:
        # Adding a line plot for perfectly calibrated predictions
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=1.5)

    if show_area:
        # Coloring area between the calibration line and the perfect calibration line
        ax.fill_between(df['expected_p'], df['expected_p'], df['observed_p'], alpha=0.2)

    # Setting labels with increased font size for better readability
    ax.set_xlabel('Expected Proportion', fontsize=12)
    ax.set_ylabel('Observed Proportion', fontsize=12)

    # Set the range of x and y axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend(frameon=False)

    return ax


def plot_precision_recall(
    df: pd.DataFrame,
    metric: str = 'mse',
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the precision-recall curve for a model.

    Args:
        df (pd.DataFrame): DataFrame containing the calibration data.
        metric (str): Metric to plot on the y-axis.
        label (str): Label for the calibration curve.
        ax (plt.Axes): Matplotlib axes to plot on.

    Returns:
        Matplotlib axes with the precision-recall plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Plot precision-recall curve using matplotlib
    ax.plot(df['percentile'], df[metric], label=label, linewidth=1.5)

    # Setting labels with increased font size for better readability
    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)

    # Set the range of x axis
    ax.set_xlim([-0.02, 1.02])

    # Add legend if label is provided
    if label is not None:
        ax.legend(frameon=False)

    return ax