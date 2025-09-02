# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.

import json
from typing import Any, List, Optional, Dict
import os

import numpy as np
import math
from matplotlib import pyplot as plt
import tempfile
import wandb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricData, DetectionMetricDataList
from nuscenes.eval.detection.constants import TP_METRICS_PLOT, DETECTION_NAMES, DETECTION_COLORS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.calibration.visualization import plot_calibration, plot_precision_recall

Axis = Any


def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None,
                     wandb_log: bool = False,
                     wandb_name: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    Optionally saves or logs the plot to Weights & Biases.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT and EST boxes to lidar frame.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT and EST boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    for box in boxes_est:
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    plt.title(sample_token)

    if verbose:
        print('Rendering sample token %s' % sample_token)

    # Save or show
    if savepath is not None:
        plt.savefig(savepath)
        if not wandb_log:
            plt.close()

    if wandb_log:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            wandb.log({wandb_name or f"sample_{sample_token}": wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)

        plt.close()
    elif savepath is None:
        plt.show()


def class_pr_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_precision: float,
                   min_recall: float,
                   savepath: str = None,
                   ax: Axis = None,
                   wandb_log: bool = False,
                   wandb_name: str = None) -> None:
    """
    Plot a precision-recall curve for the specified class.
    Optionally saves to disk or logs to Weights & Biases.
    """
    created_ax = False
    if ax is None:
        created_ax = True
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Precision',
                        xlim=1, ylim=1, min_precision=min_precision, min_recall=min_recall)

    data = md_list.get_class_data(detection_name)

    for md, dist_th in data:
        md: DetectionMetricData
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        if not wandb_log:
            plt.close()

    if wandb_log:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            wandb.log({wandb_name or f"PR_{detection_name}": wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)
        plt.close()

    elif created_ax and savepath is None:
        plt.show()


def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None,
                   wandb_log: bool = False,
                   wandb_name: str = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.
    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS_PLOT if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS_PLOT:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        if not wandb_log:
            plt.close()

    if wandb_log:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            wandb.log({wandb_name or f"TP_{detection_name}": wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)
        plt.close()

    elif savepath is None:
        plt.show()

def class_ece_curve(
    md_list: DetectionMetricDataList,
    metrics: DetectionMetrics,
    detection_name: str,
    dist_th_ece: float,
    savepath: str = None,
    ax: Axis = None,
    wandb_log: bool = False,
    wandb_name: str = None,
) -> None:
    """
    Plot the Expected Calibration Error (ECE) curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: Name of the detection class.
    :param dist_th_ece: The distance threshold used to determine matches.
    :param savepath: If given, saves the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    md = md_list[(detection_name, dist_th_ece)]

    created_ax = False
    if ax is None:
        created_ax = True
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i, (key, calib_df) in enumerate(md.calib_dfs.items()):
        label = f"{key}"
        show_ideal = (i == len(md.calib_dfs) - 1)  # Show ideal line only for the last curve
        plot_calibration(
            calib_df,
            label=label,
            ax=ax,
            show_ideal=show_ideal,
            show_ece=True,
            show_area=False,
        )

    ax.set_title(PRETTY_DETECTION_NAMES[detection_name])
    ax.legend()

    if savepath is not None:
        plt.savefig(savepath)
        if not wandb_log:
            plt.close()

    if wandb_log:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            wandb.log({wandb_name or f"ECE_{detection_name}": wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)
        plt.close()

    elif created_ax and savepath is None:
        plt.show()


def class_prec_rec_curve(
    md_list: DetectionMetricDataList,
    detection_name: str,
    dist_th_prec: float,
    savepath: str = None,
    ax: Axis = None,
    wandb_log: bool = False,
    wandb_name: str = None,
) -> None:
    """
    Plot a precision-recall metric curve for the specified class and distance threshold.
    Optionally saves to disk or logs to Weights & Biases.
    """
    md = md_list[(detection_name, dist_th_prec)]

    created_ax = False
    if ax is None:
        created_ax = True
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    for key, prec_df in md.prec_rec_dfs.items():
        plot_precision_recall(prec_df, label=key, ax=ax)

    ax.set_title(PRETTY_DETECTION_NAMES[detection_name])
    ax.legend()

    if savepath is not None:
        plt.savefig(savepath)
        if not wandb_log:
            plt.close()

    if wandb_log:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            wandb.log({wandb_name or f"PrecRec_{detection_name}": wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)
        plt.close()

    elif created_ax and savepath is None:
        plt.show()



def dist_pr_curve(md_list: DetectionMetricDataList,
                  metrics: DetectionMetrics,
                  dist_th: float,
                  min_precision: float,
                  min_recall: float,
                  savepath: str = None,
                  wandb_log: bool = False,
                  wandb_name: str = None) -> None:
    """
    Plot the PR curves for all classes at a fixed distance threshold.
    Optionally saves to disk or logs to Weights & Biases.
    """
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(7.5, 5))

    ax = setup_axis(xlabel='Recall', ylabel='Precision',
                    xlim=1, ylim=1, min_precision=min_precision, min_recall=min_recall, ax=ax)

    # Plot PR curve for each class
    data = md_list.get_dist_data(dist_th)
    for md, detection_name in data:
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision,
                label=f'{PRETTY_DETECTION_NAMES[detection_name]}: {ap * 100:.1f}%',
                color=DETECTION_COLORS[detection_name])

    # Legend
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        if not wandb_log:
            plt.close()

    if wandb_log:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            wandb.log({wandb_name or f"dist_PR_{dist_th}m": wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)
        plt.close()

    elif savepath is None:
        plt.show()


def summary_plot(md_list: DetectionMetricDataList,
                 metrics: DetectionMetrics,
                 min_precision: float,
                 min_recall: float,
                 dist_th_tp: float,
                 savepath: Optional[str] = None,
                 wandb_log: bool = False,
                 wandb_name: str = "summary_plot") -> None:
    """
    Creates a summary plot with PR and TP curves for each class.
    Optionally saves to disk and/or logs to Weights & Biases.
    """
    n_classes = len(DETECTION_NAMES)
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes))
    for ind, detection_name in enumerate(DETECTION_NAMES):
        title1, title2 = ('Recall vs Precision', 'Recall vs Error') if ind == 0 else (None, None)

        ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind, 0])
        ax1.set_ylabel('{} \n \n Precision'.format(PRETTY_DETECTION_NAMES[detection_name]), size=20)

        ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind, 1])
        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name,  min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        if not wandb_log:
            plt.close()

    if wandb_log:
        # Save to temporary file if not already saved
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            wandb.log({wandb_name: wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)  # remove temp file

        plt.close()


def detailed_results_table_tex(metrics_path: str, output_path: str) -> None:
    """
    Renders a detailed results table in tex.
    :param metrics_path: path to a serialized DetectionMetrics file.
    :param output_path: path to the output file.
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    tex = ''
    tex += '\\begin{table}[]\n'
    tex += '\\small\n'
    tex += '\\begin{tabular}{| c | c | c | c | c | c | c |} \\hline\n'
    tex += '\\textbf{Class}    &   \\textbf{AP}  &   \\textbf{ATE} &   \\textbf{ASE} & \\textbf{AOE}   & ' \
           '\\textbf{AVE}   & ' \
           '\\textbf{AAE}   \\\\ \\hline ' \
           '\\hline\n'
    for name in DETECTION_NAMES:
        ap = np.mean(metrics['label_aps'][name].values()) * 100
        ate = metrics['label_tp_errors'][name]['trans_err']
        ase = metrics['label_tp_errors'][name]['scale_err']
        aoe = metrics['label_tp_errors'][name]['orient_err']
        ave = metrics['label_tp_errors'][name]['vel_err']
        aae = metrics['label_tp_errors'][name]['attr_err']
        tex_name = PRETTY_DETECTION_NAMES[name]
        if name == 'traffic_cone':
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   N/A  &   N/A  &   N/A  \\\\ \\hline\n'.format(
                tex_name, ap, ate, ase)
        elif name == 'barrier':
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   N/A  &   N/A  \\\\ \\hline\n'.format(
                tex_name, ap, ate, ase, aoe)
        else:
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  \\\\ ' \
                   '\\hline\n'.format(tex_name, ap, ate, ase, aoe, ave, aae)

    map_ = metrics['mean_ap']
    mate = metrics['tp_errors']['trans_err']
    mase = metrics['tp_errors']['scale_err']
    maoe = metrics['tp_errors']['orient_err']
    mave = metrics['tp_errors']['vel_err']
    maae = metrics['tp_errors']['attr_err']
    tex += '\\hline {} &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  \\\\ ' \
           '\\hline\n'.format('\\textbf{Mean}', map_, mate, mase, maoe, mave, maae)

    tex += '\\end{tabular}\n'

    # All one line
    tex += '\\caption{Detailed detection performance on the val set. \n'
    tex += 'AP: average precision averaged over distance thresholds (%), \n'
    tex += 'ATE: average translation error (${}$), \n'.format(TP_METRICS_UNITS['trans_err'])
    tex += 'ASE: average scale error (${}$), \n'.format(TP_METRICS_UNITS['scale_err'])
    tex += 'AOE: average orientation error (${}$), \n'.format(TP_METRICS_UNITS['orient_err'])
    tex += 'AVE: average velocity error (${}$), \n'.format(TP_METRICS_UNITS['vel_err'])
    tex += 'AAE: average attribute error (${}$). \n'.format(TP_METRICS_UNITS['attr_err'])
    tex += 'nuScenes Detection Score (NDS) = {:.1f} \n'.format(metrics['nd_score'] * 100)
    tex += '}\n'

    tex += '\\end{table}\n'

    with open(output_path, 'w') as f:
        f.write(tex)


def plot_bev_heatmaps(
    md,
    keys: Optional[List[str]] = None,
    *,
    min_count: int = 0,
    group_vmax: bool = False,
    cmap: str = "viridis",
    ncols: int = 3,
    figsize_per_plot: float = 2.6,
    savepath: Optional[str] = None,
    wandb_log: bool = False,
    wandb_prefix: str = "BEV",
    compact: bool = True,
    # ---- bumped font sizes (3–4× larger) ----
    title_fontsize: int = 32,
    tick_fontsize: int = 28,
    label_fontsize: int = 32,
    cbar_fraction: float = 0.9,
    cbar_pad: float = 0.02,
) -> Dict[str, Any]:
    """
    Plot one or more BEV heatmaps stored in md.bev_heatmaps with compact layout.

    Independent color scaling per subplot by default (group_vmax=False).
    If compact=True, shows only outer x/y labels and uses skinny inset colorbars.
    """
    hm = getattr(md, "bev_heatmaps", None)
    if not hm or "x_edges" not in hm or "y_edges" not in hm:
        raise ValueError("md.bev_heatmaps is missing or incomplete (need x_edges, y_edges, and at least one layer).")

    x_edges = hm["x_edges"]
    y_edges = hm["y_edges"]
    count = hm.get("count", None)

    # Default set of layers to plot (preserve your preferred per-class order)
    if keys is None:
        preferred = [
            "mse_pos", "ale_pos", "epi_pos",
            "mse_vel", "ale_vel", "epi_vel",
            "ale_mean", "epi_mean", "count",
        ]
        keys = [k for k in preferred if k in hm]

    # Mask for min_count
    mask = None
    if min_count and count is not None:
        mask = (count < min_count)

    # Helper
    def group_of(k: str) -> str:
        if k.startswith("mse_"): return "mse"
        if k.startswith("ale_"): return "ale"
        if k.startswith("epi_"): return "epi"
        return k  # 'count' or anything else

    # Compute per-group ranges (only if requested)
    vmin_vmax: Dict[str, Dict[str, float]] = {}
    if group_vmax:
        groups = {}
        for k in keys:
            if k == "count":  # leave count out of grouped scaling
                continue
            g = group_of(k)
            Z = np.array(hm[k], dtype=float)
            if mask is not None:
                Z = Z.copy()
                Z[mask] = np.nan
            vmin = np.nanmin(Z)
            vmax = np.nanmax(Z)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = 0.0, 1.0
            if g not in groups:
                groups[g] = [vmin, vmax]
            else:
                groups[g][0] = min(groups[g][0], vmin)
                groups[g][1] = max(groups[g][1], vmax)
        for k in keys:
            if k == "count":
                continue
            g = group_of(k)
            vmin_vmax[k] = {"vmin": groups[g][0], "vmax": groups[g][1]}

    # Layout
    n = len(keys)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_plot * ncols, figsize_per_plot * nrows),
        squeeze=False
    )

    info = {"fig": fig, "vmin_vmax": {}}

    # Determine which axes are on the outer edges (for compact labeling)
    def is_bottom(r): return (r == nrows - 1)
    def is_left(c):   return (c == 0)

    for i, k in enumerate(keys):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        Z = np.array(hm[k], dtype=float) if k in hm else None
        if Z is None:
            ax.axis("off")
            continue

        # Apply mask to non-count layers
        if mask is not None and k != "count":
            Z = Z.copy()
            Z[mask] = np.nan

        # Color limits
        if k == "count":
            finite = np.isfinite(Z)
            if finite.any():
                this_vmin, this_vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
            else:
                this_vmin, this_vmax = 0.0, 1.0
            this_cmap = "Greys"
        else:
            if group_vmax and k in vmin_vmax:
                this_vmin, this_vmax = vmin_vmax[k]["vmin"], vmin_vmax[k]["vmax"]
            else:
                finite = np.isfinite(Z)
                if finite.any():
                    this_vmin, this_vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
                    if not np.isfinite(this_vmin) or not np.isfinite(this_vmax):
                        this_vmin, this_vmax = 0.0, 1.0
                else:
                    this_vmin, this_vmax = 0.0, 1.0
            this_cmap = cmap

        im = ax.imshow(
            Z.T, origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal",
            vmin=this_vmin,            # <— use true vmin
            vmax=this_vmax,
            cmap=this_cmap
        )

        # Titles & labels (compact mode trims noise)
        ax.set_title(k, fontsize=title_fontsize, pad=2)
        if compact:
            if is_left(c):
                ax.set_ylabel("y (m)", fontsize=label_fontsize)
                ax.tick_params(labelsize=tick_fontsize)
            else:
                ax.set_yticklabels([])
            if is_bottom(r):
                ax.set_xlabel("x (m)", fontsize=label_fontsize)
                ax.tick_params(labelsize=tick_fontsize)
            else:
                ax.set_xticklabels([])
        else:
            ax.set_xlabel("x (m)", fontsize=label_fontsize)
            ax.set_ylabel("y (m)", fontsize=label_fontsize)
            ax.tick_params(labelsize=tick_fontsize)

        # Inset skinny colorbar (per-axis, works with independent scales)
        cax = inset_axes(ax, width="3%", height=f"{int(cbar_fraction*100)}%",
                         loc="right", borderpad=0.0)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=tick_fontsize - 1)
        cb.ax.set_ylabel("", rotation=270, labelpad=6)

        info["vmin_vmax"][k] = {"vmin": this_vmin, "vmax": this_vmax}

    # Turn off any empty axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # Tighten up spacing
    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    if wandb_log:
        import tempfile, os, wandb
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, dpi=200, bbox_inches="tight")
            wandb.log({f"{wandb_prefix}/heatmaps": wandb.Image(tmpfile.name)})
            os.unlink(tmpfile.name)

    return info
