# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

from typing import Callable, List
from tqdm import tqdm

import numpy as np
from scipy import stats

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean, \
    gaussian_nll_error, within_cofidence_interval, center_offset, velocity_offset, center_offset_var, velocity_offset_var
from nuscenes.calibration.regression import regression_precision_recall_df, regression_calibration_df


def accumulate(
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    class_name: str,
    dist_fcn: Callable,
    dist_th: float,
    verbose: bool = False,
    confidence_interval_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    uncertainty_distribution: str = "gaussian",
    num_bins_precision_recall: int = 50,
    num_bins_calibration: int = 15,
    compute_ci: bool = True,
    compute_ece: bool = True,
) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    ci_accumulation = {ci: [] for ci in confidence_interval_values}
    distribution = stats.laplace if uncertainty_distribution == 'laplace' else stats.norm

    ece_util_keys = [
        'trans_err_x', 'trans_err_y', 'vel_err_x', 'vel_err_y',
        'trans_var_x', 'trans_var_y', 'vel_var_x', 'vel_var_y',
    ]

    # match_data holds the extra metrics we calculate for each match.
    match_data = {
        # original metrics
        'trans_err': [],
        'vel_err': [],
        'scale_err': [],
        'orient_err': [],
        'attr_err': [],
        'conf': [],
        'ego_dist': [],
        'vel_magn': [],
        'nll_gauss_error_all': [],
        'trans_gauss_err': [],
        'vel_gauss_err': [],
        'size_gauss_err': [],
        'ci_gauss_err': ci_accumulation,
    }

    # ECE metrics
    for key in ece_util_keys:
        match_data[key] = []

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Evaluate uncertainty metrics
            nll_pos, nll_vel, nll_size = gaussian_nll_error(gt_box_match, pred_box)
            nll_total = np.concatenate([nll_pos, nll_vel], axis=-1)
            match_data['nll_gauss_error_all'].append(nll_total.mean())
            match_data['trans_gauss_err'].append(nll_pos.mean())
            match_data['vel_gauss_err'].append(nll_vel.mean())
            match_data['size_gauss_err'].append(nll_size.mean())

            # For ECE metrics, we need to calculate the errors in x and y separately.
            offset_x, offset_y = center_offset(gt_box_match, pred_box)
            offset_vel_x, offset_vel_y = velocity_offset(gt_box_match, pred_box)
            match_data['trans_err_x'].append(offset_x)
            match_data['trans_err_y'].append(offset_y)
            match_data['vel_err_x'].append(offset_vel_x)
            match_data['vel_err_y'].append(offset_vel_y)

            x_var, y_var = center_offset_var(gt_box_match, pred_box)
            vel_x_var, vel_y_var = velocity_offset_var(gt_box_match, pred_box)
            match_data['trans_var_x'].append(x_var)
            match_data['trans_var_y'].append(y_var)
            match_data['vel_var_x'].append(vel_x_var)
            match_data['vel_var_y'].append(vel_y_var)
            
            for ci in confidence_interval_values:
                match_data['ci_gauss_err'][ci].append(within_cofidence_interval(gt_box_match, pred_box, ci, distribution=distribution))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

            # For debugging only.
            match_data['ego_dist'].append(gt_box_match.ego_dist)
            match_data['vel_magn'].append(np.sqrt(np.sum(np.array(gt_box_match.velocity) ** 2)))

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(np.float64)
    fp = np.cumsum(fp).astype(np.float64)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key in ["conf"]:
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        if key in ece_util_keys:
            continue
        
        elif key == "ci_gauss_err":
            if compute_ci:
                for ci in confidence_interval_values:
                    # Same as cummean in utils but on multidim indicator data
                    tmp = np.stack(match_data[key][ci])
                    sums = np.nancumsum(tmp.astype(float), 0)
                    counts = np.cumsum(~np.isnan(tmp), 0)
                    tmp = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
                    result = np.zeros((101, tmp.shape[-1]))
                    for i in range(tmp.shape[-1]):
                        result[:, i] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1][:, i])[::-1]
                    match_data[key][ci] = result.tolist()
            else:
                # If we do not compute confidence intervals, we just set them to 0.
                for ci in confidence_interval_values:
                    match_data[key][ci] = np.zeros((101, 1)).tolist()
        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # todo: compute ECE
    # For ECE metrics, we need to calculate the errors in x and y separately, 
    # as the euclidean distance of the errors L2 is not gaussian distributed.
    # prepare dataframes for precision recall plots

    if compute_ece:
        print("Calculating precision recall dataframes...")
        prec_rec_df_x = regression_precision_recall_df(
            y_pred=match_data['trans_err_x'],
            var_pred=match_data['trans_var_x'],
            y_true=np.zeros_like(match_data['trans_err_x']),
            n_bins=num_bins_precision_recall,
        )

        prec_rec_df_y = regression_precision_recall_df(
            y_pred=match_data['trans_err_y'],
            var_pred=match_data['trans_var_y'],
            y_true=np.zeros_like(match_data['trans_err_y']),
            n_bins=num_bins_precision_recall,
        )

        prec_rec_df_vel_x = regression_precision_recall_df(
            y_pred=match_data['vel_err_x'],
            var_pred=match_data['vel_var_x'],
            y_true=np.zeros_like(match_data['vel_err_x']),
            n_bins=num_bins_precision_recall,
        )

        prec_rec_df_vel_y = regression_precision_recall_df(
            y_pred=match_data['vel_err_y'],
            var_pred=match_data['vel_var_y'],
            y_true=np.zeros_like(match_data['vel_err_y']),
            n_bins=num_bins_precision_recall,
        )

        pred_rec_dfs = {
            'trans_x': prec_rec_df_x,
            'trans_y': prec_rec_df_y,
            'vel_x': prec_rec_df_vel_x,
            'vel_y': prec_rec_df_vel_y,
        }
    else:
        pred_rec_dfs = {}

    if compute_ece:
        # prepare dataframes for calibration plots
        print("Calculating calibration dataframes...")
        calib_df_x = regression_calibration_df(
            y_pred=match_data['trans_err_x'],
            var_pred=match_data['trans_var_x'],
            y_true=np.zeros_like(match_data['trans_err_x']),
            n_bins=num_bins_calibration,
        )

        calib_df_y = regression_calibration_df(
            y_pred=match_data['trans_err_y'],
            var_pred=match_data['trans_var_y'],
            y_true=np.zeros_like(match_data['trans_err_y']),
            n_bins=num_bins_calibration,
        )

        calib_df_vel_x = regression_calibration_df(
            y_pred=match_data['vel_err_x'],
            var_pred=match_data['vel_var_x'],
            y_true=np.zeros_like(match_data['vel_err_x']),
            n_bins=num_bins_calibration,
        )

        calib_df_vel_y = regression_calibration_df(
            y_pred=match_data['vel_err_y'],
            var_pred=match_data['vel_var_y'],
            y_true=np.zeros_like(match_data['vel_err_y']),
            n_bins=num_bins_calibration,
        )

        calib_dfs = {
            'trans_x': calib_df_x,
            'trans_y': calib_df_y,
            'vel_x': calib_df_vel_x,
            'vel_y': calib_df_vel_y,
        }
    else:
        calib_dfs = {}
    

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(
        recall=rec,
        precision=prec,
        confidence=conf,
        trans_err=match_data['trans_err'],
        vel_err=match_data['vel_err'],
        scale_err=match_data['scale_err'],
        orient_err=match_data['orient_err'],
        attr_err=match_data['attr_err'],
        nll_gauss_error_all=match_data['nll_gauss_error_all'],
        trans_gauss_err=match_data['trans_gauss_err'],
        rot_gauss_err=match_data['vel_gauss_err'],
        vel_gauss_err=match_data['vel_gauss_err'],
        size_gauss_err=match_data['size_gauss_err'],
        ci_evaluation=match_data['ci_gauss_err'],
        prec_rec_dfs=pred_rec_dfs,
        calib_dfs=calib_dfs,
    )


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.
