# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from nuscenes.eval.common.data_classes import MetricData, EvalBox
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES, TP_METRICS


class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: float,
                 mean_ap_weight: int,
                 distribution: str = 'gaussian'):

        assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.distribution = distribution
        self.class_names = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight,
            'distribution': self.distribution
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """
        return cls(content['class_range'],
                   content['dist_fcn'],
                   content['dist_ths'],
                   content['dist_th_tp'],
                   content['min_recall'],
                   content['min_precision'],
                   content['max_boxes_per_sample'],
                   content['mean_ap_weight'],
                   content['distribution'] if 'distribution' in content else 'gaussian')

    @property
    def dist_fcn_callable(self):
        """ Return the distance function corresponding to the dist_fcn string. """
        if self.dist_fcn == 'center_distance':
            return center_distance
        else:
            raise Exception('Error: Unknown distance function %s!' % self.dist_fcn)


class DetectionMetricData(MetricData):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(
        self,
        recall: np.array,
        precision: np.array,
        confidence: np.array,
        trans_err: np.array,
        vel_err: np.array,
        scale_err: np.array,
        orient_err: np.array,
        attr_err: np.array,
        nll_gauss_error_all: np.array,
        trans_gauss_err: np.array,
        rot_gauss_err: np.array,
        vel_gauss_err: np.array,
        size_gauss_err: np.array,
        ci_evaluation: dict,
        prec_rec_dfs: Dict[str, pd.DataFrame] | None = None,
        calib_dfs: Dict[str, pd.DataFrame] | None = None,
    ):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem
        assert len(nll_gauss_error_all) == self.nelem
        assert len(trans_gauss_err) == self.nelem
        # assert len(rot_gauss_err) == self.nelem TODO: add rot_gauss_err to the metrics
        assert len(vel_gauss_err) == self.nelem
        assert len(size_gauss_err) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err
        self.nll_gauss_error_all = nll_gauss_error_all
        self.trans_gauss_err = trans_gauss_err
        self.rot_gauss_err = rot_gauss_err
        self.vel_gauss_err = vel_gauss_err
        self.size_gauss_err = size_gauss_err
        self.ci_evaluation = ci_evaluation
        self.prec_rec_dfs = prec_rec_dfs if prec_rec_dfs is not None else {}
        self.calib_dfs = calib_dfs if calib_dfs is not None else {}

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
            'nll_gauss_error_all': self.nll_gauss_error_all.tolist(),
            'trans_gauss_err': self.trans_gauss_err.tolist(),
            'rot_gauss_err': self.rot_gauss_err.tolist(),
            'vel_gauss_err': self.vel_gauss_err.tolist(),
            'size_gauss_err': self.size_gauss_err.tolist(),
            'ci_evaluation': self.ci_evaluation,
            'prec_rec_dfs': {k: v.to_dict(orient='split') for k, v in self.prec_rec_dfs.items()},
            'calib_dfs': {k: v.to_dict(orient='split') for k, v in self.calib_dfs.items()}
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """

        def df_dict_to_dfs(d):
            return {k: pd.DataFrame(**v) for k, v in d.items()}
    
        return cls(
            recall=np.array(content['recall']),
            precision=np.array(content['precision']),
            confidence=np.array(content['confidence']),
            trans_err=np.array(content['trans_err']),
            vel_err=np.array(content['vel_err']),
            scale_err=np.array(content['scale_err']),
            orient_err=np.array(content['orient_err']),
            attr_err=np.array(content['attr_err']),
            nll_gauss_error_all=np.array(content['nll_gauss_error_all']),
            trans_gauss_err=np.array(content['trans_gauss_err']),
            rot_gauss_err=np.array(content['rot_gauss_err']),
            vel_gauss_err=np.array(content['vel_gauss_err']),
            size_gauss_err=np.array(content['size_gauss_err']),
            ci_evaluation=content['ci_evaluation'],
            prec_rec_dfs=df_dict_to_dfs(content.get('prec_rec_dfs', {})),
            calib_dfs=df_dict_to_dfs(content.get('calib_dfs', {})),
        )
    

    @classmethod
    def no_predictions(cls):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(
            recall=np.linspace(0, 1, cls.nelem),
            precision=np.zeros(cls.nelem),
            confidence=np.zeros(cls.nelem),
            trans_err=np.ones(cls.nelem),
            vel_err=np.ones(cls.nelem),
            scale_err=np.ones(cls.nelem),
            orient_err=np.ones(cls.nelem),
            attr_err=np.ones(cls.nelem),
            nll_gauss_error_all=np.ones(cls.nelem),
            trans_gauss_err=np.ones(cls.nelem),
            vel_gauss_err=np.ones(cls.nelem),
            rot_gauss_err=np.ones(cls.nelem),
            size_gauss_err=np.ones(cls.nelem),
            ci_evaluation={},
            prec_rec_dfs={},
            calib_dfs={},
        )

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        return cls(
            recall=np.linspace(0, 1, cls.nelem),
            precision=np.random.random(cls.nelem),
            confidence=np.linspace(0, 1, cls.nelem)[::-1],
            trans_err=np.random.random(cls.nelem),
            vel_err=np.random.random(cls.nelem),
            scale_err=np.random.random(cls.nelem),
            orient_err=np.random.random(cls.nelem),
            attr_err=np.random.random(cls.nelem),
            nll_gauss_error_all=np.random.random(cls.nelem),
            trans_gauss_err=np.random.random(cls.nelem),
            rot_gauss_err=np.random.random(cls.nelem),
            vel_gauss_err=np.random.random(cls.nelem),
            size_gauss_err=np.random.random(cls.nelem),
            ci_evaluation={},
            prec_rec_dfs={},
            calib_dfs={},
        )


class DetectionMetrics:
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    def __init__(self, cfg: DetectionConfig):

        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self._label_ece = defaultdict(lambda: defaultdict(float))
        self._target_wise_ece = defaultdict(float)
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return self._label_aps[detection_name][dist_th]

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return self._label_tp_errors[detection_name][metric_name]

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = eval_time

    def add_label_ece(self, detection_name: str, metric_name: str, ece: float):
        self._label_ece[detection_name][metric_name] = ece

    def get_label_ece(self, detection_name: str, metric_name: str) -> float:
        return self._label_ece[detection_name][metric_name]
    
    def add_target_wise_ece(self, target: str, ece: float) -> None:
        self._target_wise_ece[target] = ece

    def get_target_wise_ece(self, target: str) -> float:
        return self._target_wise_ece[target]

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        """ Calculates the mean over distance thresholds for each label. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_aps.items()}
    
    @property
    def mean_label_ece(self) -> Dict[str, float]:
        """ Calculates the mean ECE over all labels and metrics. """
        return {class_name: np.mean(list(d.values())) for class_name, d in self._label_ece.items()}

    @property
    def mean_target_wise_ece(self) -> float:
        """ Calculates the mean ECE over all targets. """
        return float(np.mean(list(self._target_wise_ece.values())))

    @property
    def mean_ap(self) -> float:
        """ Calculates the mean AP by averaging over distance thresholds and classes. """
        return float(np.mean(list(self.mean_dist_aps.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        """ Calculates the mean true positive error across all classes for each metric. """
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))

            errors[metric_name] = float(np.nanmean(class_errors))

        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            scores[metric_name] = score

        return scores

    @property
    def nd_score(self) -> float:
        """
        Compute the nuScenes detection score (NDS, weighted sum of the individual scores).
        :return: The NDS.
        """
        scores_to_ignore = ['nll_gauss_error_all', 'trans_gauss_err', 'vel_gauss_err', 'rot_gauss_err', 'size_gauss_err']
        relevant_num_keys = 0
        # Summarize.
        total = float(self.cfg.mean_ap_weight * self.mean_ap)
        for score_name, val in self.tp_scores.items():
            if score_name not in scores_to_ignore:
                relevant_num_keys +=1
                total += val

        # Normalize.
        total = total / float(self.cfg.mean_ap_weight + relevant_num_keys)

        return total

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            'nd_score': self.nd_score,
            'label_ece': self._label_ece,
            'mean_label_ece': self.mean_label_ece,
            'target_wise_ece': self._target_wise_ece,
            'mean_ece': self.mean_target_wise_ece,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize()
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = DetectionConfig.deserialize(content['cfg'])

        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        for detection_name, label_ece in content['label_ece'].items():
            for metric_name, ece in label_ece.items():
                metrics.add_label_ece(detection_name=detection_name, metric_name=metric_name, ece=float(ece))

        for target, ece in content.get('target_wise_ece', {}).items():
            metrics.add_target_wise_ece(target=target, ece=float(ece))

        return metrics

    def __eq__(self, other):
        eq = True
        eq = eq and self._label_aps == other._label_aps
        eq = eq and self._label_tp_errors == other._label_tp_errors
        eq = eq and self._label_ece == other._label_ece
        eq = eq and self._target_wise_ece == other._target_wise_ece
        eq = eq and self.eval_time == other.eval_time
        eq = eq and self.cfg == other.cfg

        return eq


class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_dist: float = 0.0,  # Distance to ego vehicle in meters.
        num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        detection_name: str = 'car',  # The class name used in the detection challenge.
        detection_score: float = -1.0,  # GT samples do not have a score.
        attribute_name: str = '', # Box attribute. Each box can have at most 1 attribute.
        uncertainty: List[float] = [],
    ): # List of uncertainty values 

        super().__init__(sample_token, translation, size, rotation, velocity, num_pts)

        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.ego_dist = ego_dist
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name
        self.uncertainty = uncertainty

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_dist == other.ego_dist and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_dist': self.ego_dist,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name,
            'uncertainty': self.uncertainty
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_dist=0.0 if 'ego_dist' not in content else float(content['ego_dist']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name'],
            uncertainty=content['uncertainty'],
        )


class DetectionMetricDataList:
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """

    def __init__(self):
        self.md = {}

    def __getitem__(self, key):
        return self.md[key]

    def __eq__(self, other):
        eq = True
        for key in self.md.keys():
            eq = eq and self[key] == other[key]
        return eq

    def get_class_data(self, detection_name: str) -> List[Tuple[DetectionMetricData, float]]:
        """ Get all the MetricData entries for a certain detection_name. """
        return [(md, dist_th) for (name, dist_th), md in self.md.items() if name == detection_name]

    def get_dist_data(self, dist_th: float) -> List[Tuple[DetectionMetricData, str]]:
        """ Get all the MetricData entries for a certain match_distance. """
        return [(md, detection_name) for (detection_name, dist), md in self.md.items() if dist == dist_th]

    def set(self, detection_name: str, match_distance: float, data: DetectionMetricData):
        """ Sets the MetricData entry for a certain detection_name and match_distance. """
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

    @classmethod
    def deserialize(cls, content: dict):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), DetectionMetricData.deserialize(md))
        return mdl
