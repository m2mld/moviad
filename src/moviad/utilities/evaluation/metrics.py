from __future__ import annotations
from enum import Enum
from abc import ABC, abstractmethod
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)
import numpy as np
from skimage.measure import label, regionprops


class MetricLvl(Enum):
    """The metrics can be computed on image or pixel level."""

    IMAGE = "img"
    PIXEL = "pxl"


class Metric(ABC):
    def __init__(self, level: MetricLvl):
        """
        Args:
            level (MetricLvl): The level of the metric (e.g. image, pixel).
        """
        self.level = level

    @property
    @abstractmethod
    def name(self): ...

    @abstractmethod
    def compute(self, gt, pred): ...


class SimpleMetric(Metric):
    def __init__(self, name, function, level: MetricLvl):
        """
        Args:
            level (MetricLvl): The level of the metric (e.g. image, pixel).
            compute (callable): A callable that takes two arguments (gt, pred) and returns a float.
        """
        self.level = level
        self.base_name = name
        self.compute = function

    def compute(self, gt, pred):
        if self.level == MetricLvl.PIXEL:
            pred, gt = pred.flatten(), gt.flatten()
        return self.compute(gt, pred)

    @property
    def name(self):
        return f"{self.level.value}_{self.base_name}"


class F1(Metric):
    @property
    def name(self):
        return f"{self.level.value}_f1"

    def compute(self, gt, pred):
        if self.level == MetricLvl.PIXEL:
            pred, gt = pred.flatten(), gt.flatten()
        precision, recall, _ = precision_recall_curve(gt, pred)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return np.max(f1)


class RocAuc(Metric):
    """Receiver operating characteristic AUC score."""

    @property
    def name(self):
        return f"{self.level.value}_roc_auc"

    def compute(self, gt, pred):
        """
        Args:
            gt (np.ndarray): Ground truth labels, either as a binary mask or list of labels.
            pred (np.ndarray): Predicted scores, either as a mask or list of scores.

        Returns:
            float: ROC AUC score for the desired metric level.
        """
        if self.level == MetricLvl.PIXEL:
            pred, gt = pred.flatten(), gt.flatten()
        return roc_auc_score(gt, pred)


class RocCurve(Metric):
    """
    Receiver operating characteristic curve, false positive and true positive rate.
    """

    @property
    def name(self):
        return f"{self.level.value}_fpr_tpr"

    def compute(self, gt, pred):
        """
        Args:
            gt (np.ndarray): Numpy array of ground truth labels.
            pred (np.ndarray): Numpy array of predicted scores.

        Returns:
            tuple: A tuple containing:
            - fpr (np.ndarray): False positive rate.
            - tpr (np.ndarray): True positive rate.
        """
        if self.level == MetricLvl.PIXEL:
            pred, gt = pred.flatten(), gt.flatten()
        fpr, tpr, _ = roc_curve(gt, pred)
        return fpr, tpr


class AvgPrec(Metric):
    """Average Precision metric."""

    @property
    def name(self):
        return f"{self.level.value}_avg_prec"

    def compute(self, gt, pred):
        """
        Args:
            gt (array-like): Ground truth binary labels.
            pred (array-like): Predicted scores or probabilities.

        Returns:
            float: The computed average precision score.
        """
        if self.level == MetricLvl.PIXEL:
            pred, gt = pred.flatten(), gt.flatten()
        return average_precision_score(gt, pred)


class ProAuc(Metric):
    """Per-Region-Overlap Curve Area Under Curve (PRO AUC) metric."""

    def __init__(self, level: MetricLvl):
        """
        Args:
            level (MetricLvl): The level of the metric (e.g. image, pixel).
        """
        if level != MetricLvl.PIXEL:
            raise ValueError(
                "ProAuc metric can only be computed on pixel level. "
                f"Got {level} instead."
            )
        super().__init__(level)

    @property
    def name(self):
        return f"{MetricLvl.PIXEL.value}_au_pro"

    @staticmethod
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    def compute(self, gt, pred):
        """
        Args:
            gt (np.ndarray): Numpy array of ground truth masks.
            pred (np.ndarray): Numpy array of predicted masks.
        Returns:
            float: PRO AUC score.
        """
        # remove the channel dimension
        gt = np.squeeze(gt, axis=1)

        if pred.shape[1] == 1:
            pred = np.squeeze(pred, axis=1)
        
        if pred.shape != gt.shape:
            raise ValueError("Ground truth and prediction must have the same shape.")

        gt[gt <= 0.5] = 0
        gt[gt > 0.5] = 1
        gt = gt.astype(np.bool_)

        max_step = 200
        expect_fpr = 0.3

        # set the max and min scores and the delta step
        max_th = pred.max()
        min_th = pred.min()
        delta = (max_th - min_th) / max_step

        pros_mean = []
        threds = []
        fprs = []

        binary_score_maps = np.zeros_like(pred, dtype=np.bool_)

        for step in range(max_step):
            thred = max_th - step * delta

            # segment the scores with different thresholds
            binary_score_maps[pred <= thred] = 0
            binary_score_maps[pred > thred] = 1

            pro = []
            for i in range(len(binary_score_maps)):

                # label the regions in the ground truth
                label_map = label(gt[i], connectivity=2)

                # calculate some properties for every corresponding region
                props = regionprops(label_map, binary_score_maps[i])

                # calculate the per-regione overlap
                for prop in props:
                    pro.append(prop.intensity_image.sum() / prop.area)

            # append the per-region overlap
            pros_mean.append(np.array(pro).mean())

            # calculate the false positive rate
            gt_neg = ~gt
            fpr = np.logical_and(gt_neg, binary_score_maps).sum() / gt_neg.sum()
            fprs.append(fpr)
            threds.append(thred)

        threds = np.array(threds)
        pros_mean = np.array(pros_mean)
        fprs = np.array(fprs)

        # select the case when the false positive rates are under the expected fpr
        idx = fprs <= expect_fpr

        fprs_selected = fprs[idx]
        fprs_selected = self.rescale(fprs_selected)
        pros_mean_selected = self.rescale(pros_mean[idx])
        per_pixel_roc_auc = auc(fprs_selected, pros_mean_selected)

        return per_pixel_roc_auc


class IoU(Metric):
    """Intersection over Union of the anomalous class at a fixed threshold, pooled over all pixels."""

    def __init__(self, level: MetricLvl = MetricLvl.PIXEL, threshold: float = 0.5):
        super().__init__(level)
        self.threshold = threshold

    @property
    def name(self):
        return f"{self.level.value}_iou"

    def compute(self, gt, pred):
        gt_binary = np.asarray(gt).flatten() > 0
        pred_binary = np.asarray(pred).flatten() >= self.threshold
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        return float(intersection / union) if union > 0 else 1.0


class Dice(Metric):
    """Dice coefficient (F1 at a fixed threshold) of the anomalous class, pooled over all pixels."""

    def __init__(self, level: MetricLvl = MetricLvl.PIXEL, threshold: float = 0.5):
        super().__init__(level)
        self.threshold = threshold

    @property
    def name(self):
        return f"{self.level.value}_dice"

    def compute(self, gt, pred):
        gt_binary = np.asarray(gt).flatten() > 0
        pred_binary = np.asarray(pred).flatten() >= self.threshold
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        total = gt_binary.sum() + pred_binary.sum()
        return float(2 * intersection / total) if total > 0 else 1.0
