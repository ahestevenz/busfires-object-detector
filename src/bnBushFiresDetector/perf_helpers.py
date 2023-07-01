from typing import Tuple, Dict, List
from loguru import logger as logging
import numpy as np


def intersection_over_union(pred_box: List,
                            true_box: List) -> List:
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis=1)
    # 1. Calculate coordinates of overlap area between boxes
    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)
    # 2. Calculates area of true and predicted boxes
    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
    # 3. Calculates overlap area and union area.
    overlap_area = np.maximum(
        (xmax_overlap - xmin_overlap), 0) * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area
    # 4. Defines a smoothing factor to prevent division by 0
    smoothing_factor = 1e-10
    # 5. Updates iou score
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    return iou
