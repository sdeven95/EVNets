from typing import Any
import numpy as np


class TsUtil:
    @staticmethod
    # get a size tensor (height, width), if only one int x is transferred, return (x, x)
    def setup_size(size: Any, error_msg="Need a tuple of length 2"):
        if isinstance(size, int):
            return size, size

        if isinstance(size, (list, tuple)) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size

    @staticmethod
    # calculate intersection area of two box(rectangle)
    def intersect(box_a, box_b):
        """Computes the intersection between box_a and box_b"""
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])  # intersection top left corner
        min_xy = np.maximum(box_a[:, :2], box_b[:2])  # intersection bott right corner
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)  # two sides of intersection
        return inter[:, 0] * inter[:, 1]  # area of intersection

    @staticmethod
    def jaccard_numpy(box_a: np.ndarray, box_b: np.ndarray):
        """
        Computes the intersection ratio of two boxes.
        Args:
            box_a (np.ndarray): Boxes of shape [Num_boxes_A, 4] (num, x1, y1, x2, y2)
            box_b (np.ndarray): Box osf shape [Num_boxes_B, 4]

        Returns:
            intersection over union scores. Shape is [box_a.shape[0], box_a.shape[1]]
        """
        inter = TsUtil.intersect(box_a, box_b)  # intersection area
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]  # box_a area
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B] # box_b area
        union = area_a + area_b - inter
        return inter / union  # [A,B]
