import torch
from typing import Optional, List, Dict

from utils.logger import Logger
from utils.tensor_utils import Tsr
from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils import Util

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model.detection import DetectionPredTuple

import os
from contextlib import redirect_stdout
import io


class COCOEvaluator(object):
    def __init__(
            self,
            iou_types: Optional[List] = ["bbox"],
            split: Optional[str] = "val",
            year: Optional[int] = 2017,
    ):
        # disable printing on console, so that pycocotools print statements are not printed on console
        Logger.disable_printing()

        root = Opt.get_by_cfg_path(Entry.Dataset, "root_val")
        ann_file = os.path.join(
            root, "annotations/instances_{}{}.json".format(split, year)
        )
        self.coco_gt = COCO(ann_file)

        coco_categories = sorted(self.coco_gt.getCatIds())
        coco_id_to_contiguous_id = {
            coco_id: i + 1 for i, coco_id in enumerate(coco_categories)
        }
        self.contiguous_id_to_coco_id = {
            v: k for k, v in coco_id_to_contiguous_id.items()
        }

        self.iou_types = iou_types
        self.coco_results = {iou_type: [] for iou_type in iou_types}

        # enable printing, to enable cvnets log printing
        Logger.enable_printing()

    # prepare predictions and image batch
    def prepare_predictions(self, predictions: Dict, targets: Dict):
        if not (
                isinstance(predictions, Dict)
                and ({"detections"} <= set(list(predictions.keys())))
        ):
            Logger.error(
                "For coco evaluation during training, the output from the model should be a dictionary "
                "and should contain the results in a key called detections"
            )

        detections = predictions["detections"]
        if isinstance(detections, List) and isinstance(
                detections[0], DetectionPredTuple
        ):
            self.prepare_cache_results(
                detection_results=detections,
                image_ids=targets["image_id"],
                image_widths=targets["image_width"],
                image_heights=targets["image_height"],
                iou_type="bbox",
            )
        elif isinstance(detections, DetectionPredTuple):
            self.prepare_cache_results(
                detection_results=[detections],  # create a list
                image_ids=targets["image_id"],
                image_widths=targets["image_width"],
                image_heights=targets["image_height"],
                iou_type="bbox",
            )
        else:
            Logger.error(
                "For coco evaluation during training, the results should be stored as a List of DetectionPredTuple"
            )

    # combine images and predictions to get results
    # return list with the length batch_size * num_classes, the element(one for a class of an image)
    # is dict { image_id, same_class_id_list, box_list, score_list }
    def prepare_cache_results(
            self, detection_results: List, image_ids, image_widths, image_heights, iou_type
    ):
        batch_results = []  # finally, [batch_size * num_classes], element is dict
        # iterate for each image
        for detection_result, img_id, img_w, img_h in zip(
                detection_results, image_ids, image_widths, image_heights
        ):
            label = detection_result.labels  # one image, all classes [num_classes, num_priors]

            if label.numel() == 0:
                # no detections
                continue
            box = detection_result.boxes  # [num_classes, num_priors]
            score = detection_result.scores  # [num_classes, num_priors]

            img_id, img_w, img_h = img_id.item(), img_w.item(), img_h.item()

            # ratio => length
            box[..., 0::2] = torch.clip(box[..., 0::2] * img_w, min=0, max=img_w)
            box[..., 1::2] = torch.clip(box[..., 1::2] * img_h, min=0, max=img_h)

            # convert box from xyxy to xywh format
            box[..., 2] = box[..., 2] - box[..., 0]
            box[..., 3] = box[..., 3] - box[..., 1]

            box = box.cpu().numpy()
            label = label.cpu().numpy()
            score = score.cpu().numpy()

            # register num_classes objects for one image
            batch_results.extend(
                [
                    # for each class
                    {
                        "image_id": img_id,  # image_id
                        "category_id": self.contiguous_id_to_coco_id[label[class_id]],  # [class_id, class_id, ...]
                        "bbox": box[class_id].tolist(),  # [num_priors]  box list for current class
                        "score": score[class_id],  # [num_priors] probability list for current class
                    }
                    for class_id in range(box.shape[0])  # num_classes
                    if label[class_id] > 0  # ignore background class
                ]
            )

        # add to results
        self.coco_results[iou_type].extend(batch_results)

    # if distributed, retrieve results from each process
    def gather_coco_results(self):

        # synchronize results across different devices
        for iou_type, coco_results in self.coco_results.items():
            # agg_coco_results as List[List].
            # The outer list is for processes and inner list is for coco_results in the process
            if Util.is_distributed():
                agg_coco_results = Tsr.all_gather_list(coco_results)

                merged_coco_results = []
                # filter the duplicates
                for (p_coco_results) in agg_coco_results:  # retrieve results from each process
                    merged_coco_results.extend(p_coco_results)
            else:
                merged_coco_results = coco_results

            self.coco_results[iou_type] = merged_coco_results

    # evaluate predictions
    def summarize_coco_results(self) -> Dict:

        Logger.disable_printing()

        stats_map = dict()
        for iou_type, coco_results in self.coco_results.items():
            if len(coco_results) < 1:
                # during initial epochs, we may not have any sample results, so we can skip this part
                map_val = 0.0
            else:
                try:
                    with redirect_stdout(io.StringIO()):
                        coco_dt = COCO.loadRes(self.coco_gt, coco_results)

                    coco_eval = COCOeval(
                        cocoGt=self.coco_gt, cocoDt=coco_dt, iouType=iou_type
                    )
                    coco_eval.evaluate()
                    coco_eval.accumulate()

                    Logger.log(f"Results for iouType={iou_type}")
                    coco_eval.summarize()
                    map_val = coco_eval.stats[0].item()
                except Exception:
                    map_val = 0.0
            stats_map[iou_type] = map_val * 100

        Logger.enable_printing()
        return stats_map
