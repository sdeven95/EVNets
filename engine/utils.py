import gc
import os
from contextlib import redirect_stdout
import io
import glob

from typing import List, Optional

from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils.logger import Logger
from utils.tensor_utils import Tsr
from utils import Util

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_semseg_eval


class EngineUtil:

	@staticmethod
	def print_summary(
			model,
			eval_dataloader=None,
			train_dataloader=None,
			val_dataloader=None,
			criteria=None,
			optimizer=None,
			scheduler=None):

		if not Util.is_master_node():
			return

		if eval_dataloader:
			Logger.log("Evaluation dataset details: ")
			print(eval_dataloader.dataset)
			Logger.log("Evaluation sampler details:")
			print(eval_dataloader.batch_sampler)
			Logger.log("Evaluation collate function details:")
			print(f"\t-------- {eval_dataloader.collate_fn.__name__} ----------")

		if train_dataloader:
			Logger.log("Train dataset details: ")
			print(train_dataloader.dataset)
			Logger.log("Train sampler details:")
			print(train_dataloader.batch_sampler)
			Logger.log("Train collate function details:")
			print(f"\t-------- {train_dataloader.collate_fn.__name__} ----------")

		if val_dataloader:
			Logger.log("Validation dataset details: ")
			print(val_dataloader.dataset)
			Logger.log("Validation sampler details:")
			print(val_dataloader.batch_sampler)
			Logger.log("Validation collate function details:")
			print(f"\t-------- {val_dataloader.collate_fn.__name__} ----------")

		Logger.log(Logger.color_text("Model"))
		print(model)

		dev = Opt.get(f"{Entry.Common}.Dev.device")
		# inp_tensor = Tsr.create_rand_tensor(device=dev)
		inp_tensor = model.generate_input().to(dev)
		if hasattr(model, "module"):
			model.module.profile_model(inp_tensor)
		else:
			model.profile_model(inp_tensor)

		del inp_tensor

		if criteria:
			Logger.log(Logger.color_text("Loss Function"))
			print(criteria)
		if optimizer:
			Logger.log(Logger.color_text("Optimizer"))
			print(optimizer)
		if scheduler:
			Logger.log(Logger.color_text("Scheduler"))
			print(scheduler)

		gc.collect()

	@staticmethod
	def coco_evaluation(
			predictions: List[list],
			split: Optional[str] = "val",
			year: Optional[int] = 2017,
			iou_type: Optional[str] = "bbox"
	) -> None:
		coco_results = []
		root = Opt.get_by_cfg_path(Entry.Dataset, "root_val")
		ann_file = os.path.join(root, f"annotations/instance_{split}{year}.json")
		coco = COCO(ann_file)

		coco_categories = sorted(coco.getCatIds())
		coco_id_to_contiguous_id = {
			coco_id: i + 1 for i, coco_id in enumerate(coco_categories)
		}
		contiguous_id_to_coco_id = {v: k for k, v in coco_id_to_contiguous_id.items()}

		for i, (image_id, boxes, labels, scores) in enumerate(predictions):
			if labels.shape[0] == 0:
				continue

			boxes = boxes.tolist()
			labels = labels.tolist()
			scores = scores.tolist()
			coco_results.extend(
				[
					{
						"image_id": image_id,
						"category_id": contiguous_id_to_coco_id[labels[k]],
						"bbox": [
							box[0],
							box[1],
							box[2] - box[0],
							box[3] - box[1],
						],  # to xywh format
						"score": scores[k],
					}
					for k, box in enumerate(boxes)
				]
			)

		assert len(coco_results) > 0, f"Cannot compute COCO stats. Please check the predictions"

		with redirect_stdout(io.StringIO()):
			coco_dt = COCO.loadRes(coco, coco_results)

		# run COCO evaluation
		coco_eval = COCOeval(coco, coco_dt, iou_type)
		coco_eval.evaluate()
		coco_eval.accumulate()
		coco_eval.summarize()

	@staticmethod
	def eval_cityscapes(pred_dir: str, gt_dir: str) -> None:
		"""Utility to evaluate on cityscapes dataset"""
		cityscapes_semseg_eval.args.predictionPath = pred_dir
		cityscapes_semseg_eval.args.predictionWalk = None
		cityscapes_semseg_eval.args.JSONOutput = False
		cityscapes_semseg_eval.args.colorized = False

		gt_img_list = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
		if len(gt_img_list) == 0:
			Logger.error("Cannot find ground truth images at: {}".format(gt_dir))

		pred_img_list = []
		for gt in gt_img_list:
			pred_img_list.append(
				cityscapes_semseg_eval.getPrediction(cityscapes_semseg_eval.args, gt)
			)

		results = cityscapes_semseg_eval.evaluateImgLists(
			pred_img_list, gt_img_list, cityscapes_semseg_eval.args
		)

		Logger.info("Evaluation results summary")
		eval_res_str = "\n\t IoU_cls: {:.2f} \n\t iIOU_cls: {:.2f} \n\t IoU_cat: {:.2f} \n\t iIOU_cat: {:.2f}".format(
			100.0 * results["averageScoreClasses"],
			100.0 * results["averageScoreInstClasses"],
			100.0 * results["averageScoreCategories"],
			100.0 * results["averageScoreInstCategories"],
		)
		print(eval_res_str)
