import glob
import os
import torch
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from PIL import Image

from common import Common, Dev, DDP

from utils.logger import Logger
from utils import Util
from utils.file_utils import File
from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils.tensor_utils import Tsr
from utils.opt_setup_utils import OptSetup
from utils.visualization_utils import Visualization

from data.constants import DataConstants
from .utils import EngineUtil

from data.sampler.utils import SplUtil
from model.detection.ssd import DetectionPredTuple
from data.dataset.detection.coco_base import COCODetection

object_names = COCODetection.class_names()


def predict_and_save(
		input_tensor: Tensor,
		model: nn.Module,
		input_np: Optional[np.ndarray] = None,
		device: Optional = torch.device("cpu"),
		is_coco_evaluation: Optional[bool] = False,
		file_name: Optional[str] = None,
		output_stride: Optional[int] = 32,
		orig_h: Optional[int] = None,
		orig_w: Optional[int] = None,
):
	if input_np is None and not is_coco_evaluation:
		# convert to numpy and remove the batch dimension
		input_np = Tsr.to_numpy(input_tensor).squeeze(0)

	curr_height, curr_width = input_tensor.shape[2:]

	# check if dimensions are multiple of out_stride, otherwise, we get dimension mismatch errors
	# if not, then resize them
	new_h = (curr_height // output_stride) * output_stride
	new_w = (curr_width // output_stride) * output_stride

	if new_h != curr_height or new_w != curr_width:
		# resize the input image, so that we don't get dimension mismatch errors in the forward pass
		input_tensor = F.interpolate(
			input=input_tensor,
			size=(new_h, new_w),
			mode="bilinear",
			align_corners=False,
		)

	# move data to device
	input_tensor = input_tensor.to(device)

	common = Common()
	with autocast(enabled=common.mixed_precision):
		# prediction
		prediction: DetectionPredTuple = model.predict(input_tensor)

	# convert tensors to numpy
	boxes = prediction.boxes.cpu().numpy()
	labels = prediction.labels.cpu().numpy()
	scores = prediction.scores.cpu().numpy()

	if orig_w is None:
		assert orig_h is None
		orig_h, orig_w = input_np.shape[:2]
	elif orig_h is None:
		assert orig_w is None
		orig_h, orig_w = input_np.shape[:2]

	assert orig_h is not None and orig_w is not None
	boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2] * orig_w)
	boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2] * orig_h)

	if is_coco_evaluation:
		return boxes, labels, scores

	detection_res_file_name = None
	if file_name is not None:
		file_name = file_name.split(os.sep)[-1].split('.')[0] + ".jpg"
		res_dir = f"{common.exp_loc}/detection_results"
		if not os.path.isdir(res_dir):
			os.makedirs(res_dir, exist_ok=True)
		detection_res_file_name = f"{res_dir}/{file_name}"

	Visualization.draw_bounding_boxes(
		image=input_np,
		boxes=boxes,
		labels=labels,
		scores=scores,
		object_names=object_names,
		is_bgr_format=True,
		save_path=detection_res_file_name,
	)


def predict_labeled_dataset():
	device = Util.get_device()
	eval_loader = Entry.get_entity(Entry.DataLoader, entry_name="eval")

	model = Entry.get_entity(Entry.DetectionModel)
	model.eval()
	model = model.to(device)
	EngineUtil.print_summary(model=model)

	assert not model.training, "Model is in training mode. Switching to evaluation mode."

	with torch.no_grad():
		predictions = []
		for img_idx, batch in tqdm(enumerate(eval_loader)):
			input_img, target_label = batch["image"], batch["label"]
			input_img = input_img["image"]
			batch_size = _get_batch_size(input_img)
			assert batch_size == 1, "We recommend to run detection evaluation with a batch size of 1."

			orig_w = target_label["image_width"].item()
			orig_h = target_label["image_height"].item()
			image_id = target_label["image_id"].item()

			boxes, labels, scores = predict_and_save(
				input_tensor=input_img,
				model=model,
				device=device,
				is_coco_evaluation=True,
				orig_w=orig_w,
				orig_h=orig_h,
			)

			predictions.append([image_id, boxes, labels, scores])

		EngineUtil.coco_evaluation(predictions=predictions)


def _get_batch_size(x):
	if isinstance(x, torch.Tensor):
		return x.shape[0]
	elif isinstance(x, Dict):
		return x["image"].shape[0]


def read_and_process_image(image_file_name: str):
	input_img = Image.open(image_file_name).convert("RGB")
	input_np = np.array(input_img)
	orig_w, orig_h = input_img.size

	# Resize the image to the resolution that detector supports
	res_h, res_w = SplUtil.image_size_from_opts()
	input_img = F_vision.resize(
		img=input_img,
		size=[res_h, res_w],
		interpolation=F_vision.InterpolationMode.BILINEAR
	)
	input_tensor = F_vision.pil_to_tensor(input_img)
	input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
	return input_tensor, input_np, orig_h, orig_w


def predict_image(image_file_path):
	assert os.path.isfile(image_file_path), f"Image file does not exist at : {image_file_path}"
	input_tensor, input_img_copy, orig_h, orig_w = read_and_process_image(image_file_name=image_file_path)
	image_file_name = image_file_path.split(os.sep)[-1]

	device = Util.get_device()
	model = Entry.get_entity(Entry.DetectionModel)
	model.eval()
	model = model.to(device)
	EngineUtil.print_summary(model=model)

	assert not model.training, "Model is training mode. Switch to evaluation mode."

	with torch.no_grad():
		predict_and_save(
			input_tensor=input_tensor,
			input_np=input_img_copy,
			file_name=image_file_name,
			model=model,
			device=device,
			orig_h=orig_h,
			orig_w=orig_w,
		)


def predict_images_in_folder():
	img_folder_path = OptSetup.get_by_cfg_path(Entry.Dataset, "evaluation_image_path")
	assert img_folder_path is not None, "Detection evaluation image folder is not passed."
	assert os.path.isdir(img_folder_path), f"Detection evaluation image folder does not exist at: {img_folder_path}"

	img_files = []
	for e in DataConstants.SUPPORTED_IMAGE_EXTNS:
		img_files_with_extn = glob.glob(f"{img_folder_path}/*{e}")
		if len(img_files_with_extn) > 0 and isinstance(img_files_with_extn, list):
			img_files.extend(img_files_with_extn)

	assert len(img_files) > 0, f"Number of image files found at {img_folder_path}: {len(img_files)}"

	device = Util.get_device()
	model = Entry.get_entity(Entry.DetectionModel)
	model.eval()
	model = model.to(device)
	EngineUtil.print_summary(model=model)

	assert not model.training, "Model is training mode. Switch to evaluation mode."

	with torch.no_grad():
		for img_idx, image_fname in enumerate(img_files):
			input_tensor, input_np, orig_h, orig_w = read_and_process_image(image_file_name=image_fname)
			image_fname = image_fname.split(os.sep)[-1]

			predict_and_save(
				input_tensor=input_tensor,
				input_np=input_np,
				file_name=image_fname,
				model=model,
				device=device,
				orig_h=orig_h,
				orig_w=orig_w,
			)


def main_detection_evaluation():
	OptSetup.get_arguments()

	common = Common()
	dev = Dev()
	Util.device_setup(common, dev)

	common.exp_loc = f"{common.result_loc}/{common.run_label}"
	Logger.log(f"Results (if any) will be stored here: {common.exp_loc}")
	File.create_directories(dir_path=common.exp_loc)

	if dev.num_gpus < 2:
		norm_name = OptSetup.get_target_subgroup_name(Entry.Norm)
		if norm_name.find("sync") > -1:
			OptSetup.set_target_subgroup_name(Entry.Norm, norm_name.replace("Sync", ""))

	ddp = DDP()
	# we disable the DDP setting for evaluation tasks
	ddp.use_distributed = False

	# No of data workers = no of CPUs (if not specified or -1)
	data_workers = OptSetup.get_by_cfg_path(Entry.Dataset, "workers")
	if data_workers is None or data_workers < 0:
		OptSetup.set_by_cfg_path(Entry.Dataset, "workers", dev.num_cpus)

	# We are not performing any operation like resizing and cropping on images
	# Because image dimensions are different, we process 1 sample at a time.
	OptSetup.set_by_cfg_path(Entry.Sampler, "train_batch_size", 1)
	OptSetup.set_by_cfg_path(Entry.Sampler, "val_batch_size", 1)
	OptSetup.set_by_cfg_path(Entry.Sampler, "eval_batch_size", 1)

	eval_mode = OptSetup.get_by_cfg_path(Entry.Dataset, "evaluation_mode")
	assert eval_mode in ["single_image", "image_folder", "validation_set"], \
		f"Supported modes are single_image, image_folder, and validation_set. Got: {eval_mode}"

	num_classes = OptSetup.get_by_cfg_path(Entry.DetectionModel, "n_classes")
	assert num_classes is not None and num_classes > 0

	if eval_mode == "single_image":
		image_file_path = OptSetup.get_by_cfg_path(Entry.Dataset, "evaluation_image_path")
		predict_image(image_file_path=image_file_path)
	elif eval_mode == "image_folder":
		predict_images_in_folder()
	elif eval_mode == "validation_set":
		# evaluate and compute stats for labeled image dataset
		# This is useful for generating results for validation set and compute quantitative results
		predict_labeled_dataset()


if __name__ == "__main__":
	main_detection_evaluation()
