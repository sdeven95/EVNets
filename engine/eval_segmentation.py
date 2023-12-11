import os
import copy
import glob
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
from torch.cuda.amp import autocast

from PIL import Image

from typing import Optional, List, Tuple

from common import Common
from metrics.confusion_matrix import ConfusionMatrix
from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils.logger import Logger
from utils.visualization_utils import Visualization, Colormap
from .utils import EngineUtil
from utils import Util

from model.segmentation import BaseSegmentation

from data.sampler.utils import SplUtil
from data.constants import DataConstants

"""
Notes:
	1) We have separate scripts for evaluating segmentation models because the size of input images varies and 
	we do not want to apply any resizing operations to input because that distorts the quality and hurts the performance
	2) [Optional] We want to save the outputs in the same size as that of the input image
"""


def predict_and_save(
		input_tensor: Tensor,
		file_name: str,
		orig_h: int,
		orig_w: int,
		model: BaseSegmentation,
		target_mask: Optional[Tensor] = None,
		device: Optional = torch.device("cup"),
		conf_mat: Optional[ConfusionMatrix] = None,
		color_map: List = None,
		orig_image: Optional[Image.Image] = None,
		adjust_label: Optional[int] = 0,
		is_cityscape: Optional[bool] = False,
):
	common = Common()
	"""Predict the segmentation mask and optionally save them"""
	mixed_precision_training = common.mixed_precision
	output_stride = model.output_stride

	if output_stride == 1:
		# we set it to 32 because most of the ImageNet models have 5 downsampling stages (2^5 = 32)
		output_stride = 32
	if orig_image is None:
		orig_image = F_vision.to_pil_image(input_tensor[0])

	curr_h, curr_w = input_tensor.shape[2:]

	# check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
	# if not, then resize them
	new_h = (curr_h // output_stride) * output_stride
	new_w = (curr_w // output_stride) * output_stride

	if new_h != curr_h or new_w != curr_w:
		# resize the input image, so that we do not get dimension mismatch errors in the forward pass
		input_tensor = F.interpolate(
			input=input_tensor, size=(new_h, new_w), mode="bilinear", align_corners=True
		)

	# get file name
	file_name = file_name.split(os.sep)[-1].split(".")[0] + ".png"

	# move data to device
	input_tensor = input_tensor.to(device)
	if target_mask is not None:
		target_mask = target_mask.to(device)

	# get prediction
	with autocast(enabled=mixed_precision_training):
		# prediction
		pred = model(input_tensor, orig_size=(orig_h, orig_w))

	if isinstance(pred, Tuple) and len(pred) == 2:
		# when segmentation mask from decoder and auxiliary decoder are returned
		pred = pred[0]
	elif isinstance(pred, Tensor):
		pred = pred
	else:
		raise NotImplementedError(
			"Predicted must should be either an instance of Tensor or Tuple[Tensor, Tensor]"
		)

	num_classes = pred.shape[1]
	pred_mask = pred.argmax(1).squeeze(0)

	# update confusion matrix
	if target_mask is not None and conf_mat is not None:
		conf_mat.update(
			ground_truth=target_mask.flatten(),
			prediction=pred_mask.flatten(),
			n_classes=num_classes,
		)

	save_dir = common.exp_loc

	# For some dataset, we need to adjust the labels. For example, we need adjust by 1 for ADE20k
	pred_mask = pred_mask + adjust_label
	if target_mask is not None:
		target_mask = target_mask + adjust_label

	# Visualize results and save to the directory
	if model.apply_color_map:
		draw_colored_masks(
			orig_image=orig_image,
			pred_mask=pred_mask,
			target_mask=target_mask,
			results_location=save_dir,
			color_map=color_map,
			file_name=file_name,
		)

	if model.save_masks:
		draw_binary_masks(
			pred_mask=pred_mask,
			file_name=file_name,
			is_cityscape=is_cityscape,
			results_location=save_dir,
		)
	Logger.log(
		"Segmentation results for {} are stored at: {}".format(file_name, save_dir)
	)


def predict_labeled_dataset() -> None:
	device = Util.get_device()
	dataset_name = Opt.get_target_subgroup_name(Entry.Dataset)

	# set-up data loaders
	eval_loader = Entry.get_entity(Entry.DataLoader, "eval")

	# set-up the model
	model = Entry.get_entity(Entry.SegmentationModel)
	model.eval()
	model = model.to(device=device)
	EngineUtil.print_summary(model=model)

	if model.training:
		Logger.log("Model is in training mode. Switching to evaluation mode")
		model.eval()

	color_map = Colormap().get_color_map_list()
	adjust_label = 0
	is_cityscape = False
	conf_mat = ConfusionMatrix()
	if hasattr(eval_loader.dataset, "color_palette"):
		color_map = eval_loader.dataset.color_palette()

	if hasattr(eval_loader.dataset, "adjust_mask_value"):
		adjust_label = eval_loader.dataset.adjust_mask_value()

	if dataset_name is not None and dataset_name.lower() == "cityscapes":
		is_cityscape = True

	with torch.no_grad():
		for batch_id, batch in tqdm(enumerate(eval_loader)):
			input_img, target_label = batch["image"], batch["label"]
			batch_size = input_img.shape[0]
			assert (
				batch_size == 1
			), "We recommend to run segmentation evaluation with a batch size of 1"

			predict_and_save(
				input_tensor=input_img,
				file_name=batch["file_name"][0],
				orig_w=batch["im_width"][0].item(),
				orig_h=batch["im_height"][0].item(),
				model=model,
				target_mask=target_label,
				device=device,
				conf_mat=conf_mat,
				color_map=color_map,
				adjust_label=adjust_label,
				is_cityscape=is_cityscape,
			)

	acc_global, acc, iu = conf_mat.compute()
	Logger.info("Quantitative results")
	print(
		"global correct: {:.2f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.2f}".format(
			acc_global.item() * 100,
			["{:.2f}".format(i) for i in (acc * 100).tolist()],
			["{:.2f}".format(i) for i in (iu * 100).tolist()],
			iu.mean().item() * 100,
		)
	)

	is_city_dataset = dataset_name == "Cityscapes"
	if is_city_dataset:
		pred_dir = "{}/predictions_no_cmap/".format(
			Util.get_export_location()
		)
		gt_dir = os.path.join(Opt.get_by_cfg_path(Entry.Dataset, "root_val"), "gtFine/val/")
		EngineUtil.eval_cityscapes(pred_dir=pred_dir, gt_dir=gt_dir)


def draw_binary_masks(
	pred_mask: Tensor,
	file_name: str,
	results_location: str,
	is_cityscape: Optional[bool] = False,
	) -> None:
	"""Save masks whose values ranges between 0 and number_of_classes - 1"""
	no_color_mask_dir = f"{results_location}/predictions_no_cmap"
	if not os.path.isdir(no_color_mask_dir):
		os.makedirs(no_color_mask_dir, exist_ok=True)
	no_color_mask_f_name = f"{no_color_mask_dir}/{file_name}"

	if is_cityscape:
		# convert mask values to cityscapes format
		pred_mask = Visualization.convert_to_cityscape_format(img=pred_mask)
	pred_mask_pil = F_vision.to_pil_image(pred_mask.byte())
	pred_mask_pil.save(no_color_mask_f_name)


def draw_colored_masks(
	orig_image: Image.Image,
	pred_mask: Tensor,
	target_mask: Tensor,
	file_name: str,
	results_location: str,
	color_map: Optional[List] = None,
	) -> None:
	"""Apply color map to segmentation masks"""

	alpha = Opt.get_by_cfg_path(Entry.SegmentationModel, "overlay_mask_weight")
	save_overlay_rgb_pred = Opt.get_by_cfg_path(Entry.SegmentationModel, "save_overlay_rgb_pred")

	if color_map is None:
		color_map = Colormap().get_color_map_list()

	# convert predicted tensor to PIL images, apply color map and save
	pred_mask_pil = F_vision.to_pil_image(pred_mask.byte())
	pred_mask_pil.putpalette(color_map)
	pred_mask_pil = pred_mask_pil.convert("RGB")
	pred_color_mask_dir = f"{results_location}/predictions_cmap"
	if not os.path.isdir(pred_color_mask_dir):
		os.makedirs(pred_color_mask_dir, exist_ok=True)
	color_mask_f_name = f"{pred_color_mask_dir}/{file_name}"
	pred_mask_pil.save(color_mask_f_name)

	if target_mask is not None:
		# convert target tensor to PIL images, apply colormap, and save
		target_mask_pil = F_vision.to_pil_image(target_mask.byte())
		target_mask_pil.putpalette(color_map)
		target_mask_pil = target_mask_pil.convert("RGB")
		target_color_mask_dir = f"{results_location}/gt_cmap"
		if not os.path.isdir(target_color_mask_dir):
			os.makedirs(target_color_mask_dir, exist_ok=True)
		gt_color_mask_f_name = f"{target_color_mask_dir}/{file_name}"
		target_mask_pil.save(gt_color_mask_f_name)

	if save_overlay_rgb_pred and orig_image is not None:
		# overlay predicted mask on top of original image and save

		if pred_mask_pil.size != orig_image.size:
			# resize if input image size is not the same as predicted mask.
			# this is likely in case of labeled datasets where we use transforms on the input image
			orig_image = F_vision.resize(
				orig_image,
				size=pred_mask_pil.size[::-1],
				interpolation=F_vision.InterpolationMode.BILINEAR,
			)

		overlayed_img = Image.blend(pred_mask_pil, orig_image, alpha=alpha)
		overlay_mask_dir = f"{results_location}/predictions_overlay"
		if not os.path.isdir(overlay_mask_dir):
			os.makedirs(overlay_mask_dir, exist_ok=True)
		overlay_mask_f_name = f"{overlay_mask_dir}/{file_name}"
		overlayed_img.save(overlay_mask_f_name)

		# save original image
		rgb_image_dir = f"{results_location}/rgb_images"
		if not os.path.isdir(rgb_image_dir):
			os.makedirs(rgb_image_dir, exist_ok=True)
		rgb_image_f_name = f"{rgb_image_dir}/{file_name}"
		orig_image.save(rgb_image_f_name)


def read_and_process_image(image_fname: str):
	input_img = Image.open(image_fname).convert("RGB")
	input_pil = copy.deepcopy(input_img)
	orig_w, orig_h = input_img.size

	# Resize the image while maintaining the aspect ratio
	res_h, res_w = SplUtil.image_size_from_opts()

	input_img = F_vision.resize(
		input_img,
		size=min(res_h, res_w),
		interpolation=F_vision.InterpolationMode.BILINEAR,
	)
	input_tensor = F_vision.pil_to_tensor(input_img)
	input_tensor = input_tensor.float().div(255.0).unsqueeze(0)
	return input_tensor, input_pil, orig_h, orig_w


def predict_image(image_file_name: str) -> None:

	if not os.path.isfile(image_file_name):
		Logger.error(f"Image file does not exist at: {image_file_name}")

	input_tensor, input_pil, orig_h, orig_w = read_and_process_image(
		image_fname=image_file_name
	)

	# get file name
	image_file_name = image_file_name.split(os.sep)[-1]

	device = Util.get_device()
	# set-up the model
	model = Entry.get_entity(Entry.SegmentationModel)
	model.eval()
	model = model.to(device=device)
	EngineUtil.print_summary(model=model)

	if model.training:
		Logger.log("Model is in training mode. Switching to evaluation mode")
		model.eval()

	with torch.no_grad():
		predict_and_save(
			input_tensor=input_tensor,
			file_name=image_file_name,
			orig_h=orig_h,
			orig_w=orig_w,
			model=model,
			target_mask=None,
			device=device,
			orig_image=input_pil,
		)


def predict_images_in_folder() -> None:
	model: BaseSegmentation = Entry.get_entity(Entry.SegmentationModel)

	img_files = []
	for e in DataConstants.SUPPORTED_IMAGE_EXTNS:
		img_files_with_extn = glob.glob(f"{model.image_folder_path}/*{e}")
		if len(img_files_with_extn) > 0 and isinstance(img_files_with_extn, list):
			img_files.extend(img_files_with_extn)

	if len(img_files) == 0:
		Logger.error(
			"Number of image files found at {}: {}".format(
				model.image_folder_path, len(img_files)
			)
		)

	Logger.log(
		"Number of image files found at {}: {}".format(model.image_folder_path, len(img_files))
	)

	device = Util.get_device()
	# set-up the model
	model.eval()
	model = model.to(device=device)
	EngineUtil.print_summary(model=model)

	if model.training:
		Logger.log("Model is in training mode. Switching to evaluation mode")
		model.eval()

	with torch.no_grad():
		for image_fname in tqdm(img_files):
			input_tensor, input_pil, orig_h, orig_w = read_and_process_image(
				image_fname=image_fname
			)

			image_file_name = image_fname.split(os.sep)[-1]

			predict_and_save(
				input_tensor=input_tensor,
				file_name=image_file_name,
				orig_h=orig_h,
				orig_w=orig_w,
				model=model,
				target_mask=None,
				device=device,
				orig_image=input_pil,
			)


def main_segmentation_evaluation():
	pass


if __name__ == "__main__":
	main_segmentation_evaluation()
