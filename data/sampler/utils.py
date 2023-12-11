from gbl import Constants
from utils.type_utils import Opt

from typing import Optional, List
from utils.math_utils import Math
import numpy as np

from utils.entry_utils import Entry


class SplUtil:
	@staticmethod
	# get image size from options
	def image_size_from_opts():
		try:
			sampler_name = Opt.get_target_subgroup_name(Entry.Sampler).lower()
			if "var" in sampler_name:
				tar_sampler = "VariableBatchSamplerDDP" if "ddp" in sampler_name else "VariableBatchSampler"
			elif "multi" in sampler_name:
				tar_sampler = "MultiScaleSamplerDDP" if "ddp" in sampler_name else "MultiScaleSampler"
			else:
				tar_sampler = "BatchSamplerDDP" if "ddp" in sampler_name else "BatchSampler"

			im_w = Opt.get(f"{Entry.Sampler}.{tar_sampler}.crop_size_width")
			im_h = Opt.get(f"{Entry.Sampler}.{tar_sampler}.crop_size_height")

		except Exception:
			im_h = Constants.DEFAULT_IMAGE_HEIGHT
			im_w = Constants.DEFAULT_IMAGE_WIDTH
		return im_h, im_w

	@staticmethod
	# build few crop-sizes for random size
	# note : batch_size will be changed
	def image_batch_pairs(
		crop_size_width: int,
		crop_size_height: int,
		batch_size_gpu0: int,
		n_gpus: int,
		max_img_scales: Optional[float] = 5,
		check_scale_div_factor: Optional[int] = 32,
		min_crop_size_width: Optional[int] = 160,
		max_crop_size_width: Optional[int] = 320,
		min_crop_size_height: Optional[int] = 160,
		max_crop_size_height: Optional[int] = 320,
		**kwargs
	) -> List:
		"""
		This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
			are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

		Args:
			crop_size_width (int): Base Image width (e.g., 224)
			crop_size_height (int): Base Image height (e.g., 224)
			batch_size_gpu0 (int): Batch size on GPU 0 for base image
			n_gpus (int): Number of available GPUs
			max_img_scales (Optional[int]): Number of scales. How many image sizes that we want to generate between min and max scale factors. Default: 5
			check_scale_div_factor (Optional[int]): Check if image scales are divisible by this factor. Default: 32
			min_crop_size_width (Optional[int]): Min. crop size along width. Default: 160
			max_crop_size_width (Optional[int]): Max. crop size along width. Default: 320
			min_crop_size_height (Optional[int]): Min. crop size along height. Default: 160
			max_crop_size_height (Optional[int]): Max. crop size along height. Default: 320

		Returns:
			a sorted list of tuples. Each index is of the form (h, w, batch_size)

		"""
		width_dims = list(np.linspace(min_crop_size_width, max_crop_size_width, max_img_scales))
		if crop_size_width not in width_dims:
			width_dims.append(crop_size_width)

		height_dims = list(np.linspace(min_crop_size_height, max_crop_size_height, max_img_scales))
		if crop_size_height not in height_dims:
			height_dims.append(crop_size_height)

		image_scales = set()

		for h, w in zip(height_dims, width_dims):
			# ensure that sampled sizes are divisible by check_scale_div_factor
			# This is important in some cases where input undergoes a fixed number of down-sampling stages
			# for instance, in ImageNet training, CNNs usually have 5 down sampling stages, which down samples the
			# input image of resolution 224x224 to 7x7 size
			h = Math.make_divisible(h, check_scale_div_factor)
			w = Math.make_divisible(w, check_scale_div_factor)
			image_scales.add((h, w))

		image_scales = list(image_scales)

		img_batch_tuples = set()
		n_elements = crop_size_width * crop_size_height * batch_size_gpu0
		for (crop_h, crop_y) in image_scales:
			# compute the batch size for sampled image resolutions with respect to the base resolution
			# specified base resolution gets batch_size_gpu0 samples, higher crop resolution gets more samples, lower corp resolution ...
			_bsz = max(1, int(round(n_elements / (crop_h * crop_y), 2)))

			_bsz = Math.make_divisible(_bsz, n_gpus)  # make batch size divisible by number of gpus
			img_batch_tuples.add((crop_h, crop_y, _bsz))

		img_batch_tuples = list(img_batch_tuples)
		return sorted(img_batch_tuples)

	@staticmethod
	def make_video_pairs(
		crop_size_height: int,
		crop_size_width: int,
		min_crop_size_height: int,
		max_crop_size_height: int,
		min_crop_size_width: int,
		max_crop_size_width: int,
		num_frames_per_clip: int,
		max_img_scales: Optional[int] = 5,
		check_scale_div_factor: Optional[int] = 32,
		**kwargs
	) -> List:
		"""
		This function creates number of frames and spatial size pairs for videos.

		Args:
			crop_size_height (int): Base Image height (e.g., 224)
			crop_size_width (int): Base Image width (e.g., 224)
			min_crop_size_width (int): Min. crop size along width.
			max_crop_size_width (int): Max. crop size along width.
			min_crop_size_height (int): Min. crop size along height.
			max_crop_size_height (int): Max. crop size along height.
			num_frames_per_clip (int): Default number of frames per clip in a video.
			max_img_scales (Optional[int]): Number of scales. Default: 5
			check_scale_div_factor (Optional[int]): Check if spatial scales are divisible by this factor. Default: 32
		Returns:
			a sorted list of tuples. Each index is of the form (h, w, n_frames)
		"""

		width_dims = list(np.linspace(min_crop_size_width, max_crop_size_width, max_img_scales))
		if crop_size_width not in width_dims:
			width_dims.append(crop_size_width)
		height_dims = list(np.linspace(min_crop_size_height, max_crop_size_height, max_img_scales))
		if crop_size_height not in height_dims:
			height_dims.append(crop_size_height)

		# ensure that spatial dimensions are divisible by check_scale_div_factor
		width_dims = [Math.make_divisible(w, check_scale_div_factor) for w in width_dims]
		height_dims = [Math.make_divisible(h, check_scale_div_factor) for h in height_dims]
		batch_pairs = set()
		n_elements = crop_size_width * crop_size_height * num_frames_per_clip
		for (h, w) in zip(height_dims, width_dims):
			n_frames = max(1, int(round(n_elements / (h * w), 2)))
			batch_pairs.add((h, w, n_frames))
		return sorted(list(batch_pairs))
