from torch import Tensor
import torch
from typing import Optional, Union, Dict
from .logger import Logger
from . import Util
import numpy as np
from torch import distributed as dist


class Tsr:
	@classmethod
	# Move Tensor or Tensor Dict To Device(Recursion)
	def move_to_device(
		cls,
		x: Union[Dict, Tensor],
		device: Optional[str] = "cpu"
	) -> Union[Dict, Tensor]:
		if isinstance(x, Dict):
			for k, v in x.items():
				if isinstance(v, Dict):
					x[k] = cls.move_to_device(v, device=device)
				elif isinstance(v, Tensor):
					x[k] = v.to(device=device, non_blocking=True)

		elif isinstance(x, Tensor):
			x = x.to(device=device)
		else:
			Logger.error(
				"Inputs of type Tensor or Dict of Tensors are only supported right now"
			)
		return x

	@staticmethod
	# set tensor to average across process
	def reduce_tensor(inp_tensor: Tensor) -> Tensor:
		size = dist.get_world_size() if dist.is_initialized() else 1
		inp_tensor_clone = inp_tensor.clone().detach()
		# dist_barrier()
		dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
		inp_tensor_clone /= size
		return inp_tensor_clone

	@staticmethod
	# set tensor to sum across process
	def reduce_tensor_sum(inp_tensor: torch.Tensor) -> torch.Tensor:
		inp_tensor_clone = inp_tensor.clone().detach()
		# dist_barrier()
		dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
		return inp_tensor_clone

	@classmethod
	# convert tensor to numpy array( n_element > 1) or float scalar
	def tensor_to_numpy_or_float(
		cls,
		inp_tensor: Union[int, float, Tensor],
	) -> Union[int, float, np.ndarray]:
		# reduce tensor across processes
		if Util.is_distributed() and isinstance(inp_tensor, Tensor):
			inp_tensor = cls.reduce_tensor(inp_tensor=inp_tensor)

		if isinstance(inp_tensor, Tensor) and inp_tensor.numel() > 1:
			# For IOU, we get a C-dimensional tensor (C - number of classes)
			# so, we convert here to a numpy array
			return inp_tensor.cpu().numpy()
		elif hasattr(inp_tensor, "item"):
			return inp_tensor.item()
		elif isinstance(inp_tensor, (int, float)):
			return inp_tensor * 1.0
		else:
			raise NotImplementedError(
				"The data type is not supported yet in tensor_to_python_float function"
			)

	@staticmethod
	# gather pick-able object to a list across process of group
	def all_gather_list(data):
		world_size = dist.get_world_size()
		data_list = [None] * world_size
		# dist_barrier()
		dist.all_gather_object(data_list, data)
		return data_list

	# @staticmethod
	# # create a random tensor
	# def create_rand_tensor(device="cpu", batch_size=1):
	# 	# !!! note: need to add code to create video random tensor !!!
	# 	from data.sampler.utils import SplUtil
	# 	im_h, im_w = SplUtil.image_size_from_opts()
	# 	inp_tensor = torch.randint(low=0, high=255, size=(batch_size, Constants.DEFAULT_IMAGE_CHANNELS, im_h, im_w), device=device)
	# 	inp_tensor = inp_tensor.float().div(255.0)
	# 	return inp_tensor

	@staticmethod
	# Covert BCHW Tensor to BHWC numpy array, (0, 1) => (0, 255), i.e. reverse Transform.ToTensor()
	def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
		# [0, 1] --> [0, 255]
		img_tensor = torch.mul(img_tensor, 255.0)
		# BCHW --> BHWC
		img_tensor = img_tensor.permute(0, 2, 3, 1)

		img_np = img_tensor.byte().cpu().numpy()
		return img_np
