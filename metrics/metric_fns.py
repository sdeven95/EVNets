from torch import Tensor
import torch
from typing import Optional, Union, Tuple


class MetricFun:

	@staticmethod
	def top_k_accuracy(
		output: Tensor, target: Tensor, top_k: Optional[tuple] = (1,)
	) -> list:
		# generally, top_k = (1, 5), maximum_k = 5
		maximum_k = max(top_k)
		batch_size = target.shape[0]  # batch_size

		# get the k-largest elements' indexes, batch in line and indexes in column [batch, k]
		_, pred = output.topk(maximum_k, 1, True, True)
		# transpose, so now indexes in line and batch in column [k, batch]
		pred = pred.t()
		# prepare target, [batch, 1] -> [1, batch] -> [k, batch]
		# and judge if prediction and target match by every position
		# note every column has only one element with value 1
		correct = pred.eq(target.reshape(1, -1).expand_as(pred))

		results = []
		for k in top_k:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # sum matches
			acc_k = correct_k.mul_(100.0 / batch_size)
			results.append(acc_k)
		return results

	@staticmethod
	# psnr = 10 * log10 (1/mse)
	def compute_psnr(
		prediction: Tensor, target: Tensor, no_uint8_conversion: Optional[bool] = False
	) -> Tensor:

		if not no_uint8_conversion:
			prediction = prediction.mul(255.0).to(torch.uint8)
			target = target.mul(255.0).to(torch.uint8)
			MAX_I = 255 ** 2
		else:
			MAX_I = 1

		error = torch.pow(prediction - target, 2).float()
		mse = torch.mean(error) + 1e-10  # mse
		psnr = 10.0 * torch.log10(MAX_I / mse)  # psnr by mse
		return psnr

	@staticmethod
	def compute_miou_batch(
		prediction: Union[Tuple[Tensor, Tensor], Tensor],
		target: Tensor,
		epsilon: Optional[float] = 1e-7,
	):
		if isinstance(prediction, Tuple) and len(prediction) == 2:
			mask = prediction[0]
			assert isinstance(mask, Tensor)
		elif isinstance(prediction, Tensor):
			mask = prediction
			assert isinstance(mask, Tensor)
		else:
			raise NotImplementedError(
				"For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor"
			)

		num_classes = mask.shape[1]
		pred_mask = torch.max(mask, dim=1)[1]
		assert (
			pred_mask.dim() == 3
		), "Predicted mask tensor should be 3-dimensional (B x H x W)"

		pred_mask = pred_mask.byte()
		target = target.byte()

		# shift by 1 so that 255 is 0
		pred_mask += 1
		target += 1

		pred_mask = pred_mask * (target > 0)
		inter = pred_mask * (pred_mask == target)
		area_inter = torch.histc(inter.float(), bins=num_classes, min=1, max=num_classes)
		area_pred = torch.histc(pred_mask.float(), bins=num_classes, min=1, max=num_classes)
		area_target = torch.histc(target.float(), bins=num_classes, min=1, max=num_classes)
		area_union = area_pred + area_target - area_inter + epsilon
		return area_inter, area_union
