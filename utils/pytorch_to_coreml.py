import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torchvision.transforms import functional as F

from typing import Optional, Dict, Tuple, Union
import coremltools as ct

from torch.utils.mobile_optimizer import optimize_for_mobile
from engine import Conversion


class CoremlConvertor:

	@classmethod
	def convert_pytorch_to_coreml(
			cls,
			pytorch_model: torch.nn.Module,
			jit_model_only: Optional[bool] = False) -> Dict:
		"""
		Convert Pytorch model to CoreML

		:param pytorch_model: Pytorch model that needs to be converted to JIT or CoreML
		:param jit_model_only:
		:return: CoreML model or package
		"""
		conversion = Conversion()

		if conversion.input_image_path is not None:
			input_pil_img = Image.open(conversion.input_image_path).convert("RGB")
			input_pil_img = F.resize(
				img=input_pil_img, size=256, interpolation=F.InterpolationMode.BILINEAR
			)
			input_pil_img = F.center_crop(img=input_pil_img, output_size=224)
			input_tensor = F.pil_to_tensor(input_pil_img).float()
			input_tensor.div_(255.0)
			input_tensor = input_tensor.unsqueeze(0)  # add dummy batch dimension
		else:
			input_pil_img = None
			input_tensor = pytorch_model.generate_input()

		# if pytorch_model.training:
		# 	pytorch_model.eval()

		pytorch_model.eval()
		pytorch_model.training = False

		with torch.no_grad():
			pytorch_out = pytorch_model(input_tensor)

			jit_model = torch.jit.trace(pytorch_model, input_tensor)
			jit_out = jit_model(input_tensor)
			cls.assertion_check(py_out=pytorch_out, jit_out=jit_out)

			jit_model_optimized = optimize_for_mobile(jit_model)
			jit_optimized_out = jit_model_optimized(input_tensor)
			cls.assertion_check(py_out=pytorch_out, jit_out=jit_optimized_out)

			# for benchmark
			if jit_model_only and torch.cuda.device_count() > 0:
				# For inference on GPU
				return {"coreml": None, "jit": jit_model, "jit_optimized": None}
			elif jit_model_only and torch.cuda.device_count() == 0:
				# For inference on CPU
				return {"coreml": None, "jit": jit_model_optimized, "jit_optimized": None}

			coreml_model = ct.convert(
				model=jit_model,
				inputs=[
					ct.ImageType(name="input", shape=input_tensor.shape, scale=1.0 / 255.0)
				],
				convert_to="neuralnetwork",
				# preprocessing_args={"scale": 1.0/255.0},
				# minimum_deployment_target=ct.target.iOS15,
				# compute_precision=ct.precision.FLOAT16
			)

			if input_pil_img is not None:
				out = coreml_model.predict({"input": input_pil_img})

			return {
				"coreml": coreml_model,
				"jit": jit_model,
				"jit_optimized": jit_model_optimized,
			}

	@staticmethod
	def assertion_check(
		py_out: Union[Tensor, Dict, Tuple], jit_out: Union[Tensor, Dict, Tuple]
	) -> None:
		if isinstance(py_out, Dict):
			assert isinstance(jit_out, Dict)
			keys = py_out.keys()
			for k in keys:
				np.testing.assert_almost_equal(
					py_out[k].cpu().numpy(),
					jit_out[k].cpu().numpy(),
					decimal=3,
					verbose=True,
				)
		elif isinstance(py_out, Tensor):
			assert isinstance(jit_out, Tensor)
			np.testing.assert_almost_equal(
				py_out.cpu().numpy(), jit_out.cpu().numpy(), decimal=3, verbose=True
			)
		elif isinstance(py_out, Tuple):
			assert isinstance(jit_out, Tuple)
			for x, y in zip(py_out, jit_out):
				np.testing.assert_almost_equal(
					x.cpu().numpy(), y.cpu().numpy(), decimal=3, verbose=True
				)

		else:
			raise NotImplementedError(
				"Only Dictionary[Tensors] or Tuple[Tensors] or Tensors are supported as outputs"
			)
