from common import Common
from engine import Conversion
from utils.entry_utils import Entry
import os
from utils.pytorch_to_coreml import CoremlConvertor
from utils.opt_setup_utils import OptSetup
from utils.type_utils import Opt
import torch
from utils.logger import Logger


class ConverseModel:
	@staticmethod
	def run(model_cls):
		OptSetup.get_arguments(model_cls)

		cmm = Common()
		conversion = Conversion()

		assert conversion.ckpt_path is not None, "For conversion, please set 'Conversion.ckpt_path' parameter!"

		# set coreml conversion flag to true
		cmm.enable_coreml_compatible_module = True

		# not support sync batch normalization
		norm_layer_name = Opt.get_target_subgroup_name(Entry.Norm)
		if "sync" in norm_layer_name.lower():
			Opt.set_target_subgroup_name(Entry.Norm, "BatchNorm2d")

		# prepare model
		model: torch.nn.Module = Entry.get_entity(Entry.Model)
		model.load_state_dict(torch.load(conversion.ckpt_path))  # load check point
		model_name = model.__class__.__name__
		model.eval()
		model.training = False

		# make saving path
		if not os.path.isdir(cmm.result_loc):
			os.makedirs(cmm.result_loc)
		model_dst_loc = f"{cmm.result_loc}/{model_name}.{conversion.coreml_extn}"
		if os.path.isfile(model_dst_loc):
			os.remove(model_dst_loc)

		try:
			# convert
			converted_models_dict = CoremlConvertor.convert_pytorch_to_coreml(pytorch_model=model)

			coreml_model = converted_models_dict["coreml"]
			jit_model = converted_models_dict["jit"]
			jit_optimized = converted_models_dict["jit_optimized"]

			# save coreml format
			coreml_model.save(model_dst_loc)
			# save jit format
			torch.jit.save(jit_model, model_dst_loc)
			# save optimized jit format
			jit_optimized._save_for_lite_interpreter(model_dst_loc)

			Logger.log("PyTorch model converted to CoreML successfully.")
			Logger.log(f"CoreML model location: {model_dst_loc}")

		except Exception as e:
			Logger.error(f"PyTorch to CoreML conversion failed. See below for error details:\n {e}")
