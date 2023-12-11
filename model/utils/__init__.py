import os
import torch
from layer.normalization import BaseNorm
from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils import Util


class MdlUtil:

	# freeze normalization layers
	@staticmethod
	def freeze_norm_layers(model) -> (bool, int):

		model = model.module if hasattr(model, "module") else model

		for m in model.modules():
			if isinstance(m, BaseNorm):
				m.eval()
				m.weight.requires_grad = False
				m.bias.requires_grad = False
				m.training = False

		for m in model.modules():
			if isinstance(m, BaseNorm):
				assert not m.weight.requires_grad, "Something is wrong when freezing normalization layer. Please check"

	# load the pretrained weights form the pretrained file
	@staticmethod
	def load_pretrained_model(model, wt_loc):
		model = Util.get_module(model)
		assert os.path.isfile(wt_loc), f"Pretrained file is not here: {wt_loc}"
		map_location = Opt.get(f"{Entry.Common}.Dev.device")
		model.load_state_dict(torch.load(wt_loc, map_location=map_location))



