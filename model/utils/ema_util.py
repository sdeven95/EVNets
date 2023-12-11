from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry
from utils import Util

import copy
import torch


@Entry.register_entry(Entry.EMAConfigure)
class EMAWraper(Cfg, Dsp):
	__slots__ = [
		"enable", "momentum", "copy_at_epoch",
		"device", "ema_model", "ema_has_module"
	]
	_keys_ = __slots__[:-3]
	_disp_ = __slots__
	_defaults_ = [False, 0.0001, -1]
	_types_ = [bool, float, int, ]

	_cfg_path_ = Entry.EMAConfigure

	def __init__(self, model, **kwargs):
		super().__init__(**kwargs)

		if not self.enable:
			return

		self.device = Util.get_device()

		self.ema_model = copy.deepcopy(model)
		self.ema_model.eval()
		if self.device:
			self.ema_model.to(self.device)
		for param in self.ema_model.parameters():
			param.requires_grad = False

	def update_parameters(self, model):
		with torch.no_grad():  # no gradient
			msd = model.state_dict()
			for k, ema_v in self.ema_model.state_dict().items():
				if hasattr(model, "module") and not hasattr(self.ema_model, "module"):
					# .module is added if we use DistributedDataParallel or DataParallel wrappers around model
					k = "module." + k
				elif not hasattr(model, "module") and hasattr(self.ema_model, "module"):
					k = k.replace("module.", "")
				model_v = msd[k].detach()
				if self.device:
					model_v = model_v.to(device=self.device)
				ema_v.copy_((ema_v * (1.0 - self.momentum)) + (self.momentum * model_v))  # get parameter ema value
