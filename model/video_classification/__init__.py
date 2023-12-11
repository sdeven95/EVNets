from utils.type_utils import Cfg, Dsp, Prf
from utils.entry_utils import Entry
from utils.logger import Logger

from ..utils import MdlUtil
from layer.utils import LayerUtil

from torch import Tensor, nn


class VideoClassificationModelUtil:
	@staticmethod
	def get_model(entry_name, *args, **kwargs):

		model = Entry.dict_entry[Entry.VideoClassificationModel][entry_name](*args, **kwargs)

		if model.pretrained:
			MdlUtil.load_pretrained_model(model, model.pretrained)

		if model.freeze_batch_norm:
			MdlUtil.freeze_norm_layers(model)

		return model


class BaseVideoEncoder(Cfg, Dsp, Prf, nn.Module):
	__slots__ = [
		"classifier_dropout", "n_classes", "global_pool",
		"pretrained", "freeze_batch_norm",
		"clip_out_voting_fn", "inference_mode",
		"round_nearest", "model_conf_dict"
	]
	_keys_ = __slots__[:-2]
	_disp_ = __slots__
	_defaults_ = [
				0.2, 400, "mean",
				None, False,
				"sum", False
			]
	_types_ = [
				float, int, str,
				str, bool,
				str, bool
			]

	_cfg_path_ = Entry.VideoClassificationModel

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		nn.Module.__init__(self)
		self.round_nearest = 8
		self.model_conf_dict = dict()

	def get_training_parameters(self, weight_decay=0.0, no_decay_bn_filter_bias=False):
		param_list = LayerUtil.split_parameters(self.named_parameters(), weight_decay, no_decay_bn_filter_bias)
		return param_list, [1.0] * len(param_list)

	def profile_model(self, x: Tensor) -> None:
		"""
		This function computes FLOPs using fvcore (if installed).
		"""
		Logger.double_dash_line(dashes=65)
		print("{:>35} Summary".format(self.__class__.__name__))
		Logger.double_dash_line(dashes=65)
		overall_params_py = sum([p.numel() for p in self.parameters()])

		try:
			from fvcore.nn import FlopCountAnalysis

			flop_analyzer = FlopCountAnalysis(self.eval(), x)
			flop_analyzer.unsupported_ops_warnings(False)
			total_flops = flop_analyzer.total()

			print(
				"Flops computed using FVCore for an input of size={} are {:>8.3f} G".format(
					list(x.shape), total_flops / 1e9
				)
			)
		except ModuleNotFoundError:
			pass

		print(
			"{:<20} = {:>8.3f} M".format(
				"Overall parameters (sanity check)", overall_params_py / 1e6
			)
		)
		Logger.double_dash_line(dashes=65)


# from .mobilevit_st import SpatioTemporalMobileViT
