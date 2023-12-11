from ..utils import MdlUtil
from layer.utils import LayerUtil

from utils.type_utils import Cfg, Dsp, Prf
from utils.entry_utils import Entry

from .. import classification

from torch import nn
from typing import NamedTuple, Any


class DetectionModelUtil:
	@staticmethod
	def get_model(entry_name, *args, **kwargs):

		encoder = Entry.get_entity(Entry.ClassificationModel)
		model = Entry.get_entity(Entry.DetectionModel)[entry_name](encoder=encoder, *args, **kwargs)

		if model.pretrained:
			MdlUtil.load_pretrained_model(model, model.pretrained)

		if model.freeze_batch_norm:
			MdlUtil.freeze_norm_layers(model)

		return model


DetectionPredTuple = NamedTuple(
	"DetectionPredTuple", [("labels", Any), ("scores", Any), ("boxes", Any)]
)


class BaseDetection(Cfg, Dsp, Prf, nn.Module):
	__slots__ = [
		"n_classes", "pretrained", "output_stride", "replace_stride_with_dilation", "freeze_batch_norm",
		"encoder", "enc_l1_channels", "enc_l2_channels", "enc_l3_channels", "enc_l4_channels", "enc_l5_channels"
	]
	_keys_ = __slots__[:5]
	_disp_ = __slots__
	_defaults_ = [None, None, None, False, False]
	_types_ = [int, str, int, bool, bool]

	_cfg_path_ = Entry.DetectionModel

	def __init__(self, encoder: classification.BaseEncoder, **kwargs):
		super().__init__(**kwargs)
		nn.Module.__init__(self)

		self.encoder = encoder
		enc_conf = self.encoder.model_conf_dict

		self.enc_l1_channels = enc_conf["layer1"]["out"]
		self.enc_l2_channels = enc_conf["layer2"]["out"]
		self.enc_l3_channels = enc_conf["layer3"]["out"]
		self.enc_l4_channels = enc_conf["layer4"]["out"]
		self.enc_l5_channels = enc_conf["layer5"]["out"]

	def get_training_parameters(self, weight_decay=0.0, no_decay_bn_filter_bias=False):
		param_list = LayerUtil.split_parameters(self.named_parameters(), weight_decay, no_decay_bn_filter_bias)
		return param_list, [1.0] * len(param_list)

	@staticmethod
	def profile_layer(layer, x):
		# profile a layer
		block_params = block_macs = 0.0
		if isinstance(layer, nn.Sequential):
			for layer_i in range(len(layer)):
				x, layer_param, layer_macs = layer[layer_i].profile_module(x)
				block_params += layer_param
				block_macs += layer_macs
				print(
					"{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
						layer[layer_i].__class__.__name__,
						"Params",
						round(layer_param / 1e6, 3),
						"MACs",
						round(layer_macs / 1e6, 3),
					)
				)
		else:
			x, layer_param, layer_macs = layer.profile_module(x)
			block_params += layer_param
			block_macs += layer_macs
			print(
				"{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
					layer.__class__.__name__,
					"Params",
					round(layer_param / 1e6, 3),
					"MACs",
					round(layer_macs / 1e6, 3),
				)
			)
		return x, block_params, block_macs


# from .ssd import SingleShotMaskDetector
