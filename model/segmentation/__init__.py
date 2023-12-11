from torch import nn

from utils.type_utils import Cfg, Opt, Dsp, Prf
from utils.entry_utils import Entry

from .. import classification
from ..utils import MdlUtil


class SegmentationModelUtil:
	@staticmethod
	def get_model(entry_name, *args, **kwargs):

		output_stride = Opt.get_by_cfg_path(Entry.SegmentationModel, "output_stride")
		encoder = Entry.get_entity(Entry.ClassificationModel, output_stride=output_stride)
		model = Entry.dict_entry[Entry.SegmentationModel][entry_name](encoder=encoder)

		if model.pretrained:
			MdlUtil.load_pretrained_model(model, model.pretrained)

		if model.freeze_batch_norm:
			MdlUtil.freeze_norm_layers(model)

		return model


class BaseSegmentation(Cfg, Dsp, Prf, nn.Module):
	__slots__ = [
		"head_layer", "n_classes", "pretrained", "lr_multiplier", "classifier_dropout",
		"use_aux_head", "aux_dropout",
		"output_stride", "replace_stride_with_dilation",
		"freeze_batch_norm", "use_tail",
		"apply_color_map", "save_masks",
		"overlay_mask_weight", "save_overlay_rgb_pred",
		"image_folder_path", "resize_input_image_on_evaluation"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
				"DeeplabV3", 20, None, 1.0, 0.1,
				False, 0.1,
				1, False,
				False, False, False, 0.5, False, None, False,
			]
	_types_ = [
				str, int, str, float, float,
				bool, float,
				int, bool,
				bool, bool, bool, float, bool, str, bool
			]

	_cfg_path_ = Entry.SegmentationModel

	def __init__(self, encoder: classification.BaseEncoder, **kwargs):
		super().__init__(**kwargs)
		nn.Module.__init__(self)

		self.encoder: classification.BaseEncoder = encoder


# from .enc_dec import SegEncoderDecoder
