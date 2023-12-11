from torch import Tensor
from typing import Union, Dict, Tuple, Optional

from utils.entry_utils import Entry

from . import BaseSegmentation
from ..classification import BaseEncoder


@Entry.register_entry(Entry.SegmentationModel)
class SegEncoderDecoder(BaseSegmentation):
	"""
	This class defines an encoder-decoder architecture for the task of semantic segmentation. Different segmentation
	heads (e.g., PSPNet and DeepLabv3) can be used

	Args:
		encoder (BaseEncoder): Backbone network (e.g., MobileViT or ResNet)
	"""
	def __init__(self, encoder: BaseEncoder):
		super().__init__(encoder=encoder)

		# delete layers that are not required in segmentation network
		self.encoder.classifier = None
		if not self.use_tail:
			self.encoder.tail = None

		self.seg_head = Entry.get_entity(Entry.Layer, entry_name=self.head_layer, seg_model=self)

	def get_trainable_parameters(
		self,
		weight_decay: Optional[float] = 0.0,
		no_decay_bn_filter_bias: Optional[bool] = False,
	):
		"""This function separates the parameters for backbone and segmentation head, so that
		different learning rates can be used for backbone and segmentation head
		"""
		encoder_params, enc_lr_mult = self.encoder.get_trainable_parameters(
			weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias
		)
		decoder_params, dec_lr_mult = self.seg_head.get_trainable_parameters(
			weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias
		)

		total_params = sum([p.numel() for p in self.parameters()])
		encoder_params_count = sum([p.numel() for p in self.encoder.parameters()])
		decoder_params_count = sum([p.numel() for p in self.seg_head.parameters()])

		assert total_params == encoder_params_count + decoder_params_count, (
			"Total network parameters are not equal to "
			"the sum of encoder and decoder. "
			"{} != {} + {}".format(
				total_params, encoder_params_count, decoder_params_count
			)
		)

		return encoder_params + decoder_params, enc_lr_mult + dec_lr_mult

	def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
		enc_end_points: Dict = self.encoder.extract_end_points_all(
			x, use_l5=True, use_tail=self.use_tail
		)
		return self.seg_head(enc_out=enc_end_points)
