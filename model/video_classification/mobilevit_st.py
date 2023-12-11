import torch
from torch import nn, Tensor
from typing import Dict, Tuple, Optional

from . import BaseVideoEncoder
from layer import LinearLayer, GlobalPool, Dropout, ConvLayer3D

from layer.utils import InitParaUtil
from utils.entry_utils import Entry


@Entry.register_entry(Entry.VideoClassificationModel)
class SpatioTemporalMobileViT(BaseVideoEncoder):
	"""
	This class defines the spatio-temporal `MobileViT <https://arxiv.org/abs/2110.02178>`_ model for the
	task of video classification
	"""

	os_keys = ["os_8", "os_16", "os_32"]

	def __init__(self) -> None:
		super().__init__()

		# load MobileViT Image model
		self.mobilevit_img_model = Entry.get_entity(Entry.ClassificationModel)
		self.mobilevit_img_model.tail = None
		self.mobilevit_img_model.classifier = None

		in_channels = self.mobilevit_img_model.model_conf_dict["tail"]["in"]
		out_channels = self.mobilevit_img_model.model_conf_dict["tail"]["out"]

		self.temporal_conv = nn.Sequential(
			ConvLayer3D(
				in_channels=in_channels,
				out_channels=in_channels,
				kernel_size=3,
				stride=1,
				padding=1,
				use_norm=True,
				use_act=False,
				groups=in_channels,
			),
			ConvLayer3D(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				stride=1,
				padding=1,
				use_norm=True,
				use_act=True,
			),
		)

		self.global_pool_layer = GlobalPool(pool_type=self.global_pool, keep_dim=False)

		self.classifier = nn.Sequential()
		if 0.0 < self.classifier_dropout < 1.0:
			self.classifier.add_module(
				name="dropout", module=Dropout(p=self.classifier_dropout, inplace=True)
			)
		self.classifier.add_module(
			name="fc",
			module=LinearLayer(
				in_features=out_channels, out_features=self.num_classes, bias=True
			),
		)

		# weight initialization
		InitParaUtil.initialize_weights(self.temporal_conv)
		InitParaUtil.initialize_weights(self.classifier)

	def _init_cache(self) -> Dict[str, Optional[Tensor]]:
		"""Initialize a cache to store intermediate results at time step t so that they can be used at step t+1"""
		return dict(zip(self.os_keys, [None] * len(self.os_keys)))

	def _extract_mobilevit_features(
		self, layer, x: Tensor, cached_fm: Tensor
	) -> Tuple[Tensor, Tensor]:
		# mv2 block
		x = layer[0](x)
		# mobilevit block
		x, cached_fm = layer[1]((x, cached_fm))
		return x, cached_fm

	def _forward_cnn(self, x: Tensor) -> Tensor:
		x = self.mobilevit_model.head(x)
		x = self.mobilevit_model.body["layer1"](x)
		x = self.mobilevit_model.body["layer2"](x)
		return x

	def _forward_fn(
		self,
		x: Tensor,
		spatial_cache: Optional[Dict[str, Optional[Tensor]]] = None,
	) -> Tuple[Tensor, Dict[str, Tensor]]:
		x = self._forward_cnn(x)

		mvit_cache = dict()
		for os, layers_os in zip(
			self.os_keys,
			[
				self.mobilevit_model.body["layer3"],
				self.mobilevit_model.body["layer4"],
				self.mobilevit_model.body["layer5"],
			],
		):
			x, curr_patches = self._extract_mobilevit_features(
				layer=layers_os,
				x=x,
				cached_fm=spatial_cache[os].clone()
				if spatial_cache[os] is not None
				else None,
			)
			mvit_cache[os] = curr_patches

		return x, mvit_cache

	def _forward_train(self, x: Tensor) -> Tensor:
		assert x.dim() == 5, "Input should be 5-dimensional [B,N,C,H,W]"

		num_frames = x.shape[1]

		# run spatio_temporal model
		spatial_cache = self._init_cache()
		outputs = []
		for i in range(num_frames):
			out, spatial_cache = self._forward_fn(
				x=x[:, i], spatial_cache=spatial_cache
			)
			outputs.append(out)

		# aggregate representations
		# [B, C, H , W] x N --> [B, C, N, H, W]
		outputs = torch.stack(outputs, dim=2)
		outputs = self.temporal_conv(outputs)
		outputs = self.global_pool_layer(outputs)
		return self.classifier(outputs)

	def _forward_inference(self, x: Tensor) -> Tensor:
		assert x.dim() == 5, "Input should be 5-dimensional [B,N,C,H,W]"
		batch_size, num_frames = x.shape[:2]

		batch_outputs = torch.zeros(
			size=(batch_size, self.num_classes), dtype=x.dtype, device=x.device
		)
		for b in range(batch_size):

			# run spatio_temporal model
			spatial_cache = self._init_cache()
			frame_outputs = []
			for i in range(num_frames):
				# [1, C, H, W]
				out, spatial_cache = self._forward_fn(
					x=x[b, i].unsqueeze(0), spatial_cache=spatial_cache
				)
				frame_outputs.append(out)

			# Stack output of all frames
			frame_outputs = torch.stack(frame_outputs, dim=2)  # [1, C, N, H, W]

			# Temporal conv followed by global pool and classification
			frame_outputs = self.temporal_conv(frame_outputs)
			frame_outputs = self.global_pool_layer(frame_outputs)
			frame_outputs = self.classifier(frame_outputs)

			batch_outputs[b] = frame_outputs[0]

		return batch_outputs

	def forward(self, x: Tensor) -> Tensor:
		if self.inference_mode and not self.training:
			return self._forward_inference(x)
		else:
			return self._forward_train(x)
