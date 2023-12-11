from model.classification import BaseEncoder

from layer import ConvLayer2D, LinearLayer, GlobalPool, Identity, Dropout
from model.classification.cvnets.mobilenetv2 import InvertedResidual

from torch import nn, Tensor
from typing import Tuple, Dict, Optional, Union, Sequence

from utils.entry_utils import Entry
from utils.math_utils import Math
from layer.utils import InitParaUtil
from layer import ExtensionLayer
from utils import Util

from utils.type_utils import Prf, Par
import torch
import torch.nn.functional as F

import numpy as np
import math


@Entry.register_entry(Entry.Layer)
class LinearSelfAttention(ExtensionLayer):
	"""
	This layer applies a self-attention with linear complexity, as described in `this paper <>`_
	This layer can be used for self- as well as cross-attention.

	Args:
		embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
		attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
		add_bias (Optional[bool]): Use bias in learnable layers. Default: True

	Shape:
		- Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
		:math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
		- Output: same as the input

	.. note::
		For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
		in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
		we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
		expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
		channel-first to channel-last format in case of a linear layer.
	"""
	__slots__ = ["embed_dim", "attn_dropout_layer", "add_bias"]
	_disp_ = __slots__

	# defaults = [None, 0.0, True]

	def __init__(
		self,
		embed_dim: int,
		attn_dropout: Optional[float] = 0.0,
		add_bias: Optional[bool] = True,
	) -> None:
		super().__init__(**Par.purify(locals()))

		self.qkv_proj_layer = ConvLayer2D(
			in_channels=embed_dim,
			out_channels=1 + (2 * embed_dim),
			add_bias=add_bias,
			kernel_size=1,
			use_norm=False,
			use_act=False,
		)

		self.attn_dropout_layer = Dropout(p=attn_dropout)

		self.out_proj_layer = ConvLayer2D(
			in_channels=embed_dim,
			out_channels=embed_dim,
			add_bias=add_bias,
			kernel_size=1,
			use_norm=False,
			use_act=False,
		)

	@staticmethod
	def visualize_context_scores(context_scores):
		# [B, 1, P, N]
		batch_size, channels, num_pixels, num_patches = context_scores.shape

		assert batch_size == 1, "For visualization purposes, use batch size of 1"
		assert (
			channels == 1
		), "The inner-product between input and latent node (query) is a scalar"

		up_scale_factor = int(num_pixels ** 0.5)
		patch_h = patch_w = int(context_scores.shape[-1] ** 0.5)
		# [1, 1, P, N] --> [1, P, h, w]
		context_scores = context_scores.reshape(1, num_pixels, patch_h, patch_w)
		# Fold context scores [1, P, h, w] using pixel shuffle to obtain [1, 1, H, W]
		context_map = F.pixel_shuffle(context_scores, upscale_factor=up_scale_factor)
		# [1, 1, H, W] --> [H, W]
		context_map = context_map.squeeze()

		# For ease of visualization, we do min-max normalization
		min_val = torch.min(context_map)
		max_val = torch.max(context_map)
		context_map = (context_map - min_val) / (max_val - min_val)

		try:
			import cv2
			from glob import glob
			import os

			# convert from float to byte
			context_map = (context_map * 255).byte().cpu().numpy()
			context_map = cv2.resize(
				context_map, (80, 80), interpolation=cv2.INTER_NEAREST
			)

			colored_context_map = cv2.applyColorMap(context_map, cv2.COLORMAP_JET)
			# Lazy way to dump feature maps in attn_res folder. Make sure that directory is empty and copy
			# context maps before running on different image. Otherwise, attention maps will be overridden.
			res_dir_name = "attn_res"
			if not os.path.isdir(res_dir_name):
				os.makedirs(res_dir_name)
			f_name = "{}/h_{}_w_{}_index_".format(res_dir_name, patch_h, patch_w)

			files_cmap = glob(
				"{}/h_{}_w_{}_index_*.png".format(res_dir_name, patch_h, patch_w)
			)
			idx = len(files_cmap)
			f_name += str(idx)

			cv2.imwrite("{}.png".format(f_name), colored_context_map)
			return colored_context_map
		except ModuleNotFoundError as mnfe:
			print("Please install OpenCV to visualize context maps")
			return context_map

	def _forward_self_attn(self, x: Tensor) -> Tensor:
		# [B, C, P, N] --> [B, 1 + 2d, P, N]
		qkv = self.qkv_proj_layer(x)

		# Project x into query, key and value
		# Query --> [B, 1, P, N]
		# value, key --> [B, d, P, N]
		query, key, value = torch.split(
			qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
		)

		# apply softmax along N dimension
		context_scores = F.softmax(query, dim=-1)
		# Uncomment below line to visualize context scores
		# self.visualize_context_scores(context_scores=context_scores)
		context_scores = self.attn_dropout_layer(context_scores)

		# Compute context vector
		# [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
		context_vector = key * context_scores
		# [B, d, P, N] --> [B, d, P, 1]
		context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

		# combine context vector with values
		# [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
		out = F.relu(value) * context_vector.expand_as(value)
		out = self.out_proj_layer(out)
		return out

	def _forward_cross_attn(self, x: Tensor, x_prev: Optional[Tensor] = None) -> Tensor:
		# x --> [B, C, P, N]
		# x_prev = [B, C, P, M]

		batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

		q_patch_area, q_num_patches = x.shape[-2:]

		assert (
			kv_patch_area == q_patch_area
		), "The number of pixels in a patch for query and key_value should be the same"

		# compute query, key, and value
		# [B, C, P, M] --> [B, 1 + d, P, M]
		qk = F.conv2d(
			x_prev,
			weight=self.qkv_proj_layer.block.conv.weight[: self.embed_dim + 1, ...],
			bias=self.qkv_proj_layer.block.conv.bias[: self.embed_dim + 1, ...],
		)
		# [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
		query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
		# [B, C, P, N] --> [B, d, P, N]
		value = F.conv2d(
			x,
			weight=self.qkv_proj_layer.block.conv.weight[self.embed_dim + 1:, ...],
			bias=self.qkv_proj_layer.block.conv.bias[self.embed_dim + 1:, ...],
		)

		# apply softmax along M dimension
		context_scores = F.softmax(query, dim=-1)
		context_scores = self.attn_dropout_layer(context_scores)

		# compute context vector
		# [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
		context_vector = key * context_scores
		# [B, d, P, M] --> [B, d, P, 1]
		context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

		# combine context vector with values
		# [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
		out = F.relu(value) * context_vector.expand_as(value)
		out = self.out_proj_layer(out)
		return out

	def forward(self, x: Tensor, x_prev: Optional[Tensor] = None) -> Tensor:
		if x_prev is None:
			return self._forward_self_attn(x)
		else:
			return self._forward_cross_attn(x, x_prev=x_prev)

	def profile(self, x) -> Tuple[Tensor, float, float]:
		params = macs = 0.0

		qkv, p, m = Prf.profile_list(module_list=self.qkv_proj_layer, x=x)
		params += p
		macs += m

		query, key, value = torch.split(
			qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
		)

		if self.out_proj_layer is not None:
			out_p, p, m = Prf.profile_list(module_list=self.out_proj_layer, x=value)
			params += p
			macs += m

		return x, params, macs


@Entry.register_entry(Entry.Layer)
class LinearAttnFFN(ExtensionLayer):
	"""
	This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
	Args:
		embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
		ffn_latent_dim (int): Inner dimension of the FFN
		attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
		dropout (Optional[float]): Dropout rate. Default: 0.0
		ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
		norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

	Shape:
		- Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
			:math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
		- Output: same shape as the input
	"""
	__slots__ = ["embed_dim", "fnn_latent_dim", "attn_dropout", "ffn_dropout" "dropout", "norm_layer"]
	_disp_ = __slots__

	# defaults = [None, None, 0.0, 0.0, 0.1, "LayerNorm2D"]

	def __init__(
		self,
		embed_dim: int,
		ffn_latent_dim: int,
		attn_dropout: Optional[float] = 0.0,
		ffn_dropout: Optional[float] = 0.0,
		dropout: Optional[float] = 0.1,
		norm_layer: Optional[str] = "LayerNorm2D",
	) -> None:
		super().__init__(**Par.purify(locals()))
		attn_unit = LinearSelfAttention(
			embed_dim=embed_dim, attn_dropout=attn_dropout, add_bias=True
		)

		self.pre_norm_attn = nn.Sequential(
			Entry.get_entity(
				Entry.Norm, entry_name=norm_layer, num_channels=embed_dim
			),
			attn_unit,
			Dropout(p=dropout),
		)

		self.pre_norm_ffn = nn.Sequential(
			Entry.get_entity(
				Entry.Norm, entry_name=norm_layer, num_channels=embed_dim
			),
			ConvLayer2D(
				in_channels=embed_dim,
				out_channels=ffn_latent_dim,
				kernel_size=1,
				stride=1,
				bias=True,
				use_norm=False,
				use_act=True,
			),
			Dropout(p=ffn_dropout),
			ConvLayer2D(
				in_channels=ffn_latent_dim,
				out_channels=embed_dim,
				kernel_size=1,
				stride=1,
				bias=True,
				use_norm=False,
				use_act=False,
			),
			Dropout(p=dropout),
		)

		self.attn_fn_name = attn_unit.__repr__()
		self.norm_name = norm_layer

	def forward(
		self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
	) -> Tensor:
		if x_prev is None:
			# self-attention
			x = x + self.pre_norm_attn(x)
		else:
			# cross-attention
			res = x
			x = self.pre_norm_attn[0](x)  # norm
			x = self.pre_norm_attn[1](x, x_prev)  # attn
			x = self.pre_norm_attn[2](x)  # drop
			x = x + res  # residual

		# Feed forward network
		x = x + self.pre_norm_ffn(x)
		return x

	def profile(self, x: Tensor) -> Tuple[Tensor, float, float]:
		out, p_mha, m_mha = Prf.profile_list(module_list=self.pre_norm_attn, x=x)
		out, p_ffn, m_ffn = Prf.profile_list(module_list=self.pre_norm_ffn, x=x)

		macs = m_mha + m_ffn
		params = p_mha + p_ffn

		return x, params, macs


@Entry.register_entry(Entry.Layer)
class MobileViTBlockv2(ExtensionLayer):
	"""
	This class defines the `MobileViTv2 block <>`_

	Args:
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
		attn_unit_dim (int): Input dimension to the attention unit
		ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
		n_attn_blocks (Optional[int]): Number of attention units. Default: 2
		attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
		dropout (Optional[float]): Dropout rate. Default: 0.0
		ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
		patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
		patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
		kernel_size (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
		dilation (Optional[int]): Dilation rate in convolutions. Default: 1
		attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
	"""
	__slots__ = [
		"in_channels", "kernel_size", "dilation",
		"patch_h", "patch_w",
		"n_attn_blocks", "attn_unit_dim", "attn_norm_layer", "attn_dropout", "ffn_multiplier", "ffn_dropout", "dropout"
	]
	_disp_ = __slots__

	# defaults = [
	# 	None, 3, 1,
	# 	8, 8,
	# 	2, None, "LayerNorm2D", 0.0, 2.0, 0.0, 0.0
	# ]

	def __init__(
		self,
		in_channels: int,

		patch_h: int,
		patch_w: int,
		n_attn_blocks: int,
		attn_unit_dim: int,

		kernel_size: Optional[int] = 3,
		dilation: Optional[int] = 1,

		attn_dropout: Optional[float] = 0.0,
		attn_norm_layer: Optional[str] = "LayerNorm2D",
		ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
		ffn_dropout: Optional[float] = 0.0,
		dropout: Optional[float] = 0.0,
	) -> None:
		super().__init__(**Par.purify(locals()))

		cnn_out_dim = attn_unit_dim

		# --------- local representation ----------------
		# depth-wise convolution
		conv_3x3_in = ConvLayer2D(
			in_channels=in_channels,
			out_channels=in_channels,
			kernel_size=kernel_size,
			stride=1,
			use_norm=True,
			use_act=True,
			dilation=dilation,
			groups=in_channels,
		)
		# change number of channels : in_channels -> attn_unit_dim(cnn_out_dim)
		conv_1x1_in = ConvLayer2D(
			in_channels=in_channels,
			out_channels=cnn_out_dim,
			kernel_size=1,
			stride=1,
			use_norm=False,
			use_act=False,
		)
		self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

		self.global_rep, attn_unit_dim = self._build_attn_layer(
			d_model=attn_unit_dim,
			ffn_mult=ffn_multiplier,
			n_layers=n_attn_blocks,
			attn_dropout=attn_dropout,
			dropout=dropout,
			ffn_dropout=ffn_dropout,
			attn_norm_layer=attn_norm_layer,
		)

		# 1X1 convolution : change number of channels : cnn_out_dim -> in_channels
		self.conv_proj = ConvLayer2D(
			in_channels=cnn_out_dim,
			out_channels=in_channels,
			kernel_size=1,
			stride=1,
			use_norm=True,
			use_act=False,
		)

		self.patch_area = self.patch_w * self.patch_h

		self.cnn_in_dim = in_channels
		self.cnn_out_dim = cnn_out_dim
		self.transformer_in_dim = attn_unit_dim
		self.enable_coreml_compatible_fn = Util.is_coreml_conversion()

		if self.enable_coreml_compatible_fn:
			# we set persistent to false so that these weights are not part of model's state_dict
			self.register_buffer(
				name="unfolding_weights",
				tensor=self._compute_unfolding_weights(),
				persistent=False,
			)

	def _compute_unfolding_weights(self) -> Tensor:
		# [P_h * P_w, P_h * P_w]
		weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
		# [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
		weights = weights.reshape(
			(self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w)
		)
		# [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
		weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
		return weights

	def _build_attn_layer(
		self,
		d_model: int,
		ffn_mult: Union[Sequence, int, float],
		n_layers: int,
		attn_dropout: float,
		dropout: float,
		ffn_dropout: float,
		attn_norm_layer: str,
	) -> Tuple[nn.Module, int]:

		if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
			ffn_dims = (
				np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
			)
		elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
			ffn_dims = [ffn_mult[0] * d_model] * n_layers
		elif isinstance(ffn_mult, (int, float)):
			ffn_dims = [ffn_mult * d_model] * n_layers
		else:
			raise NotImplementedError

		# ensure that dims are multiple of 16
		ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

		global_rep = [
			LinearAttnFFN(
				embed_dim=d_model,
				ffn_latent_dim=ffn_dims[block_idx],
				attn_dropout=attn_dropout,
				dropout=dropout,
				ffn_dropout=ffn_dropout,
				norm_layer=attn_norm_layer,
			)
			for block_idx in range(n_layers)
		]
		global_rep.append(
			Entry.get_entity(Entry.Norm, entry_name=attn_norm_layer, num_features=d_model)
		)

		return nn.Sequential(*global_rep), d_model

	def __repr__(self) -> str:
		repr_str = "{}(".format(self.__class__.__name__)

		repr_str += "\n\t Local representations"
		if isinstance(self.local_rep, nn.Sequential):
			for m in self.local_rep:
				repr_str += "\n\t\t {}".format(m)
		else:
			repr_str += "\n\t\t {}".format(self.local_rep)

		repr_str += "\n\t Global representations with patch size of {}x{}".format(
			self.patch_h,
			self.patch_w,
		)
		if isinstance(self.global_rep, nn.Sequential):
			for m in self.global_rep:
				repr_str += "\n\t\t {}".format(m)
		else:
			repr_str += "\n\t\t {}".format(self.global_rep)

		if isinstance(self.conv_proj, nn.Sequential):
			for m in self.conv_proj:
				repr_str += "\n\t\t {}".format(m)
		else:
			repr_str += "\n\t\t {}".format(self.conv_proj)

		repr_str += "\n)"
		return repr_str

	def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

		batch_size, in_channels, img_h, img_w = feature_map.shape

		# [B, C, H, W] --> [B, C, P, N]
		patches = F.unfold(
			feature_map,
			kernel_size=(self.patch_h, self.patch_w),
			stride=(self.patch_h, self.patch_w),
		)
		patches = patches.reshape(
			batch_size, in_channels, self.patch_h * self.patch_w, -1
		)

		return patches, (img_h, img_w)

	def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
		batch_size, in_dim, patch_size, n_patches = patches.shape

		# [B, C, P, N]
		patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

		feature_map = F.fold(
			patches,
			output_size=output_size,
			kernel_size=(self.patch_h, self.patch_w),
			stride=(self.patch_h, self.patch_w),
		)

		return feature_map

	def unfolding_coreml(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
		# im2col is not implemented in Coreml, so here we hack its implementation using conv2d
		# we compute the weights

		# [B, C, H, W] --> [B, C, P, N]
		batch_size, in_channels, img_h, img_w = feature_map.shape
		#
		patches = F.conv2d(
			feature_map,
			self.unfolding_weights,
			bias=None,
			stride=(self.patch_h, self.patch_w),
			padding=0,
			dilation=1,
			groups=in_channels,
		)
		patches = patches.reshape(
			batch_size, in_channels, self.patch_h * self.patch_w, -1
		)
		return patches, (img_h, img_w)

	def folding_coreml(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
		# col2im is not supported on coreml, so tracing fails
		# We hack folding function via pixel_shuffle to enable coreml tracing
		batch_size, in_dim, patch_size, n_patches = patches.shape

		n_patches_h = output_size[0] // self.patch_h
		n_patches_w = output_size[1] // self.patch_w

		feature_map = patches.reshape(
			batch_size, in_dim * self.patch_h * self.patch_w, n_patches_h, n_patches_w
		)
		assert (
			self.patch_h == self.patch_w
		), "For Coreml, we need patch_h and patch_w are the same"
		feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
		return feature_map

	def resize_input_if_needed(self, x):
		batch_size, in_channels, orig_h, orig_w = x.shape
		if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
			new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
			new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
			x = F.interpolate(
				x, size=(new_h, new_w), mode="bilinear", align_corners=True
			)
		return x

	def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
		x = self.resize_input_if_needed(x)

		fm = self.local_rep(x)

		# convert feature map to patches
		if self.enable_coreml_compatible_fn:
			patches, output_size = self.unfolding_coreml(fm)
		else:
			patches, output_size = self.unfolding_pytorch(fm)

		# learn global representations on all patches
		patches = self.global_rep(patches)

		# [B x Patch x Patches x C] --> [B x C x Patches x Patch]
		if self.enable_coreml_compatible_fn:
			fm = self.folding_coreml(patches=patches, output_size=output_size)
		else:
			fm = self.folding_pytorch(patches=patches, output_size=output_size)
		fm = self.conv_proj(fm)

		return fm

	def forward_temporal(
		self, x: Tensor, x_prev: Tensor, *args, **kwargs
	) -> Union[Tensor, Tuple[Tensor, Tensor]]:
		x = self.resize_input_if_needed(x)

		fm = self.local_rep(x)

		# convert feature map to patches
		if self.enable_coreml_compatible_fn:
			patches, output_size = self.unfolding_coreml(fm)
		else:
			patches, output_size = self.unfolding_pytorch(fm)

		# learn global representations
		for global_layer in self.global_rep:
			if isinstance(global_layer, LinearAttnFFN):
				patches = global_layer(x=patches, x_prev=x_prev)
			else:
				patches = global_layer(patches)

		# [B x Patch x Patches x C] --> [B x C x Patches x Patch]
		if self.enable_coreml_compatible_fn:
			fm = self.folding_coreml(patches=patches, output_size=output_size)
		else:
			fm = self.folding_pytorch(patches=patches, output_size=output_size)
		fm = self.conv_proj(fm)

		return fm, patches

	def forward(
		self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
	) -> Union[Tensor, Tuple[Tensor, Tensor]]:
		if isinstance(x, Tuple) and len(x) == 2:
			# for spatio-temporal data (e.g., videos)
			return self.forward_temporal(x=x[0], x_prev=x[1])
		elif isinstance(x, Tensor):
			# for image data
			return self.forward_spatial(x)
		else:
			raise NotImplementedError

	def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
		params = macs = 0.0
		x = self.resize_input_if_needed(x)

		res = x
		out, p, m = Prf.profile_list(module_list=self.local_rep, x=x)
		params += p
		macs += m

		patches, output_size = self.unfolding_pytorch(feature_map=out)

		patches, p, m = Prf.profile_list(module_list=self.global_rep, x=patches)
		params += p
		macs += m

		fm = self.folding_pytorch(patches=patches, output_size=output_size)

		out, p, m = Prf.profile_list(module_list=self.conv_proj, x=fm)
		params += p
		macs += m

		return res, params, macs


def get_configuration(width_multiplier) -> Dict:

	ffn_multiplier = (
		2  # bound_fn(min_val=2.0, max_val=4.0, value=2.0 * width_multiplier)
	)
	mv2_exp_mult = 2  # max(1.0, min(2.0, 2.0 * width_multiplier))

	layer_0_dim = Math.bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
	layer_0_dim = int(Math.make_divisible(layer_0_dim, divisor=8, min_value=16))
	config = {
		"layer0": {
			"img_channels": 3,
			"out_channels": layer_0_dim,
		},
		# 128x128
		"layer1": {
			"out_channels": int(Math.make_divisible(64 * width_multiplier, divisor=16)),
			"expand_ratio": mv2_exp_mult,
			"num_blocks": 1,
			"stride": 1,
			"block_type": "mv2",
		},
		"layer2": {
			"out_channels": int(Math.make_divisible(128 * width_multiplier, divisor=8)),
			"expand_ratio": mv2_exp_mult,
			"num_blocks": 2,
			"stride": 2,
			"block_type": "mv2",
		},
		# 64x64
		"layer3": {
			"out_channels": int(Math.make_divisible(256 * width_multiplier, divisor=8)),
			"attn_unit_dim": int(Math.make_divisible(128 * width_multiplier, divisor=8)),
			"ffn_multiplier": ffn_multiplier,
			"attn_blocks": 2,
			"patch_h": 2,
			"patch_w": 2,
			"stride": 2,
			"mv_expand_ratio": mv2_exp_mult,
			"block_type": "mobilevit",
		},
		# 32x32
		"layer4": {  # 14x14
			"out_channels": int(Math.make_divisible(384 * width_multiplier, divisor=8)),
			"attn_unit_dim": int(Math.make_divisible(192 * width_multiplier, divisor=8)),
			"ffn_multiplier": ffn_multiplier,
			"attn_blocks": 4,
			"patch_h": 2,
			"patch_w": 2,
			"stride": 2,
			"mv_expand_ratio": mv2_exp_mult,
			"block_type": "mobilevit",
		},
		# 16x16
		"layer5": {  # 7x7
			"out_channels": int(Math.make_divisible(512 * width_multiplier, divisor=8)),
			"attn_unit_dim": int(Math.make_divisible(256 * width_multiplier, divisor=8)),
			"ffn_multiplier": ffn_multiplier,
			"attn_blocks": 3,
			"patch_h": 2,
			"patch_w": 2,
			"stride": 2,
			"mv_expand_ratio": mv2_exp_mult,
			"block_type": "mobilevit",
		},
		# 8x8
		"last_layer_exp_factor": 4,
	}

	return config


@Entry.register_entry(Entry.ClassificationModel)
class MobileViTv2(BaseEncoder):
	__slots__ = ["attn_dropout", "ffn_dropout", "dropout", "width_multiplier", "attn_norm_layer"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [0.0, 0.0, 0.0, 1.0, "LayerNorm2D"]
	_types_ = [float, float, float, float, str]

	def __init__(self) -> None:
		super().__init__()

		mobilevit_config = get_configuration(self.width_multiplier)
		image_channels = mobilevit_config["layer0"]["img_channels"]
		out_channels = mobilevit_config["layer0"]["out_channels"]

		# store model configuration in a dictionary
		self.model_conf_dict = dict()
		# 3X3 convolution with changing number of channels
		self.head = ConvLayer2D(
			in_channels=image_channels,
			out_channels=out_channels,
			kernel_size=3,
			stride=2,
			use_norm=True,
			use_act=True,
		)
		self.model_conf_dict["head"] = {"in": image_channels, "out": out_channels}

		self.body = nn.Sequential()
		in_channels = out_channels
		layer_1, out_channels = self._make_layer(
			input_channel=in_channels, cfg=mobilevit_config["layer1"]
		)
		self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}
		self.body.add_module(name="layer1", module=layer_1)

		in_channels = out_channels
		layer_2, out_channels = self._make_layer(
			input_channel=in_channels, cfg=mobilevit_config["layer2"]
		)
		self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}
		self.body.add_module(name="layer2", module=layer_2)

		in_channels = out_channels
		layer_3, out_channels = self._make_layer(
			input_channel=in_channels, cfg=mobilevit_config["layer3"]
		)
		self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}
		self.body.add_module(name="layer3", module=layer_3)

		in_channels = out_channels
		layer_4, out_channels = self._make_layer(
			input_channel=in_channels,
			cfg=mobilevit_config["layer4"],
			dilate=self.dilate_l4,
		)
		self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}
		self.body.add_module(name="layer4", module=layer_4)

		in_channels = out_channels
		layer_5, out_channels = self._make_layer(
			input_channel=in_channels,
			cfg=mobilevit_config["layer5"],
			dilate=self.dilate_l5,
		)
		self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}
		self.body.add_module(name="layer5", module=layer_5)

		self.tail = Identity()
		self.model_conf_dict["tail"] = {
			"in": out_channels,
			"out": out_channels,
		}

		# GlobalPool
		# Linear
		self.classifier = nn.Sequential(
			GlobalPool(pool_type=self.global_pool, keep_dim=False),
			LinearLayer(in_features=out_channels, out_features=self.n_classes, add_bias=True),
		)

		# weight initialization
		InitParaUtil.initialize_weights(self)

	def _make_layer(
		self, input_channel, cfg: Dict, dilate: Optional[bool] = False
	) -> Tuple[nn.Sequential, int]:
		block_type = cfg.get("block_type", "mobilevit")
		if block_type.lower() == "mobilevit":
			return self._make_mit_layer(
				input_channel=input_channel, cfg=cfg, dilate=dilate
			)
		else:
			return self._make_mobilenet_layer(
				input_channel=input_channel, cfg=cfg
			)

	@staticmethod
	def _make_mobilenet_layer(
		input_channel: int, cfg: Dict
	) -> Tuple[nn.Sequential, int]:
		output_channels = cfg.get("out_channels")
		num_blocks = cfg.get("num_blocks", 2)
		expand_ratio = cfg.get("expand_ratio", 4)
		block = []

		for i in range(num_blocks):
			stride = cfg.get("stride", 1) if i == 0 else 1

			layer = InvertedResidual(
				in_channels=input_channel,
				out_channels=output_channels,
				stride=stride,
				expand_ratio=expand_ratio,
			)
			block.append(layer)
			input_channel = output_channels
		return nn.Sequential(*block), input_channel

	def _make_mit_layer(
		self, input_channel, cfg: Dict, dilate: Optional[bool] = False
	) -> Tuple[nn.Sequential, int]:
		prev_dilation = self.dilation
		block = []
		stride = cfg.get("stride", 1)

		if stride == 2:
			if dilate:
				self.dilation *= 2
				stride = 1

			layer = InvertedResidual(
				in_channels=input_channel,
				out_channels=cfg.get("out_channels"),
				stride=stride,
				expand_ratio=cfg.get("mv_expand_ratio", 4),
				dilation=prev_dilation,
			)

			block.append(layer)
			input_channel = cfg.get("out_channels")

		attn_unit_dim = cfg["attn_unit_dim"]
		ffn_multiplier = cfg.get("ffn_multiplier")

		block.append(
			MobileViTBlockv2(
				in_channels=input_channel,
				attn_unit_dim=attn_unit_dim,
				ffn_multiplier=ffn_multiplier,
				n_attn_blocks=cfg.get("attn_blocks", 1),
				patch_h=cfg.get("patch_h", 2),
				patch_w=cfg.get("patch_w", 2),
				dropout=self.dropout,
				ffn_dropout=self.ffn_dropout,
				attn_dropout=self.attn_dropout,
				kernel_size=3,
				attn_norm_layer=self.attn_norm_layer,
				dilation=self.dilation,
			)
		)

		return nn.Sequential(*block), input_channel
