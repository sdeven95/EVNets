from torch import nn, Tensor
import torch.nn.functional as F
import torch

from typing import Dict, Tuple, Optional, Union


from utils.logger import Logger
from utils.entry_utils import Entry
from model.classification import BaseEncoder

from layer import ConvLayer2D, LinearLayer, GlobalPool, Dropout, TransformerEncoder
from model.classification.cvnets.mobilenetv2 import InvertedResidual
from layer.utils import InitParaUtil
from layer import ExtensionLayer

from utils.type_utils import Prf, Par
import math


@Entry.register_entry(Entry.Layer)
class MobileViTBlock(ExtensionLayer):
	"""
	This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

	Args:
		in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
		transformer_dim (int): Input dimension to the transformer unit
		ffn_latent_dim (int): Dimension of the FFN block
		n_transformer_blocks (Optional[int]): Number of transformer blocks. Default: 2
		head_dim (Optional[int]): Head dimension in the multi-head attention. Default: 32
		attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
		dropout (Optional[float]): Dropout rate. Default: 0.0
		ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
		patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
		patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
		transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
		kernel_size (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
		dilation (Optional[int]): Dilation rate in convolutions. Default: 1
		no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
	"""
	__slots__ = [
		"in_channels", "kernel_size", "dilation", "no_fusion",
		"patch_h", "patch_w", "n_transformer_blocks", "transformer_dim", "transformer_norm_layer",
		"head_dim", "ffn_latent_dim", "attn_dropout", "ffn_dropout", "dropout"
	]
	_disp_ = __slots__

	# defaults = [
	# 	None, 3, 1, False,
	# 	8, 8, 2, None, "LayerNorm",
	# 	32, None, 0.0, 0.0, 0.0
	# ]

	def __init__(
		self,
		in_channels: int,
		patch_h: Optional[int],
		patch_w: Optional[int],
		kernel_size: Optional[int] = 3,
		dilation: Optional[int] = 1,
		no_fusion: Optional[bool] = False,

		n_transformer_blocks: Optional[int] = None,
		transformer_dim: int = None,
		transformer_norm_layer: Optional[str] = "LayerNorm",

		head_dim: Optional[int] = None,
		ffn_latent_dim: int = None,
		attn_dropout: Optional[float] = None,
		ffn_dropout: Optional[int] = None,
		dropout: Optional[int] = None,
	) -> None:
		super().__init__(**Par.purify(locals()))
		# ----- local representations -------
		conv_3x3_in = ConvLayer2D(
			in_channels=in_channels,
			out_channels=in_channels,
			kernel_size=kernel_size,
			stride=1,
			use_norm=True,
			use_act=True,
			dilation=dilation,
		)
		# n_channels: in_channels => transformer_dim
		conv_1x1_in = ConvLayer2D(
			in_channels=in_channels,
			out_channels=transformer_dim,
			kernel_size=1,
			stride=1,
			use_norm=False,
			use_act=False,
		)

		self.local_rep = nn.Sequential()
		self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
		self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

		assert transformer_dim % head_dim == 0
		num_heads = transformer_dim // head_dim

		# -------- global representations ----------
		global_rep = [
			TransformerEncoder(
				embed_dim=transformer_dim,
				num_heads=num_heads,
				transformer_norm_layer=transformer_norm_layer,
				ffn_latent_dim=ffn_latent_dim,

				attn_dropout=attn_dropout,
				ffn_dropout=ffn_dropout,
				dropout=dropout,
			)
			for _ in range(n_transformer_blocks)
		]
		# Note: Add a normalization Layer for transformation!!!
		global_rep.append(
			Entry.get_entity(Entry.Norm, entry_name=transformer_norm_layer)
		)
		self.global_rep = nn.Sequential(*global_rep)

		# for output : change number of channel to original, conv_1x1
		self.conv_proj = ConvLayer2D(
			in_channels=transformer_dim,
			out_channels=in_channels,
			kernel_size=1,
			stride=1,
			use_norm=True,
			use_act=True,
		)

		# if fusion with tensor before transformation,
		# add a convolution layer to change number of channels(with 3X3 kernel)
		# conv_3x3
		self.fusion = None
		if not no_fusion:
			self.fusion = ConvLayer2D(
				in_channels=2 * in_channels,
				out_channels=in_channels,
				kernel_size=kernel_size,
				stride=1,
				use_norm=True,
				use_act=True,
			)

		self.patch_area = self.patch_w * self.patch_h

		self.cnn_in_dim = in_channels
		self.cnn_out_dim = transformer_dim
		self.num_heads = num_heads

	def __repr__(self) -> str:
		repr_str = "{}(".format(self.__class__.__name__)

		repr_str += "\n\t Local representations"
		if isinstance(self.local_rep, nn.Sequential):
			for m in self.local_rep:
				repr_str += "\n\t\t {}".format(m)
		else:
			repr_str += "\n\t\t {}".format(self.local_rep)

		repr_str += "\n\t Global representations with patch size of {}x{}".format(
			self.patch_h, self.patch_w
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

		if self.fusion is not None:
			repr_str += "\n\t Feature fusion"
			if isinstance(self.fusion, nn.Sequential):
				for m in self.fusion:
					repr_str += "\n\t\t {}".format(m)
			else:
				repr_str += "\n\t\t {}".format(self.fusion)

		repr_str += "\n)"
		return repr_str

	# convert feature map (N, C, H, W) to patches
	def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
		patch_w, patch_h = self.patch_w, self.patch_h
		patch_area = int(patch_w * patch_h)
		batch_size, in_channels, orig_h, orig_w = feature_map.shape

		# make sure they could be evenly divided by patch sizes
		new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
		new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

		# change to target size
		interpolate = False
		if new_w != orig_w or new_h != orig_h:
			# Note: Padding can be done, but then it needs to be handled in attention function.
			feature_map = F.interpolate(
				feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
			)
			interpolate = True

		# number of patches along width and height
		num_patch_w = new_w // patch_w  # n_w
		num_patch_h = new_h // patch_h  # n_h
		num_patches = num_patch_h * num_patch_w  # N

		# [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
		reshaped_fm = feature_map.reshape(
			batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
		)
		# [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
		transposed_fm = reshaped_fm.transpose(1, 2)
		# [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
		reshaped_fm = transposed_fm.reshape(
			batch_size, in_channels, num_patches, patch_area
		)
		# [B, C, N, P] --> [B, P, N, C]
		transposed_fm = reshaped_fm.transpose(1, 3)
		# [B, P, N, C] --> [BP, N, C]
		patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

		info_dict = {
			"orig_size": (orig_h, orig_w),
			"batch_size": batch_size,  # B
			"interpolate": interpolate,
			"total_patches": num_patches,  # N
			"num_patches_w": num_patch_w,
			"num_patches_h": num_patch_h,
		}

		return patches, info_dict

	# convert patches to feature map
	def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
		n_dim = patches.dim()
		assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
			patches.shape
		)
		# [BP, N, C] --> [B, P, N, C]
		patches = patches.contiguous().view(
			info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
		)

		batch_size, pixels, num_patches, channels = patches.size()
		num_patch_h = info_dict["num_patches_h"]
		num_patch_w = info_dict["num_patches_w"]

		# [B, P, N, C] --> [B, C, N, P]
		patches = patches.transpose(1, 3)

		# [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
		feature_map = patches.reshape(
			batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w
		)
		# [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
		feature_map = feature_map.transpose(1, 2)
		# [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
		feature_map = feature_map.reshape(
			batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
		)
		if info_dict["interpolate"]:
			feature_map = F.interpolate(
				feature_map,
				size=info_dict["orig_size"],
				mode="bilinear",
				align_corners=False,
			)
		return feature_map

	def forward_spatial(self, x: Tensor) -> Tensor:
		res = x

		# local representations
		# 3X3 convolution(holding number of channel)
		# -> 1x1 convolution(change number of channel: in_channels -> transformer_dim
		fm = self.local_rep(x)

		# convert feature map to patches
		patches, info_dict = self.unfolding(fm)

		# learn global representations'
		# transformer: normalization-> attention -> dropout -> addition
		#              ->ffn(normalization -> Linear(Change number of channel to ffnLatent_dim)
		#                    -> Activation -> Dropout -> Linear(Change number of channel back)->dropout)
		#              -> addition
		# normalization
		for transformer_layer in self.global_rep:
			patches = transformer_layer(patches)

		# [B x Patch x Patches x C] --> [B x C x Patches x Patch]
		fm = self.folding(patches=patches, info_dict=info_dict)

		# 1X1 convolution(change number of channels: transformer_dim -> in_channels)
		fm = self.conv_proj(fm)

		# if fusion, 3X3 convolution(change number of channels : 2 * in_channels -> in_channels)
		if self.fusion is not None:
			fm = self.fusion(torch.cat((res, fm), dim=1))
		return fm

	def forward_temporal(
		self, x: Tensor, x_prev: Optional[Tensor] = None
	) -> Union[Tensor, Tuple[Tensor, Tensor]]:

		res = x
		fm = self.local_rep(x)

		# convert feature map to patches
		patches, info_dict = self.unfolding(fm)

		# learn global representations
		for global_layer in self.global_rep:
			# if transformer, consider self-attention and cross-attention
			if isinstance(global_layer, TransformerEncoder):
				patches = global_layer(x=patches, x_prev=x_prev)
			else:
				patches = global_layer(patches)

		# [B x Patch x Patches x C] --> [B x C x Patches x Patch]
		fm = self.folding(patches=patches, info_dict=info_dict)

		fm = self.conv_proj(fm)

		if self.fusion is not None:
			fm = self.fusion(torch.cat((res, fm), dim=1))
		return fm, patches

	def forward(
		self, x: Union[Tensor, Tuple[Tensor]],
	) -> Union[Tensor, Tuple[Tensor, Tensor]]:
		if isinstance(x, Tuple) and len(x) == 2:
			# for spatio-temporal MobileViT
			return self.forward_temporal(x=x[0], x_prev=x[1])
		elif isinstance(x, Tensor):
			# For image data
			return self.forward_spatial(x)
		else:
			raise NotImplementedError

	def profile(self, x: Tensor) -> Tuple[Tensor, float, float]:
		params = macs = 0.0

		res = x
		out, p, m = Prf.profile_list(module_list=self.local_rep, x=x)
		params += p
		macs += m

		patches, info_dict = self.unfolding(feature_map=out)

		patches, p, m = Prf.profile_list(module_list=self.global_rep, x=patches)
		params += p
		macs += m

		fm = self.folding(patches=patches, info_dict=info_dict)

		out, p, m = Prf.profile_list(module_list=self.conv_proj, x=fm)
		params += p
		macs += m

		if self.fusion is not None:
			out, p, m = Prf.profile_list(
				module_list=self.fusion, x=torch.cat((out, res), dim=1)
			)
			params += p
			macs += m

		return res, params, macs


def get_configuration(mode, head_dim, num_heads) -> Dict:
	if mode == "xx_small":
		mv2_exp_mult = 2
		config = {
			"layer1": {
				"out_channels": 16,
				"expand_ratio": mv2_exp_mult,
				"num_blocks": 1,
				"stride": 1,
				"block_type": "mv2",
			},
			"layer2": {
				"out_channels": 24,
				"expand_ratio": mv2_exp_mult,
				"num_blocks": 3,
				"stride": 2,
				"block_type": "mv2",
			},
			"layer3": {  # 28x28
				"out_channels": 48,
				"transformer_channels": 64,
				"ffn_dim": 128,
				"transformer_blocks": 2,
				"patch_h": 2,  # 8,
				"patch_w": 2,  # 8,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"layer4": {  # 14x14
				"out_channels": 64,
				"transformer_channels": 80,
				"ffn_dim": 160,
				"transformer_blocks": 4,
				"patch_h": 2,  # 4,
				"patch_w": 2,  # 4,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"layer5": {  # 7x7
				"out_channels": 80,
				"transformer_channels": 96,
				"ffn_dim": 192,
				"transformer_blocks": 3,
				"patch_h": 2,
				"patch_w": 2,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"last_layer_exp_factor": 4,
		}
	elif mode == "x_small":
		mv2_exp_mult = 4
		config = {
			"layer1": {
				"out_channels": 32,
				"expand_ratio": mv2_exp_mult,
				"num_blocks": 1,
				"stride": 1,
				"block_type": "mv2",
			},
			"layer2": {
				"out_channels": 48,
				"expand_ratio": mv2_exp_mult,
				"num_blocks": 3,
				"stride": 2,
				"block_type": "mv2",
			},
			"layer3": {  # 28x28
				"out_channels": 64,
				"transformer_channels": 96,
				"ffn_dim": 192,
				"transformer_blocks": 2,
				"patch_h": 2,
				"patch_w": 2,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"layer4": {  # 14x14
				"out_channels": 80,
				"transformer_channels": 120,
				"ffn_dim": 240,
				"transformer_blocks": 4,
				"patch_h": 2,
				"patch_w": 2,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"layer5": {  # 7x7
				"out_channels": 96,
				"transformer_channels": 144,
				"ffn_dim": 288,
				"transformer_blocks": 3,
				"patch_h": 2,
				"patch_w": 2,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"last_layer_exp_factor": 4,
		}
	elif mode == "small":
		mv2_exp_mult = 4
		config = {
			"layer1": {
				"out_channels": 32,
				"expand_ratio": mv2_exp_mult,
				"num_blocks": 1,
				"stride": 1,
				"block_type": "mv2",
			},
			"layer2": {
				"out_channels": 64,
				"expand_ratio": mv2_exp_mult,
				"num_blocks": 3,
				"stride": 2,
				"block_type": "mv2",
			},
			"layer3": {  # 28x28
				"out_channels": 96,
				"transformer_channels": 144,
				"ffn_dim": 288,
				"transformer_blocks": 2,
				"patch_h": 2,
				"patch_w": 2,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"layer4": {  # 14x14
				"out_channels": 128,
				"transformer_channels": 192,
				"ffn_dim": 384,
				"transformer_blocks": 4,
				"patch_h": 2,
				"patch_w": 2,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"layer5": {  # 7x7
				"out_channels": 160,
				"transformer_channels": 240,
				"ffn_dim": 480,
				"transformer_blocks": 3,
				"patch_h": 2,
				"patch_w": 2,
				"stride": 2,
				"mv_expand_ratio": mv2_exp_mult,
				"head_dim": head_dim,
				"num_heads": num_heads,
				"block_type": "mobilevit",
			},
			"last_layer_exp_factor": 4,
		}
	else:
		raise NotImplementedError

	return config


@Entry.register_entry(Entry.ClassificationModel)
class MobileViT(BaseEncoder):
	"""
	This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
	"""
	__slots__ = [
		"mode",
		"attn_dropout", "ffn_dropout", "dropout",
		"transformer_norm_layer", "no_fuse_local_global_features",
		"conv_kernel_size", "head_dim", "number_heads"
	]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
				"small",
				0.0, 0.0, 0.0,
				"LayerNorm", False,
				3, None, None
			]
	_types_ = [
				str,
				float, float, float,
				str, bool,
				int, int, int
			]

	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

		image_channels = 3
		out_channels = 16

		mobilevit_config = get_configuration(self.mode, self.head_dim, self.number_heads)

		# store model configuration in a dictionary
		self.model_conf_dict = dict()
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

		in_channels = out_channels
		exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
		self.tail = ConvLayer2D(
			in_channels=in_channels,
			out_channels=exp_channels,
			kernel_size=1,
			stride=1,
			use_act=True,
			use_norm=True,
		)

		self.model_conf_dict["tail"] = {
			"in": in_channels,
			"out": exp_channels,
		}

		# Global Pool
		self.classifier = nn.Sequential()
		self.classifier.add_module(
			name="global_pool", module=GlobalPool(pool_type=self.global_pool, keep_dim=False)
		)
		# Dropout
		if 0.0 < self.classifier_dropout < 1.0:
			self.classifier.add_module(
				name="dropout", module=Dropout(p=self.classifier_dropout, inplace=True)
			)
		# Linear
		self.classifier.add_module(
			name="fc",
			module=LinearLayer(
				in_features=exp_channels, out_features=self.n_classes, add_bias=True
			),
		)

		# weight initialization
		InitParaUtil.initialize_weights(self)

	def _make_layer(
		self,
		input_channel,
		cfg: Dict,
		dilate: Optional[bool] = False,
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
		input_channel: int, cfg: Dict,
	) -> Tuple[nn.Sequential, int]:
		output_channels = cfg.get("out_channels")
		num_blocks = cfg.get("num_blocks", 2)
		expand_ratio = cfg.get("expand_ratio", 4)
		block = []

		for i in range(num_blocks):
			# stride only affect the first block not other blocks
			stride = cfg.get("stride", 1) if i == 0 else 1

			# 1X1 convolution : change channel size : in_channels -> hidden_dim
			#    optional, hidden_dim = in_channels * expand_ratio
			# 3X3 convolution:  depth-wise, holding channel size
			#    if first block and stride = 2, down sampling
			# 1X1 convolution: change channel size : hidden_dim -> out_channels
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
		self,
		input_channel,
		cfg: Dict,
		dilate: Optional[bool] = False,
	) -> Tuple[nn.Sequential, int]:
		prev_dilation = self.dilation
		block = []
		stride = cfg.get("stride", 1)

		# Note : if stride = 2, Add a mv2 block !!!!!!!!!!!
		# if dilate = True change dilation instead of stride
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

		# MobileVitBlock block
		# configurations for transformation
		head_dim = cfg.get("head_dim", 32)  # dim count per head
		transformer_dim = cfg["transformer_channels"]
		ffn_dim = cfg.get("ffn_dim")
		if head_dim is None:
			num_heads = cfg.get("num_heads", 4)  # head count
			if num_heads is None:
				num_heads = 4
			head_dim = transformer_dim // num_heads

		if transformer_dim % head_dim != 0:
			Logger.error(
				"Transformer input dimension should be divisible by head dimension. "
				"Got {} and {}.".format(transformer_dim, head_dim)
			)

		block.append(
			MobileViTBlock(
				in_channels=input_channel,  # original Channel
				transformer_dim=transformer_dim,  # dim used by transformation, by convolution : input_channel -> transform_dim
				ffn_latent_dim=ffn_dim,  # Feed Forward Net's hidden dimension size
				n_transformer_blocks=cfg.get("transformer_blocks", 1),  # Add several block continuously
				patch_h=cfg.get("patch_h", 2),  # patch height
				patch_w=cfg.get("patch_w", 2),  # patch width
				dropout=self.dropout,  # dropout rate for last step of transformation and FFN
				ffn_dropout=self.ffn_dropout,  # dropout rate for FFN
				attn_dropout=self.attn_dropout,  # dropout rate for attention
				head_dim=head_dim,  # dimension size per head
				no_fusion=self.no_fuse_local_global_features,  # if true, concat handled tensor with original tensor
				kernel_size=self.conv_kernel_size,  # 3X3 convolution kernel size, default 3
			)
		)

		return nn.Sequential(*block), input_channel
