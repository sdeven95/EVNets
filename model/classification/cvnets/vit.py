import torch
from torch import nn, Tensor

from model.classification import BaseEncoder
from layer import ConvLayer2D, LinearLayer, SinusoidalPositionalEncoding, LearnablePositionEncoding, TransformerEncoder

from layer.utils import InitParaUtil
from utils.logger import Logger
from utils.entry_utils import Entry

from typing import Dict


def get_configuration(mode, dropout) -> Dict:
	vit_config = dict()
	if mode == "tiny":
		vit_config = {
			"patch_size": 16,
			"embed_dim": 192,
			"n_transformer_layers": 12,
			"n_attn_heads": 3,
			"ffn_dim": 192 * 4,
			"norm_layer": "layer_norm",
			"pos_emb_drop_p": 0.1,
			"attn_dropout": 0.0,
			"ffn_dropout": 0.0,
			"dropout": dropout,
		}
	elif mode == "small":
		vit_config = {
			"patch_size": 16,
			"embed_dim": 384,
			"n_transformer_layers": 12,
			"n_attn_heads": 6,
			"ffn_dim": 384 * 4,
			"norm_layer": "layer_norm",
			"pos_emb_drop_p": 0.0,
			"attn_dropout": 0.0,
			"ffn_dropout": 0.0,
			"dropout": dropout,
		}
	else:
		Logger.error("Not supported")
	return vit_config


@Entry.register_entry(Entry.ClassificationModel)
class VisionTransformer(BaseEncoder):
	"""
	This class defines the `Vision Transformer architecture <https://arxiv.org/abs/2010.11929>`_

	.. note::
		Our implementation is different from the original implementation in two ways:
		1. Kernel size is odd.
		2. Use sinusoidal positional encoding, allowing us to use ViT with any input size
		3. Do not use DropoutPath
	"""

	__slots__ = ["mode", "dropout", "vocab_size", "learnable_pos_emb", "no_cls_token"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["tiny", 0.0, 1000, False, False]
	_types_ = [str, float, int, bool, bool]

	def __init__(self) -> None:
		super().__init__()
		image_channels = 3

		vit_config = get_configuration(self.mode, self.dropout)

		patch_size = vit_config["patch_size"]
		embed_dim = vit_config["embed_dim"]
		ffn_dim = vit_config["ffn_dim"]
		pos_emb_drop_p = vit_config["pos_emb_drop_p"]
		n_transformer_layers = vit_config["n_transformer_layers"]
		num_heads = vit_config["n_attn_heads"]
		attn_dropout = vit_config["attn_dropout"]
		dropout = vit_config["dropout"]
		ffn_dropout = vit_config["ffn_dropout"]
		norm_layer = vit_config["norm_layer"]

		kernel_size = patch_size
		if patch_size % 2 == 0:
			kernel_size += 1

		self.patch_emb = ConvLayer2D(
			in_channels=image_channels,
			out_channels=embed_dim,
			kernel_size=kernel_size,
			stride=patch_size,
			bias=True,
			use_norm=False,
			use_act=False,
		)

		use_cls_token = not self.no_cls_token
		if use_cls_token:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		else:
			self.cls_token = None

		transformer_blocks = [
			TransformerEncoder(
				embed_dim=embed_dim,
				ffn_latent_dim=ffn_dim,
				num_heads=num_heads,
				attn_dropout=attn_dropout,
				dropout=dropout,
				ffn_dropout=ffn_dropout,
				transformer_norm_layer=norm_layer,
			)
			for _ in range(n_transformer_layers)
		]
		transformer_blocks.append(
			Entry.get_entity(Entry.Norm, entry_name=norm_layer, num_features=embed_dim)
		)

		self.transformer = nn.Sequential(*transformer_blocks)
		self.classifier = LinearLayer(embed_dim, self.num_classes)

		InitParaUtil.initialize_weights(self)

		if self.cls_token is not None:
			torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

		if self.learnable_pos_emb:
			self.pos_embed = LearnablePositionEncoding(
				num_embeddings=self.vocab_size,
				embed_dim=embed_dim,
				dropout=pos_emb_drop_p,
				channels_last=True,
			)
			nn.init.normal_(
				self.pos_embed.pos_emb.weight, mean=0, std=embed_dim ** -0.5
			)
		else:
			self.pos_embed = SinusoidalPositionalEncoding(
				d_model=embed_dim,
				dropout=pos_emb_drop_p,
				channels_last=True,
				max_len=self.vocab_size,
			)

	def extract_patch_embeddings(self, x: Tensor) -> Tensor:
		# x -> [B, C, H, W]
		B_ = x.shape[0]

		# [B, C, H, W] --> [B, C, n_h, n_w]
		patch_emb = self.patch_emb(x)
		# [B, C, n_h, n_w] --> [B, C, N]
		patch_emb = patch_emb.flatten(2)
		# [B, C, N] --> [B, N, C]
		patch_emb = patch_emb.transpose(1, 2).contiguous()

		# add classification token
		if self.cls_token is not None:
			cls_tokens = self.cls_token.expand(B_, -1, -1)
			patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

		patch_emb = self.pos_embed(patch_emb)
		return patch_emb

	def forward(self, x: Tensor) -> Tensor:
		x = self.extract_patch_embeddings(x)

		x = self.transformer(x)

		# grab the first token and classify
		if self.cls_token is not None:
			x = self.classifier(x[:, 0])
		else:
			x = torch.mean(x, dim=1)
			x = self.classifier(x)
		return x

	def profile_model(self, x: Tensor, *args, **kwargs) -> None:
		Logger.log("Model statistics for an input of size {}".format(x.size()))
		Logger.double_dash_line(dashes=65)
		print("{:>35} Summary".format(self.__class__.__name__))
		Logger.double_dash_line(dashes=65)

		out_dict = {}
		overall_params, overall_macs = 0.0, 0.0
		patch_emb, overall_params, overall_macs = self._profile_layers(
			self.patch_emb,
			x=x,
			overall_params=overall_params,
			overall_macs=overall_macs,
		)
		patch_emb = patch_emb.flatten(2)

		# [B, C, N] --> [B, N, C]
		patch_emb = patch_emb.transpose(1, 2)

		if self.cls_token is not None:
			# add classification token
			cls_tokens = self.cls_token.expand(patch_emb.shape[0], -1, -1)
			patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

		patch_emb, overall_params, overall_macs = self._profile_layers(
			self.transformer,
			x=patch_emb,
			overall_params=overall_params,
			overall_macs=overall_macs,
		)

		patch_emb, overall_params, overall_macs = self._profile_layers(
			self.classifier,
			x=patch_emb[:, 0],
			overall_params=overall_params,
			overall_macs=overall_macs,
		)

		Logger.double_dash_line(dashes=65)
		print("{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6))
		# Counting Addition and Multiplication as 1 operation
		print("{:<20} = {:>8.3f} M".format("Overall MACs", overall_macs / 1e6))
		overall_params_py = sum([p.numel() for p in self.parameters()])
		print(
			"{:<20} = {:>8.3f} M".format(
				"Overall parameters (sanity check)", overall_params_py / 1e6
			)
		)
		Logger.double_dash_line(dashes=65)
