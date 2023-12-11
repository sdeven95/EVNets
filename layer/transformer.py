
from utils.type_utils import Par, Prf
from utils.entry_utils import Entry
from utils import Util

from . import ExtensionLayer, LinearLayer, Dropout

from torch import nn, Tensor
import torch
from torch.nn import functional as F

from typing import Optional, Tuple


@Entry.register_entry(Entry.Layer)
class TransformerEncoder(ExtensionLayer):
	"""
	This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
	Args:
		embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
		ffn_latent_dim (int): Inner dimension of the FFN
		num_heads (Optional[int]) : Number of heads in multi-head attention. Default: 8
		attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
		dropout (Optional[float]): Dropout rate. Default: 0.0
		ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
		transformer_norm_layer (Optional[str]): Normalization layer. Default: layer_norm

	Shape:
		- Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
		and :math:`C_{in}` is input embedding dim
		- Output: same shape as the input
	"""
	__slots__ = [
		"embed_dim", "num_heads", "attn_dropout",
		"ffn_latent_dim", "fnn_dropout", "transformer_norm_layer",
		"dropout",
		"attn_fn_name", "act_fn_name"
	]
	_disp_ = __slots__

	# defaults = [
	# 	None, 8, 0.0,
	# 	None, 0.0, "LayerNorm",
	# 	0.0
	# ],

	def __init__(
		self,
		embed_dim: int,
		ffn_latent_dim: int,
		num_heads: Optional[int] = 8,
		attn_dropout: Optional[float] = 0.0,

		ffn_dropout: Optional[float] = 0.0,
		transformer_norm_layer: Optional[str] = "LayerNorm",

		dropout: Optional[float] = None,
	) -> None:

		super().__init__(**Par.purify(locals()))

		coreml_compatible = Util.is_coreml_conversion()

		# attention net
		self.pre_norm_mha = nn.Sequential(
			Entry.get_entity(Entry.Norm, entry_name=transformer_norm_layer),
			SingleHeadAttention(embed_dim=embed_dim, attn_dropout=attn_dropout, add_bias=True)
			if num_heads <= 1
			else MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_dropout=attn_dropout, coreml_compatible=coreml_compatible),
			Dropout(p=dropout),
		)

		# feed forward net
		act_layer = Entry.get_entity(Entry.Activation)
		self.pre_norm_ffn = nn.Sequential(
			Entry.get_entity(Entry.Norm, entry_name=transformer_norm_layer),
			LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, add_bias=True),
			act_layer,
			Dropout(p=ffn_dropout),
			LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
			Dropout(p=dropout),
		)
		self.attn_fn_name = "MultiHeadAttention" if num_heads > 1 else "SingleHeadAttention"
		self.act_fn_name = act_layer.__class__.__name__

	def forward(
		self, x: Tensor, x_prev: Optional[Tensor] = None
	) -> Tensor:

		# Multi-head attention
		res = x
		x = self.pre_norm_mha[0](x)  # normalization
		x = self.pre_norm_mha[1](x_q=x, x_kv=x_prev)  # multi_head attention(self or cross)
		x = self.pre_norm_mha[2](x)  # dropout
		x = x + res  # residual adjustment (addition)

		# Feed forward network & residual adjustment
		# FFN: normalization -> Linear(Change number of channel to ffnLatent_dim)
		# -> Activation -> Dropout -> Linear(Change number of channel back)->dropout
		x = x + self.pre_norm_ffn(x)
		return x

	def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
		b_sz, seq_len = x.shape[:2]  # N P

		out, p_mha, m_mha = Prf.profile_list(module_list=self.pre_norm_mha, x=x)

		out, p_ffn, m_ffn = Prf.profile_list(module_list=self.pre_norm_ffn, x=x)
		m_ffn = m_ffn * b_sz * seq_len

		macs = m_mha + m_ffn
		params = p_mha + p_ffn

		return x, params, macs


@Entry.register_entry(Entry.Layer)
class SingleHeadAttention(ExtensionLayer):
	"""
	This layer applies a single-head attention as described in `DeLighT <https://arxiv.org/abs/2008.00623>`_ paper

	Args:
		embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
		attn_dropout (Optional[float]): Attention dropout. Default: 0.0
		add_bias (Optional[bool]): Use bias or not. Default: ``True``

	Shape:
		- Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
		and :math:`C_{in}` is input embedding dim
		- Output: same shape as the input

	"""

	__slots__ = ["embed_dim", "att_dropout", "add_bias"]
	_disp_ = __slots__

	# defaults = [None, 0.0, True]

	def __init__(
		self,
		embed_dim: int,
		attn_dropout: float = 0.0,
		add_bias: Optional[bool] = True,
	) -> None:
		super().__init__(**Par.purify(locals()))

		# for generating Q, K, V, both are [embed_dim, embed_dim]
		self.qkv_proj_layer = LinearLayer(
			in_features=embed_dim, out_features=3 * embed_dim, add_bias=add_bias
		)

		self.scaling = self.embed_dim ** -0.5

		self.softmax = nn.Softmax(dim=-1)
		# attention dropout layer
		self.attn_dropout_layer = Dropout(p=attn_dropout)

		# attached out linear layer, necessary?
		self.out_proj = LinearLayer(
			in_features=embed_dim, out_features=embed_dim, add_bias=add_bias
		)

	def forward(self, x: Tensor) -> Tensor:

		# --------- generate Q K V, Q = x @ Q_para, K = x @ K_para, V = x @ V_para
		# [N, P, C] --> [N, P, 3C]
		qkv = self.qkv_proj_layer(x)
		# [N, P, 3C] --> [N, P, C] x 3
		query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

		# scale query
		query = query * self.scaling

		# ------- calc attention ----------------
		# K^T
		# [N, P, C] --> [N, C, P]
		key = key.transpose(-2, -1)
		# A = Q @ K^T
		# [N, P, C] @ [N, C, P] --> [N, P, P]
		attn = torch.matmul(query, key)
		attn = self.softmax(attn)  # normalization
		attn = self.attn_dropout_layer(attn)  # dropout

		# out = A @ V
		# [N, P, P] x@[N, P, C] --> [N, P, C]
		out = torch.matmul(attn, value)
		out = self.out_proj(out)   # linear projection, necessary?

		return out

	def profile(self, x: Tensor) -> Tuple[Tensor, float, float]:
		b_sz, seq_len, in_channels = x.shape
		params = macs = 0.0

		qkv, p, m = Prf.profile_list(module_list=self.qkv_proj_layer, x=x)
		params += p
		macs += m * seq_len * b_sz

		# number of operations in QK^T
		m_qk = (seq_len * in_channels * in_channels) * b_sz
		macs += m_qk

		# number of operations in computing weighted sum
		m_wt = (seq_len * in_channels * in_channels) * b_sz
		macs += m_wt

		out_p, p, m = Prf.profile_list(module_list=self.out_proj, x=x)
		params += p
		macs += m * seq_len * b_sz

		return x, params, macs


@Entry.register_entry(Entry.Layer)
class MultiHeadAttention(ExtensionLayer):
	"""
	This layer applies a multi-head self- or cross-attention as described in
	`Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

	Args:
		embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
		num_heads (int): Number of heads in multi-head attention
		attn_dropout (Optional[float]): Attention dropout. Default: 0.0
		add_bias (Optional[bool]): Use bias or not. Default: ``True``

	Shape:
		- Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
		and :math:`C_{in}` is input embedding dim
		- Output: same shape as the input

	"""
	__slots__ = ["embed_dim", "num_heads", "attn_dropout_layer", "add_bias", "coreml_compatible"]
	_disp_ = __slots__

	# defaults = [None, None, 0.0, True, False]

	def __init__(
		self,
		embed_dim: int,
		num_heads: int,
		attn_dropout: Optional[float] = 0.0,
		add_bias: Optional[bool] = True,
		coreml_compatible: Optional[bool] = False,
	) -> None:
		super().__init__(**Par.purify(locals()))

		assert embed_dim % num_heads == 0, \
			f"Embedding dim must be divisible by number of heads in {self.__class__.__name__}. Got: embed_dim={embed_dim} and num_heads={num_heads}"

		self.qkv_proj_layer = LinearLayer(
			in_features=embed_dim, out_features=3 * embed_dim, add_bias=add_bias
		)

		self.head_dim = embed_dim // num_heads
		self.scaling = self.head_dim ** -0.5

		self.softmax = nn.Softmax(dim=-1)
		self.attn_dropout_layer = Dropout(p=attn_dropout)

		# note: Linear layer regards last dimension as input vector
		self.out_proj_layer = LinearLayer(
			in_features=embed_dim, out_features=embed_dim, add_bias=add_bias
		)

	# multi-head data in list form considering coreml doesn't support 4 dimensions matrix multiplication
	def forward_tracing(self, x_q: Tensor, x_kv: Optional[Tensor] = None) -> Tensor:
		# self attention
		if x_kv is None:
			# [N, P, C] --> # [N, P, 3C]
			qkv = self.qkv_proj_layer(x_q)
			# # [N, P, 3C] --> # [N, P, C] x 3
			query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
		# cross attention
		else:
			# [N, P, C]
			query = F.linear(
				x_q,
				weight=self.qkv_proj_layer.weight[: self.embed_dim, ...],
				bias=self.qkv_proj_layer.bias[: self.embed_dim],
			)

			# [N, P, C] --> [N, P, 2C]
			kv = F.linear(
				x_kv,
				weight=self.qkv_proj_layer.weight[self.embed_dim:, ...],
				bias=self.qkv_proj_layer.bias[self.embed_dim:],
			)
			key, value = torch.chunk(kv, chunks=2, dim=-1)

		# scaling query
		query = query * self.scaling

		# split to different heads in list form
		# Note: coreml doesn't support 4 dimensions matrix multiplication !!!
		# [N, P, C] --> [N, P, c] x h, where C = c * h
		query = torch.chunk(query, chunks=self.num_heads, dim=-1)
		value = torch.chunk(value, chunks=self.num_heads, dim=-1)
		key = torch.chunk(key, chunks=self.num_heads, dim=-1)

		wt_out = []
		# iterate on each head
		for h in range(self.num_heads):
			attn_h = torch.matmul(query[h], key[h].transpose(-1, -2))  # calc attention
			attn_h = self.softmax(attn_h)  # normalization
			attn_h = self.attn_dropout_layer(attn_h)  # dropout
			out_h = torch.matmul(attn_h, value[h])  # calc output
			wt_out.append(out_h)

		wt_out = torch.cat(wt_out, dim=-1)  # recat heads
		wt_out = self.out_proj_layer(wt_out)  # linear projection, is it necessary?
		return wt_out

	# multi-head data in matrix form
	def forward_default(self, x_q: Tensor, x_kv: Optional[Tensor] = None) -> Tensor:

		# [N, P, C]
		b_sz, n_patches, in_channels = x_q.shape

		if x_kv is None:
			# self-attention
			# [N, P, C] --> [N, P, 3C] --> [N, P, 3, h, c] where C = hc
			qkv = self.qkv_proj_layer(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)
			# [N, P, 3, h, c] --> [N, h, 3, P, c]
			qkv = qkv.transpose(1, 3).contiguous()

			# [N, h, 3, P, c] --> [N, h, P, c] x 3
			query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
		else:
			# cross-attention, calc Query by x_q, calc Key and Value by x_kv
			# first calc Query by x_q, 0: self.embed_dim, 1/3 parameters
			# [N, P, C]
			query = F.linear(
				x_q,
				weight=self.qkv_proj_layer.weight[: self.embed_dim, ...],
				bias=self.qkv_proj_layer.bias[: self.embed_dim],
			)
			# split to different heads
			# [N, P, C] --> [N, P, h, c] --> [N, h, P, c]
			query = (
				query.reshape(b_sz, n_patches, self.num_heads, self.head_dim)
				.transpose(1, 2)
				.contiguous()
			)

			# second calc Key and Value by x_kv, self.embed_dim: 3 * self.embed_dim, 2/3 parameters
			# [N, P, C] --> [N, P, 2C]
			kv = F.linear(
				x_kv,
				weight=self.qkv_proj_layer.weight[self.embed_dim:, ...],
				bias=self.qkv_proj_layer.bias[self.embed_dim:],
			)

			# split to different heads
			# [N, P, 2C] --> [N, P, 2, h, c]
			kv = kv.reshape(b_sz, n_patches, 2, self.num_heads, self.head_dim)
			# [N, P, 2, h, c] --> [N, h, 2, P, c]
			kv = kv.transpose(1, 3).contiguous()
			key, value = kv[:, :, 0], kv[:, :, 1]

		# scale query
		query = query * self.scaling

		# transpose key
		# [N h, P, c] --> [N, h, c, P]
		key = key.transpose(-1, -2)

		# calc attention,  A = Q @ K^T
		# [N, h, P, c] x [N, h, c, P] --> [N, h, P, P]
		attn = torch.matmul(query, key)
		attn = self.softmax(attn)  # normalization
		attn = self.attn_dropout_layer(attn)  # dropout

		# out = A @ V weighted sum
		# [N, h, P, P] x [N, h, P, c] --> [N, h, P, c]
		out = torch.matmul(attn, value)

		# [N, h, P, c] --> [N, P, h, c] --> [N, P, C]
		out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)  # recat heads

		out = self.out_proj_layer(out)  # linear projection, necessary?

		return out

	def forward(
		self, x_q: Tensor, x_kv: Optional[Tensor] = None
	) -> Tensor:
		if self.coreml_compatible:
			return self.forward_tracing(x_q=x_q, x_kv=x_kv)
		else:
			return self.forward_default(x_q=x_q, x_kv=x_kv)

	def profile(self, x) -> Tuple[Tensor, float, float]:
		b_sz, seq_len, in_channels = x.shape
		params = macs = 0.0

		qkv, p, m = Prf.profile_list(module_list=self.qkv_proj_layer, x=x)
		params += p
		macs += m * seq_len * b_sz

		# number of operations in QK^T
		m_qk = (seq_len * seq_len * in_channels) * b_sz
		macs += m_qk

		# number of operations in computing weighted sum
		m_wt = (seq_len * seq_len * in_channels) * b_sz
		macs += m_wt

		out_p, p, m = Prf.profile_list(module_list=self.out_proj_layer, x=x)
		params += p
		macs += m * seq_len * b_sz

		return x, params, macs
