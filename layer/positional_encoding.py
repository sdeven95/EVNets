import math

from . import ExtensionLayer, Dropout
from utils.type_utils import Par
from utils.entry_utils import Entry

from typing import Optional, Tuple

from torch import Tensor, nn
import torch


# for each input element(P x C elements), add it with a number as a position coder
@Entry.register_entry(Entry.Layer)
class SinusoidalPositionalEncoding(ExtensionLayer):
	"""
	This layer adds sinusoidal positional embeddings to a 3D input tensor. The code has been adapted from
	`Pytorch tutorial <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_

	Args:
		d_model (int): dimension of the input tensor
		dropout (Optional[float]): Dropout rate. Default: 0.0
		max_len (Optional[int]): Max. number of patches (or seq. length). Default: 5000
		channels_last (Optional[bool]): Channels dimension is the last in the input tensor

	Shape:
		- Input: :math:`(N, C, P)` or :math:`(N, P, C)` where :math:`N` is the batch size, :math:`C` is the embedding dimension,
			:math:`P` is the number of patches
		- Output: same shape as the input

	"""
	__slots__ = ["d_model", "dropout_layer", "max_len", "channel_last"]
	_disp_ = __slots__

	# defaults = [None, 0.0, 5000, True]

	def __init__(
		self,
		d_model: int,  # n_channels (embed_dim)
		dropout: Optional[float] = 0.0,
		max_len: Optional[int] = 5000,  # vocabulary size
		channels_last: Optional[bool] = True,
	) -> None:
		super().__init__(**Par.purify(locals()))

		position_last = not channels_last

		# M[max_len, d_model], encoding matrix
		pos_encoding = torch.zeros(max_len, d_model)
		# M[max_len, 1] = [[0], [1], ..., [max_len-1]]
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

		# M[d_model/2]
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
		)

		# M[max_len, d_model]
		pos_encoding[:, 0::2] = torch.sin(position * div_term)
		pos_encoding[:, 1::2] = torch.cos(position * div_term)

		# add dummy batch dimension
		# [1, max_len, d_model], generally max_len = vocabulary size, d_model = embed_dim (n_channels)
		# [1, P, C]
		pos_encoding = pos_encoding.unsqueeze(0)  # [1 x C x P_max] Is comment wrong???

		patch_dim = -2  # patch dimension is second last (N, P, C), position encoding (1, P, C)
		if position_last:
			pos_encoding = pos_encoding.transpose(
				1, 2
			)  # position encoding (1, C, P)
			patch_dim = -1  # patch dimension is last (N, C, P)

		self.dropout_layer = Dropout(p=dropout)
		self.patch_dim = patch_dim
		self.register_buffer("pe", pos_encoding)

	# patch_dim = -1, pe = (1, C, P), x = (N, C, P)
	def forward_patch_last(
		self, x, indices: Optional[Tensor] = None
	) -> Tensor:
		# seq_length should be the last dim
		# x -> [N, C, P]
		if indices is None:
			x = x + self.pe[..., : x.shape[-1]]  # add position value
		else:
			ndim = x.ndim
			repeat_size = [x.shape[0]] + [-1] * (ndim - 1)

			pe = self.pe.expand(repeat_size)

			# select passed vocabularies' encoding values
			selected_pe = torch.gather(pe, index=indices, dim=-1)
			x = x + selected_pe
		return self.dropout_layer(x)  # dropout

	# patch_dim = -2 , pe = (1, P, C), x = (N, P, C)
	def forward_others(
		self, x, indices: Optional[Tensor] = None
	) -> Tensor:
		# seq_length should be the second last dim
		# x -> [N, P, C]
		if indices is None:
			x = x + self.pe[..., : x.shape[-2], :]  # add position value
		else:
			ndim = x.ndim
			repeat_size = [x.shape[0]] + [-1] * (ndim - 1)

			pe = self.pe.expand(repeat_size)
			selected_pe = torch.gather(pe, index=indices, dim=-2)
			x = x + selected_pe
		return self.dropout_layer(x)  # dropout

	def forward(self, x, indices: Optional[Tensor] = None) -> Tensor:
		if self.patch_dim == -1:
			return self.forward_patch_last(x, indices=indices)
		else:
			return self.forward_others(x, indices=indices)

	def profile(self, x: Tensor) -> Tuple[Tensor, float, float]:
		return x, 0.0, 0.0


# nn.Embedding generate a position encoding parameter matrix [vocabulary_size, embed_dim]
@Entry.register_entry(Entry.Layer)
class LearnablePositionEncoding(ExtensionLayer):
	"""
	This layer adds learnable positional embeddings to a 3D input tensor.

	Args:
		embed_dim (int): dimension of the input tensor
		num_embeddings (int): number of input embeddings. This is similar to vocab size in NLP.
		dropout (Optional[float]): Dropout rate. Default: 0.0
		channels_last (Optional[bool]): Channels dimension is the last in the input tensor

	Shape:
		- Input: :math:`(N, *, C, P)` or :math:`(N, *, P, C)` where :math:`N` is the batch size, :math:`C` is the embedding dimension,
			:math:`P` is the number of patches
		- Output: same shape as the input

	"""
	__slots__ = ["embed_dim", "num_embeddings", "drop_out", "channels_last"]
	_disp_ = __slots__

	# defaults = [None, None, 0.0, True]

	def __init__(
		self,
		embed_dim: int,  # n_channels
		num_embeddings: int,  # vocabulary size
		dropout: Optional[float] = 0.0,
		channels_last: Optional[bool] = True,
	) -> None:
		super().__init__(**Par.purify(locals()))
		self.pos_emb = nn.Embedding(
			num_embeddings=num_embeddings, embedding_dim=embed_dim
		)
		self.dropout_layer = Dropout(p=dropout)

	def forward(self, x) -> Tensor:
		# x -> (N, num_patch, embed_dim)
		num_embeddings = x.shape[-2] if self.channel_last else x.shape[-1]  # number of patch
		positions = torch.arange(num_embeddings, dtype=torch.int64, device=x.device)  # [num_patch]
		position_emb = self.pos_emb(positions)  # [num_patch, embed_dim] note: parameters

		# add position encoding parameters
		position_emb = position_emb.expand_as(x)
		x = x + position_emb
		return self.dropout_layer(x)

	def profile(self, x: Tensor) -> Tuple[Tensor, float, float]:
		return x, 0.0, 0.0
