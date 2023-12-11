import torch
from torch import nn


class SingleHeadAttention(nn.Module):
	def __init__(
		self,
		embed_dim: int,
		attn_dropout: float = 0.0,
		add_bias: bool = True,
	):
		super().__init__()

		self.qkv_proj_layer = nn.Linear(
			in_features=embed_dim, out_features=3 * embed_dim, bias=add_bias
		)

		self.scaling = embed_dim ** -0.5

		self.softmax_handler = nn.Softmax(dim=-1)
		self.attn_dropout_layer = nn.Dropout(p=attn_dropout)

		self.out_proj = nn.Linear(
			in_features=embed_dim, out_features=embed_dim, bias=add_bias
		)

	def forward(self, x):

		# --------- generate Q K V, Q = x @ Q_para, K = x @ K_para, V = x @ V_para
		# [N, P, C] --> [N, P, 3C]
		# [b, num_patches, embed_dim] -> [b, num_patches, 3 * embed_dim]
		qkv = self.qkv_proj_layer(x)
		# [b, num_patches, 3 * embed_dim] -> [b, num_patches, embed_dim] * 3
		query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

		# scale query
		query = query * self.scaling

		# ------- calc attention ----------------
		# K^T
		# [b, num_patches, embed_dim] --> [b, embed_dim, num_patches]
		key = key.transpose(-2, -1)
		# A = Q @ K^T [b, num_patches, num_patches]
		attn = torch.matmul(query, key)
		attn = self.softmax_handler(attn)  # normalization
		attn = self.attn_dropout_layer(attn)  # dropout

		# out = A @ V
		# [b, num_patches, num_patches] @ [b, num_patches, embed_dim] --> [b, num_patches, embed_dim]
		out = torch.matmul(attn, value)
		out = self.out_proj(out)   # linear projection

		return out


class MultiHeadAttention(nn.Module):
	def __init__(
		self,
		embed_dim: int,
		num_heads: int,
		attn_dropout: float = 0.0,
		add_bias: bool = True,
	):
		super().__init__()

		self.num_heads = num_heads
		self.embed_dim = embed_dim

		assert embed_dim % num_heads == 0, \
			f"Embedding dim must be divisible by number of heads in {self.__class__.__name__}. " \
			f"Got: embed_dim={embed_dim} and num_heads={num_heads}"

		self.qkv_proj_layer = nn.Linear(
			in_features=embed_dim, out_features=3 * embed_dim, bias=add_bias
		)

		self.head_dim = embed_dim // num_heads
		self.scaling = self.head_dim ** -0.5

		self.softmax_handler = nn.Softmax(dim=-1)
		self.attn_dropout_layer = nn.Dropout(p=attn_dropout)

		# note: Linear layer regards last dimension as input vector
		self.out_proj_layer = nn.Linear(
			in_features=embed_dim, out_features=embed_dim, bias=add_bias
		)

	def forward(self, x):
		b, num_patches, embed_dim = x.shape
		# [b, num_patches, embed_dim] => [b, num_patches, 3*embed_dim] => [b, num_patches, 3, num_heads, head_dim]
		qkv = self.qkv_proj_layer(x).reshape(b, num_patches, 3, self.num_heads, -1)
		# => [b, num_heads, 3, num_patches, head_dim]
		qkv = qkv.transpose(1, 3).contiguous()
		# => [b, num_heads, num_patches, head_dim] * 3
		query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

		query = query * self.scaling
		key = key.transpose(-1, -2)   # => [b, num_heads, head_dim, num_patches]

		attn = torch.matmul(query, key)  # => [b, num_heads, num_patches, num_patches]
		attn = self.softmax_handler(attn)
		attn = self.attn_dropout_layer(attn)

		out = torch.matmul(attn, value)  # => [b, num_heads, num_patches, head_dim]
		out = out.transpose(1, 2)  # => [b, num_patches, num_heads, head_dim]
		out = out.reshape(b, num_patches, -1)   # => [b, num_patches, embed_dim]

		out = self.out_proj_layer(out)

		return out


class TransformerEncoder(nn.Module):
	def __init__(
			self,
			embed_dim: int,
			ffn_latent_dim: int,

			num_heads: int = 8,
			attn_add_bias: bool = True,
			attn_dropout: float = 0.0,

			ffn_activation_layer: nn.Module = nn.Hardswish,
			ffn_dropout: float = 0.0,

			transformer_norm_layer: nn.Module = nn.LayerNorm,
			transformer_dropout: float = 0.0,
	):
		super().__init__()

		# attention net
		self.attn_part = nn.Sequential(
			transformer_norm_layer(embed_dim),
			MultiHeadAttention(
				embed_dim=embed_dim,
				num_heads=num_heads,
				attn_dropout=attn_dropout,
				add_bias=attn_add_bias,
			) if num_heads > 1 else
			SingleHeadAttention(
				embed_dim=embed_dim,
				attn_dropout=attn_dropout,
				add_bias=attn_add_bias,
			),
			nn.Dropout(p=transformer_dropout)
		)

		# fnn net
		self.fnn_part = nn.Sequential(
			transformer_norm_layer(embed_dim),
			nn.Linear(embed_dim, ffn_latent_dim, bias=True),
			ffn_activation_layer(),
			nn.Dropout(p=ffn_dropout),
			nn.Linear(ffn_latent_dim, embed_dim, bias=True),
			nn.Dropout(p=transformer_dropout),
		)

	def forward(self, x):
		x += self.attn_part(x)
		x += self.fnn_part(x)
		return x


class ComponentFocus(nn.Module):
	def __init__(
		self,
		num_patches: int,

		embed_dim: int,
		ffn_latent_dim: int,

		num_heads: int = 8,
		attn_add_bias: bool = True,
		attn_dropout: float = 0.0,

		ffn_activation_layer: nn.Module = nn.Hardswish,
		ffn_dropout: float = 0.0,

		transformer_norm_layer: nn.Module = nn.LayerNorm,
		transformer_dropout: float = 0.0,
	):
		super().__init__()

		self.num_patches = num_patches

		self.transformer = TransformerEncoder(
			embed_dim=embed_dim,
			ffn_latent_dim=ffn_latent_dim,

			num_heads=num_heads,
			attn_add_bias=attn_add_bias,
			attn_dropout=attn_dropout,

			ffn_activation_layer=ffn_activation_layer,
			ffn_dropout=ffn_dropout,

			transformer_norm_layer=transformer_norm_layer,
			transformer_dropout=transformer_dropout,
		)

	def forward(self, x):
		b, c, h, w = x.shape

		# ===== group to vectors ======
		x = x.view(b, c, -1)  # [b, c, h*w]
		channel_sum = x.sum(axis=1)  # [b, h*w]
		sort_indices = torch.argsort(channel_sum, dim=-1, descending=True)  # [b, h*w]

		# for getting original positions
		reverse_sort_indices = torch.zeros(b, h*w, dtype=torch.int)
		for i in range(b):
			for idx, pos in enumerate(sort_indices[i]):
				reverse_sort_indices[i, pos] = idx

		x = torch.stack([x[i, :, sort_indices[i]] for i in range(b)], dim=0)  # [b, c, h*w]

		patch_dim = h*w // self.num_patches
		x = torch.split(x, patch_dim, dim=-1)  # tuple([b, c, h*w // num_patches], ...)
		avg_list = [ele.sum(dim=-1) / patch_dim for ele in x]  # [ [b, c], ... ]
		x = torch.stack(avg_list, dim=1)  # [b, num_patches, c]

		# ======= vectors attention transform ==========

		out = self.transformer(x) if self.num_patches > 1 else x  # [b, num_patches, c]

		# ===== decompose vectors =====
		out = out.transpose(-1, -2)  # => [b, c, num_patches]
		out = torch.cat([out[:, :, i: (i+1)].repeat(1, 1, patch_dim) for i in range(self.num_patches)], dim=-1)  # => [b, c, h*w]

		out = torch.stack([out[i, :, reverse_sort_indices[i]] for i in range(b)], dim=0)  # recover orders

		out = out.reshape(b, c, h, w)

		return out


class PartChannelAdaptiveConv(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int = 3,
		stride: int = 1,
		drop_splits: int = 2,
	):
		super().__init__()

		self.actual_in_channels = in_channels - in_channels // drop_splits

		self.conv = nn.Conv2d(
			in_channels=self.actual_in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=stride,
			padding=kernel_size // 2,
		)

	def forward(self, x):
		batch_size = x.shape[0]
		channel_sum = x.sum(dim=(-2, -1))  # [b, c]
		sort_indices = channel_sum.argsort(axis=1)  # [b, c]

		out = torch.stack([
			x[i, sort_indices[i][0: self.actual_in_channels], :, :] for i in range(batch_size)
		])
		return self.conv(out)


class SpindleResidual(nn.Module):
	def __init__(
			self,
			in_channels: int,
			mid_channels: int,
			out_channels: int,
			kernel_size: int = 3,
			stride: int = 1,
			norm_layer: nn.Module = nn.BatchNorm2d,
			act_layer: nn.Module = nn.Hardswish,
			drop_splits: int = 2,
	):
		super().__init__()

		self.head = nn.Sequential(
			PartChannelAdaptiveConv(
				in_channels=in_channels,
				out_channels=mid_channels,
				kernel_size=1,
				drop_splits=drop_splits,
			),
			norm_layer(mid_channels),
			act_layer(),
		)

		self.body = nn.Sequential(
			nn.Conv2d(
				in_channels=mid_channels,
				out_channels=mid_channels,
				kernel_size=kernel_size,
				stride=stride,
				groups=mid_channels,
				padding=kernel_size // 2,
			),
			norm_layer(mid_channels),
			act_layer(),
		)

		self.tail = nn.Sequential(
			PartChannelAdaptiveConv(
				in_channels=mid_channels,
				out_channels=out_channels,
				kernel_size=1,
				drop_splits=drop_splits,
			),
			norm_layer(out_channels),
		)

		self.down_sample = None
		if in_channels != out_channels or stride == 2:
			self.down_sample = nn.Sequential(
				nn.Conv2d(
					in_channels=in_channels,
					out_channels=out_channels,
					kernel_size=1,
					stride=stride,
				),
				norm_layer(out_channels)
			)

	def forward(self, x):
		res = x
		out = self.tail(self.body(self.head(x)))
		return out + self.down_sample(res) if self.down_sample else out + res


class LocalGlobalResidualLayer(nn.Module):
	def __init__(
			self,
			# local SpindleResidual: Required
			in_channels: int,
			mid_channels: int,
			out_channels: int,

			# global ComponentFocus: Required
			num_patches: int,

			embed_dim: int,
			ffn_latent_dim: int,

			# local SpindleResidual: Optional
			kernel_size: int = 3,
			stride: int = 1,
			local_norm_layer: nn.Module = nn.BatchNorm2d,
			local_act_layer: nn.Module = nn.Hardswish,
			drop_splits: int = 2,

			# global ComponentFocus: Optional
			num_heads: int = 8,
			attn_add_bias: bool = True,
			attn_dropout: float = 0.0,

			ffn_activation_layer: nn.Module = nn.Hardswish,
			ffn_dropout: float = 0.0,

			transformer_norm_layer: nn.Module = nn.LayerNorm,
			transformer_dropout: float = 0.0,
	):
		super().__init__()

		self.lcl = SpindleResidual(
			in_channels=in_channels,
			mid_channels=mid_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			stride=stride,
			drop_splits=drop_splits,
			norm_layer=local_norm_layer,
			act_layer=local_act_layer,
		)

		self.gbl = ComponentFocus(
			num_patches=num_patches,

			embed_dim=embed_dim,
			num_heads=num_heads,
			attn_add_bias=attn_add_bias,
			attn_dropout=attn_dropout,

			ffn_latent_dim=ffn_latent_dim,
			ffn_activation_layer=ffn_activation_layer,
			ffn_dropout=ffn_dropout,

			transformer_norm_layer=transformer_norm_layer,
			transformer_dropout=transformer_dropout,
		)

		self.residual_sample = None
		if in_channels != out_channels or stride == 2:
			self.residual_sample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
				nn.BatchNorm2d(out_channels),
			)

	def forward(self, x):
		res = x
		x = x + self.gbl(x)
		x = self.lcl(x)
		out = x + self.residual_sample(res) if self.residual_sample else x + res
		print(out.shape)
		return out


class Cfg:   # tiny small large huge
	# --------- local -------------
	strides = [2, 1, 2, 2, 2, 2]

	in_channels = [3,   32,  64, 128, 256, 512]
	out_channels = [32, 64, 128, 256, 512, 1024]

	mid_channels = [64, 128, 256, 512, 1024, 2048]

	kernel_sizes = [3, 3, 3, 3, 3, 3]

	drop_splits = [32, 16, 16, 8, 8, 4]  # ########

	# ----------- global ---------------
	# 224 => 112(layer0) => 112(layer1) => 56(layer2) => 28(layer3) => 14(layer4) => 7(layer5)
	num_patches = [64,          64,            16,           16,            4,           1]

	# embed_dim == in_channels
	# ffn_latent_dim = embed_dim * 2

	num_heads = [2, 4, 8, 16, 32, 64]  # layer0: 1
	attn_dropouts = [0.0, 0.0, 0.1, 0.1, 0.2, 0.2]  # ######
	ffn_dropouts = [0.0, 0.0, 0.1, 0.1, 0.2, 0.2]  # #####
	transformer_dropout = [0.0, 0.0, 0.1, 0.1, 0.2, 0.2]  # #####

	# number of blocks for per layer
	num_blocks = {
		"tiny": [1, 1, 2, 2, 2, 2],
		"small": [1, 1, 3, 4, 6, 3],
		"large": [1, 1, 3, 4, 23, 3],
		"huge": [1, 1, 3, 8, 36, 3],
	}


class AutoAdaptiveNet(nn.Module):
	def __init__(self, model_type):
		super().__init__()
		self.model_type = model_type

		self.block = nn.Sequential()

		self.block.add_module(
			name="layer0",
			module=LocalGlobalResidualLayer(
				in_channels=Cfg.in_channels[0],
				mid_channels=Cfg.mid_channels[0],
				out_channels=Cfg.out_channels[0],
				drop_splits=Cfg.drop_splits[0],
				stride=Cfg.strides[0],

				num_patches=Cfg.num_patches[0],
				embed_dim=Cfg.in_channels[0],
				num_heads=1,
				ffn_latent_dim=Cfg.in_channels[0] * 4,
				attn_dropout=Cfg.attn_dropouts[0],
				ffn_dropout=Cfg.ffn_dropouts[0],
				transformer_dropout=Cfg.transformer_dropout[0],
			)
		)

		for layer_id in range(1, 6):
			self.block.add_module(
				name=f"layer{layer_id}",
				module=self.__make_layer(layer_id)
			)

	def forward(self, x):
		return self.block(x)

	def __make_layer(self, layer_id):
		blocks = nn.Sequential()
		blocks.add_module(
			name=f"layer{layer_id}_block0",
			module=LocalGlobalResidualLayer(
				in_channels=Cfg.in_channels[layer_id],
				mid_channels=Cfg.mid_channels[layer_id],
				out_channels=Cfg.out_channels[layer_id],
				drop_splits=Cfg.drop_splits[layer_id - 1],
				stride=Cfg.strides[layer_id],

				num_patches=Cfg.num_patches[layer_id - 1],
				embed_dim=Cfg.in_channels[layer_id],
				num_heads=Cfg.num_heads[layer_id - 1],
				ffn_latent_dim=Cfg.in_channels[layer_id] * 2,
				attn_dropout=Cfg.attn_dropouts[layer_id - 1],
				ffn_dropout=Cfg.ffn_dropouts[layer_id - 1],
				transformer_dropout=Cfg.transformer_dropout[layer_id-1],
			),
		)

		for i in range(1, Cfg.num_blocks[self.model_type][layer_id]):
			blocks.add_module(
				name=f"layer{layer_id}_block{i}",
				module=LocalGlobalResidualLayer(
					in_channels=Cfg.out_channels[layer_id],
					mid_channels=Cfg.mid_channels[layer_id],
					out_channels=Cfg.out_channels[layer_id],
					drop_splits=Cfg.drop_splits[layer_id],
					stride=1,

					num_patches=Cfg.num_patches[layer_id],
					embed_dim=Cfg.out_channels[layer_id],
					ffn_latent_dim=Cfg.in_channels[layer_id] * 2,
					attn_dropout=Cfg.attn_dropouts[layer_id],
					ffn_dropout=Cfg.ffn_dropouts[layer_id],
					transformer_dropout=Cfg.transformer_dropout[layer_id],
				)
			)

		return blocks


if __name__ == "__main__":
	# x = torch.rand(2, 64, 48, 48).view(2, 64, -1).permute(0, 2, 1)
	# print(x.shape)
	# m = MultiHeadAttention(embed_dim=64, num_heads=8)
	# y = m(x)
	# print(y.shape)

	# x = torch.rand(2, 64, 48, 48).view(2, 64, -1).permute(0, 2, 1)
	# print(x.shape)
	# m = TransformerEncoder(embed_dim=64, ffn_latent_dim=128)
	# y = m(x)
	# print(y.shape)

	# x = torch.rand(2, 64, 48, 48)
	# print(x.shape)
	# m = ComponentFocus(num_patches=16, embed_dim=64, ffn_latent_dim=128)
	# y = m(x)
	# print(y.shape)

	# x = torch.rand(2, 64, 48, 48)
	# print(x.shape)
	# m = PartChannelAdaptiveConv(in_channels=64, out_channels=128)
	# y = m(x)
	# print(y.shape)

	# x = torch.rand(2, 64, 48, 48)
	# print(x.shape)
	# m = SpindleResidual(in_channels=64, hidden_channels=128, out_channels=64, stride=2)
	# y = m(x)
	# print(y.shape)

	# x = torch.rand(2, 64, 48, 48)
	# print(x.shape)
	# m = LocalGlobalResidualLayer(
	# 	in_channels=64,
	# 	mid_channels=128,
	# 	out_channels=128,
	# 	stride=2,
	#
	# 	num_patches=16,
	# 	embed_dim=64,
	# 	ffn_latent_dim=128,
	# )
	# y = m(x)
	# print(y.shape)

	x = torch.rand(2, 3, 224, 224)
	m = AutoAdaptiveNet(model_type="tiny")
	y = m(x)
	print(y.shape)
