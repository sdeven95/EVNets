from torch import nn

from utils.type_utils import Cfg, Prf, Dsp, Arg, Dct, Par
from utils.entry_utils import Entry


# obsoleted ???
class BaseLayer(Cfg, Dsp, Prf, nn.Module):
	__slots__ = ["conv_init", "linear_init", "conv_init_std_dev", "linear_init_std_dev", "group_linear_init_std_dev"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = ["kaiming_normal", "normal", 0.01, 0.01, 0.01]
	_types_ = [str, str, float, float, float]

	_cfg_path_ = Entry.Layer

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		nn.Module.__init__(self)


class BuiltinLayer(Arg, Dsp, Prf):

	def __init__(self, **kwargs):
		Arg.__init__(self, **kwargs)
		if "add_bias" in kwargs:
			Dct.change_key_names(kwargs, ["add_bias"], ["bias"])
		kwargs = Dct.update_key_values(target=kwargs, source=self.__dict__)
		super(Prf, self).__init__(**kwargs)


class ExtensionLayer(Arg, Dsp, Prf, nn.Module):
	def __init__(self, **kwargs):
		Arg.__init__(self, **kwargs)
		nn.Module.__init__(self)


from .builtins import Identity, LinearLayer, \
	Conv1d, Conv2d, Conv3d, \
	ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, \
	MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, \
	Dropout, Dropout2D, Flatten, PixelShuffle, Softmax, UpSample

from .extensions import GlobalPool, ConvLayer2D, ConvLayer3D
from .transformer import SingleHeadAttention, MultiHeadAttention, TransformerEncoder
from .positional_encoding import SinusoidalPositionalEncoding, LearnablePositionEncoding



