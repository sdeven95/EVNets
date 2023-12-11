from . import BaseLossBuiltin
from utils.type_utils import Par
from utils.entry_utils import Entry
from torch import nn


@Entry.register_entry(Entry.Loss)
class L1Loss(BaseLossBuiltin, nn.L1Loss):
	__slots__ = ["size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, None]
	_types_ = [bool, bool]

	def __init__(self, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class MSELoss(BaseLossBuiltin, nn.MSELoss):
	__slots__ = ["size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, None]
	_types_ = [bool, bool]

	def __init__(self, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class CrossEntropyLoss(BaseLossBuiltin, nn.CrossEntropyLoss):
	__slots__ = ["weight", "ignore_index", "label_smoothing", "average_size", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, -100, 0.0, None, None]
	_types_ = [list, int, float, bool, bool]

	def __init__(self, weight=None, ignore_index=None, label_smoothing=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class CTCLoss(BaseLossBuiltin, nn.CTCLoss):
	__slots__ = ["blank", "zero_infinity"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [0, False]
	_types_ = [int, bool]

	def __init__(self, blank=None, zero_infinity=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class NLLLoss(BaseLossBuiltin, nn.NLLLoss):
	__slots__ = ["weight", "ignore_index", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, -100, None, None]
	_types_ = [list, int, bool, bool]

	def __init__(self, weight=None, ignore_index=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class PoissonNLLLoss(BaseLossBuiltin, nn.PoissonNLLLoss):
	__slots__ = ["log_input", "full", "size_average", "eps", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [True, False, None, 1.e-08, None]
	_types_ = [bool, bool, bool, float, bool]

	def __init__(self, log_input=None, full=None, size_average=None, eps=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class GaussianNLLLoss(BaseLossBuiltin, nn.GaussianNLLLoss):
	__slots__ = ["full", "eps"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [False, 1.0e-6]
	_types_ = [bool, float]

	def __init__(self, full=None, eps=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class KLDivLoss(BaseLossBuiltin, nn.KLDivLoss):
	__slots__ = ["log_target", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [False, None, None]
	_types_ = [bool, bool, bool]

	def __init__(self, log_target=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class BCELoss(BaseLossBuiltin, nn.BCELoss):
	__slots__ = ["weight", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, None, None]
	_types_ = [list, bool, bool]

	def __init__(self, weight=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class BCEWithLogitsLoss(BaseLossBuiltin, nn.BCEWithLogitsLoss):
	__slots__ = ["weight", "pos_weight", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, None, None, None]
	_types_ = [list, list, bool, bool]

	def __init__(self, weight=None, pos_weight=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class MarginRankingLoss(BaseLossBuiltin, nn.MarginRankingLoss):
	__slots__ = ["margin", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [0.0, None, None]
	_types_ = [float, bool, bool]

	def __init__(self, margin=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class HingeEmbeddingLoss(BaseLossBuiltin, nn.HingeEmbeddingLoss):
	__slots__ = ["margin", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1.0, None, None]
	_types_ = [float, bool, bool]

	def __init__(self, margin=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class MultiLabelMarginLoss(BaseLossBuiltin, nn.MultiLabelMarginLoss):
	__slots__ = ["size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, None]
	_types_ = [bool, bool]

	def __init__(self, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class HuberLoss(BaseLossBuiltin, nn.HuberLoss):
	__slots__ = ["delta"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1.0]
	_types_ = [float]

	def __init__(self, delta=None, reduction=None):
		super().__init__(delta=delta, reduction=reduction)


@Entry.register_entry(Entry.Loss)
class SmoothL1Loss(BaseLossBuiltin, nn.SmoothL1Loss):
	__slots__ = ["beta", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1.0, None, None]
	_types_ = [float, bool, bool]

	def __init__(self, beta=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class SoftMarginLoss(BaseLossBuiltin, nn.SoftMarginLoss):
	__slots__ = ["size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, None]
	_types_ = [bool, bool]

	def __init__(self, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class MultiLabelSoftMarginLoss(BaseLossBuiltin, nn.MultiLabelSoftMarginLoss):
	__slots__ = ["weight", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [None, None, None]
	_types_ = [list, bool, bool]

	def __init__(self, weight=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class CosineEmbeddingLoss(BaseLossBuiltin, nn.CosineEmbeddingLoss):
	__slots__ = ["margin", "size_average", "recude"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [0.0, None, None]
	_types_ = [float, bool, bool]

	def __init__(self, margin=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class MultiMarginLoss(BaseLossBuiltin, nn.MultiMarginLoss):
	__slots__ = ["p", "margin", "weight", "size_average", "reduce"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [1, 1.0, None, None, None]
	_types_ = [int, float, list, bool, bool]

	def __init__(self, p=None, margin=None, weight=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class TripletMarginLoss(BaseLossBuiltin, nn.TripletMarginLoss):
	__slots__ = ["margin", "p", "eps", "swap", "size_average", "reduce"]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [1.0, 2.0, 1.0e-06, False, None, None]
	_types_ = [float, float, float, bool, bool, bool]

	def __init__(self, margin=None, p=None, eps=None, swap=None, size_average=None, reduce=None, reduction=None):
		super().__init__(**Par.purify(locals()))


@Entry.register_entry(Entry.Loss)
class TripletMarginWithDistanceLoss(BaseLossBuiltin, nn.TripletMarginWithDistanceLoss):
	__slots__ = ["margin", "swap"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [1.0, False]
	_types_ = [float, bool]

	def __init__(self, distance_function=None, margin=None, swap=None, reduction=None):
		super().__init__(**Par.purify(locals()))
