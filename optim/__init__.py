from utils.type_utils import Cfg, Dsp, Opt, Par
from utils.entry_utils import Entry


class OptimUtil:
	@staticmethod
	def get_optimizer(entry_name, *args, **kwargs):
		model = kwargs["model"] if "model" in kwargs else args[0]

		weight_decay = Opt.get_by_cfg_path(Entry.Optimizer, "weight_decay")
		no_decay_bn_filter_bias = Opt.get_by_cfg_path(Entry.Optimizer, "no_decay_bn_filter_bias")

		from utils import Util
		model = Util.get_module(model)
		model_params, lr_mult = model.get_training_parameters(
			weight_decay=weight_decay, no_decay_bn_filter_bias=no_decay_bn_filter_bias
		)
		Opt.set_by_cfg_path(Entry.Scheduler, "lr_multipliers", lr_mult)

		if 'model' in kwargs:
			kwargs.pop('model')
		else:
			del args[0]

		return Entry.dict_entry[Entry.Optimizer][entry_name](params=model_params, *args, **kwargs)


class BaseOptim(Cfg, Dsp):
	__slots__ = ["weight_decay", "no_decay_bn_filter_bias"]
	_cfg_path_ = Entry.Optimizer
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [4e-5, False]
	_types_ = [float, bool]

	def __init__(self, **kwargs):
		Cfg.__init__(self, **kwargs)
		Par.init_helper(self, kwargs, super(Dsp, self))

	def __repr__(self):
		group_dict = dict()
		for group in self.param_groups:
			for key in sorted(group.keys()):
				if key == "params":
					continue
				if key not in group_dict:
					group_dict[key] = [group[key]]
				else:
					group_dict[key].append(group[key])

		format_string = self.__class__.__name__ + " (\n"
		for k, v in group_dict.items():
			format_string += "\t {0}: {1}\n".format(k, v)
		format_string += ")"
		return format_string

# obsoleted
# from . builtins import SGD, Adadelta, Adagrad, Adam, Adamax, AdamW, \
# 	ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SparseAdam


