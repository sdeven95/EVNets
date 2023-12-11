from torch import nn
from . import Logger
import sys

import torch


# disable __slots__
class MetaClass(type):
	def __new__(mcs, name, bases, attrs):
		if "__slots__" in attrs:
			attrs.pop("__slots__")
		return type.__new__(mcs, name, bases, attrs)


# helper class for repressing a module
class Dsp(metaclass=MetaClass):
	__slots__ = ["_disp_keys_"]

	_disp_ = None

	def __init__(self):
		if not hasattr(self, "_disp_keys_"):
			self.__class__.add_disp_keys()

	@classmethod
	def _add_disp_keys_(cls, tar_cls):
		if cls._disp_:
			tar_cls._disp_keys_ = tar_cls._disp_keys_.union(cls._disp_)

	# called by bottom class
	@classmethod
	def add_disp_keys(cls):
		cls._disp_keys_ = set()
		for middle_cls in cls.mro()[:-1][::-1]:
			if issubclass(middle_cls, Dsp):
				middle_cls._add_disp_keys_(cls)

	def __repr__(self):
		if not self._disp_keys_:
			return self.__class__.__name__ + "()"

		tar_keys = self._disp_keys_

		tpl = self.__class__.__name__ + "("
		tpl_keys = list(map(lambda key: key + "={}", tar_keys))
		tpl += ", ".join(tpl_keys) + ")"
		return tpl.format(*[self.__dict__[key_] for key_ in tar_keys])

	def _repr_by_line(self):
		if not self._disp_keys_:
			return self.__class__.__name__ + "()\n"

		tar_keys = self._disp_keys_

		tpl = self.__class__.__name__ + "(\n"
		tpl_keys = map(lambda key: "\t" + key + " = {}", tar_keys)
		tpl += ",\n".join(tpl_keys) + "\n)"
		return tpl.format(*[self.__dict__[key_] for key_ in tar_keys])


# helper class, update self.__dict__ by kwargs
# and call Disp.init
class Arg(metaclass=MetaClass):
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		if isinstance(self, Dsp):
			Dsp.__init__(self)


# base class of configurable class
class Cfg(metaclass=MetaClass):
	__slots__ = ["_arg_dict_", "_arg_pth_tpl_"]

	# _arg_dict_: { key0: default0, key1: default1, ...}
	# _arg_pth_tpl_: cfg_path.class_name.{}

	_cfg_path_ = None
	_keys_ = None
	_defaults_ = None
	_types_ = None

	def __init__(self, **kwargs):
		# create _arg_dict_ and _arg_pth_tpl_, and add variables to arg parser
		if not hasattr(self.__class__, "_arg_dict_"):
			self.__class__.add_arguments()

		# priority level: **kwargs(except None) > opts > _defaults_
		# get values from opts, if key does not exist in opts, use _defaults_
		para_name_tuple = tuple(self._arg_pth_tpl_.format(key_) for key_ in self._arg_dict_.keys())
		if "_opts_" in sys.modules and sys.modules["_opts_"]:
			defaults = tuple(getattr(sys.modules["_opts_"], name_, default_[0]) for name_, default_ in zip(para_name_tuple, self._arg_dict_.values()))
		else:
			defaults = tuple(default_[0] for default_ in self._arg_dict_.values())
		# get keys' values from **kwargs, note: None value in **kwargs will be ignored
		self.__dict__.update(
			{key_: (kwargs.get(key_, default_) if kwargs.get(key_, None) is not None else default_)
			 for key_, default_ in zip(self._arg_dict_.keys(), defaults)})

		if isinstance(self, Dsp):
			Dsp.__init__(self)

	def __setattr__(self, key, value):
		if key in self._arg_dict_ and "_opts_" in sys.modules and sys.modules["_opts_"]:
			setattr(sys.modules["_opts_"], self._arg_pth_tpl_.format(key), value)  # sync options
		super().__setattr__(key, value)

	# only add arguments to argument parser
	@classmethod
	def _add_arguments_(cls, arg_dict):
		if cls._keys_:
			if isinstance(cls._keys_, tuple):
				cls._keys_ = cls._keys_[0]
			if isinstance(cls._defaults_, tuple):
				cls._defaults_ = cls._defaults_[0]
			if isinstance(cls._types_, tuple):
				cls._types_ = cls._types_[0]

			assert cls._defaults_ is not None and cls._types_ is not None, \
				f"defaults and types could not be None for class {cls.__name__}"

			assert len(cls._keys_) == len(cls._defaults_) == len(cls._types_), \
				f"length mismatch of keys, defaults, types for class {cls.__name__}"
			arg_dict.update({k: [v, t] for k, v, t in zip(cls._keys_, cls._defaults_, cls._types_)})

	@classmethod
	def add_arguments(cls):
		cls._arg_dict_ = {}
		cls._arg_pth_tpl_ = cls._cfg_path_ + "." + cls.__name__ + ".{}"
		for middle_cls in cls.mro()[:-1][::-1]:
			if issubclass(middle_cls, Cfg):
				middle_cls._add_arguments_(cls._arg_dict_)

		if "_parser_" not in sys.modules:
			return

		if len(cls._arg_dict_) == 0:
			return

		group_name = cls._arg_pth_tpl_[:cls._arg_pth_tpl_.rindex(".")]
		arg_pth_tpl = "--" + cls._arg_pth_tpl_.replace("_", "-")  # for adding arguments to parser
		arg_name_tuple = tuple(
			arg_pth_tpl.format(key_.replace("_", "-")) for key_ in cls._arg_dict_.keys())  # for adding arguments to parser

		group = sys.modules["_parser_"].add_argument_group(
			title=group_name,
			description=group_name
		)
		for name_, val_type_ in zip(arg_name_tuple, cls._arg_dict_.values()):
			if type(val_type_[1]) is tuple:
				group.add_argument(
					name_,
					type=val_type_[1][0],
					default=val_type_[0],
					help=name_,
					nargs="+"
				)
			elif val_type_[1] is bool:
				group.add_argument(
					name_,
					action="store_true",
					default=val_type_[0],
					help=name_
				)
			else:
				group.add_argument(
					name_,
					type=val_type_[1],
					default=val_type_[0],
					help=name_
				)
		return group

	@classmethod
	def print_arguments(cls):
		dot_count = cls._arg_pth_tpl_.count(".")
		prefix = dot_count * "  "
		tpl = "  " * (dot_count - 1) + cls.__name__ + ":\n"

		tpl_keys = map(lambda key: prefix + key + ": {}", cls._arg_dict_.keys())
		tpl += "\n".join(tpl_keys)

		arg_values = [v[0] for v in cls._arg_dict_.values()]
		Logger.plain_text(tpl.format(*arg_values))

	def print_arguments_values(self):
		dot_count = self._arg_pth_tpl_.count(".") if self._arg_pth_tpl_ else 1

		prefix = dot_count * "  "
		tpl = "  " * (dot_count - 1) + self.__class__.__name__ + ":\n"
		tpl_keys = map(lambda key: prefix + key + ": {}", self._arg_dict_.keys())
		tpl += "\n".join(tpl_keys)
		tar_values = [self.__dict__[key] for key in self._arg_dict_.keys()]

		Logger.plain_text(tpl.format(*tar_values))


# helper class for profiling a module
class Prf:

	configure_file = None

	def generate_input(self, batch_size=1):
		return torch.rand(batch_size, 3, 256, 256)

	# profile single module, get output, number of parameters, and macs
	# if the module has "block" attribute, profile the "block"
	def profile(self, x):
		if hasattr(self, "block"):
			return Prf.profile_list(self.block, x)
		y, paras, macs = x, 0.0, 0.0

		if isinstance(self, nn.Module):
			paras = sum([p.numel() for p in self.parameters()])
			y = self(x)

		return y, paras, macs

	# profile module list or single module(with profile method) recursively
	@staticmethod
	def profile_list(module_list, x):
		n_paras = n_macs = 0.0
		# if module list
		if isinstance(module_list, nn.Sequential) \
				or isinstance(module_list, list) \
				or isinstance(module_list, tuple):
			for m in module_list:
				x, l_p, l_macs = Prf.profile_list(m, x)
				n_macs += l_macs
				n_paras += l_p
		# if single module
		else:
			if hasattr(module_list, "profile"):
				x, n_paras, n_macs = module_list.profile(x)
			else:
				n_paras = sum([p.numel() for p in module_list.parameters()])

		return x, n_paras, n_macs


# region helper classes


# helper for configuration
class Opt:
	# get value by key
	@staticmethod
	def get(key):
		# default_exist, default_value = get_default_value(*args, **kwargs)
		# return getattr(_opts, key, default_value) if default_exist else getattr(_opts, key)
		return getattr(sys.modules["_opts_"], key)

	# set value by key
	@staticmethod
	def set(key, value):
		return setattr(sys.modules["_opts_"], key, value)

	# get/set target configure name
	#  Group One:
	#     name: sub_group_name_1 (determine the used configure name)
	#     sub_group_name_1:
	#         key1: val1
	#         key2: val2
	#     sub_group_name_2:
	#         ...
	#  Group Two：
	#  ...
	@staticmethod
	def get_target_subgroup_name(cfg_path, name_label="name"):
		return Opt.get(f"{cfg_path}.{name_label}")

	@staticmethod
	def set_target_subgroup_name(cfg_path, value, name_label="name"):
		Opt.set(f"{cfg_path}.{name_label}", value)

	@staticmethod
	def get_by_cfg_path(cfg_path, key, name_label="name", subgroup_name=None):
		subgroup_name = subgroup_name or Opt.get_target_subgroup_name(cfg_path, name_label=name_label)
		return Opt.get(f"{cfg_path}.{subgroup_name}.{key}")

	@staticmethod
	def set_by_cfg_path(cfg_path, key, value, name_label="name", subgroup_name=None):
		subgroup_name = subgroup_name or Opt.get_target_subgroup_name(cfg_path, name_label=name_label)
		Opt.set(f"{cfg_path}.{subgroup_name}.{key}", value)

	@staticmethod
	# help to get or set arguments
	def add_arguments(pth_tpl: str, keys: tuple, types: tuple, defaults: tuple):
		arg_path_tpl = "--" + pth_tpl.replace("_", "-")
		names = tuple(arg_path_tpl.format(key_.replace("_", "-")) for key_ in keys)
		group_name = pth_tpl[:pth_tpl.rindex(".")]

		group = sys.modules["_parser_"].add_argument_group(
			title=group_name,
			description=group_name
		)

		for name_, type_, default_ in zip(names, types, defaults):
			if type(type_) == tuple:
				group.add_argument(
					name_,
					type=type_[0],
					default=default_,
					help=name_,
					nargs="+"
				)
			elif type_ is bool:
				group.add_argument(
					name_,
					action="store_true",
					default=default_,
					help=name_
				)
			else:
				group.add_argument(
					name_,
					type=type_,
					default=default_,
					help=name_
				)
		return group

	@staticmethod
	def get_argument_values(pth_tpl, keys, defaults):
		return tuple(
			getattr(sys.modules["_opts_"], pth_tpl.format(key_), default_) for key_, default_ in zip(keys, defaults))

	@staticmethod
	def get_argument_dict(pth_tpl, keys, defaults):
		return {key_: getattr(pth_tpl.format(key_), default_) for key_, default_ in zip(keys, defaults)}


# helper class for dictionary operations
class Dct:
	# pop some elements from the dictionary
	@staticmethod
	def pop_keys(dictionary, pop_keys) -> dict:
		return {key: value for key, value in dictionary.items() if key not in pop_keys}

	# select some elements from the dictionary
	@staticmethod
	def select_keys(dictionary, keys) -> dict:
		return {key: value for key, value in dictionary.items() if key in keys}

	# change key names
	@staticmethod
	def change_key_names(dictionary, old_names, new_names) -> None:
		for old_, new_ in zip(old_names, new_names):
			val = dictionary.pop(old_)
			dictionary[new_] = val

	# change key values
	@staticmethod
	def change_key_values(dictionary, keys, values) -> None:
		for i in range(len(keys)):
			dictionary[keys[i]] = values[i]

	# update the target dictionary by the source dictionary, if target[key] == None and key in source
	@staticmethod
	def update_key_values(target, source) -> dict:
		return {
			key_: (source[key_] if target[key_] is None and key_ in source else target[key_])
			for key_ in target}


# helper for method parameters
class Par:
	# remove context parameters: self, __class__, enable, args, kwargs
	# and, move parameters in kwargs to root directory
	@staticmethod
	def purify(dictionary):
		if "self" in dictionary:
			dictionary.pop("self")
		if "__class__" in dictionary:
			dictionary.pop("__class__")
		if "args" in dictionary:
			dictionary.pop("args")
		if "enable" in dictionary:
			dictionary.pop("enable")
		if "kwargs" in dictionary:
			kwargs = dictionary.pop("kwargs")
			dictionary.update(kwargs)
		return dictionary

	# get "default" parameter value from args or kwargs
	# if args exists, default = args[0]
	# else if "default" in kwargs, default = kwargs["default"]
	@staticmethod
	def get_default_value(*args, **kwargs):
		# get default value
		default_exist = False
		default_value = None
		if args:
			default_exist = True
			default_value = args[0]
		elif kwargs and "default" in kwargs:
			default_exist = True
			default_value = kwargs["default"]
		return default_exist, default_value

	# 1 construct parameters: update None values from obj.__dict__
	# 2 additional processes
	# 3 call super class init method
	@staticmethod
	def init_helper(obj, para_dict, ancestor_cls, pre_handler=None):
		para_dict = Dct.update_key_values(target=para_dict, source=obj.__dict__)
		if pre_handler:
			pre_handler(para_dict)
		if isinstance(ancestor_cls, super):
			ancestor_cls.__init__(**para_dict)
		else:
			ancestor_cls.__init__(obj, **para_dict)

# endregion
