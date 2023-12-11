import argparse
import sys

import yaml
import os
import collections

from .entry_utils import Entry

from .logger import Logger

try:
	collections_abc = collections.abc
except AttributeError:
	collections_abc = collections


class OptSetup:

	DEFAULT_CONFIG_DIR = "config"

	# region helper functions

	# Parse override arguments located after '--common.override-kwargs'
	class ParseKwargs(argparse.Action):
		def __call__(self, parser, namespace, values, option_string=None):
			namespace_dict = vars(namespace)

			if len(values) > 0:
				override_dict = {}
				# values are list of key-value pairs
				for value in values:
					key = None
					try:
						key, value = value.split("=")
					except ValueError:
						Logger.error(
							"For override arguments, a key-value pair of the form key=value is expected"
						)

					if key in namespace_dict:
						value_namespace = namespace_dict[key]
						if value_namespace is None and value is None:
							value = None
						elif value_namespace is None and value is not None:
							# possibly a string or list of strings or list of integers

							# check if string is a list or not
							value = value.split(",")
							if len(value) == 1:
								# it is a string
								value = str(value[0])

								# check if its empty string or not
								if value == "" or value.lower() == "none":
									value = None
							else:
								# it is a list of integers or strings
								try:
									# convert to int
									value = [int(v) for v in value]
								except:
									# pass because it is a string
									pass
						else:
							try:
								if value.lower() == "true":  # check for boolean
									value = True
								elif value.lower() == "false":
									value = False
								else:
									desired_type = type(value_namespace)
									value = desired_type(value)
							except ValueError:
								Logger.warning(f"Type mismatch while over-riding. Skipping key: {key}")
								continue

						override_dict[key] = value
				setattr(namespace, "override_args", override_dict)
			else:
				setattr(namespace, "override_args", None)

	@staticmethod
	def __flatten_yaml_as_dict(d, parent_key="", sep="."):
		items = []
		for k, v in d.items():
			new_key = parent_key + sep + k if parent_key else k
			if isinstance(v, collections_abc.MutableMapping):
				items.extend(OptSetup.__flatten_yaml_as_dict(v, new_key, sep=sep).items())
			else:
				items.append((new_key, v))
		return dict(items)

	@classmethod
	def __load_config_file(cls):
		from common import Common
		comm = Common()

		config_file_name = comm.config_file

		if config_file_name is not None and not os.path.isfile(config_file_name):
			if len(config_file_name.split("/")) == 1:
				# loading files from default config folder
				new_config_file_name = "{}/{}".format(cls.DEFAULT_CONFIG_DIR, config_file_name)
				if not os.path.isfile(new_config_file_name):
					Logger.error(f"Configuration file neither exists at {config_file_name} nor at {new_config_file_name}")
					config_file_name = None
				else:
					config_file_name = new_config_file_name
			else:
				# If absolute path of the file is passed
				if not os.path.isfile(config_file_name):
					Logger.error(f"Configuration file does not exists at {config_file_name}")
					config_file_name = None

		# load configurations from file
		comm.config_file = config_file_name
		if config_file_name:
			with open(config_file_name, "r") as yaml_file:

				import re
				loader = yaml.SafeLoader
				loader.add_implicit_resolver(
					u'tag:yaml.org,2002:float',
					re.compile(u'''^(?:
					 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
					|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
					|\\.[0-9_]+(?:[eE][-+][0-9]+)?
					|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
					|[-+]?\\.(?:inf|Inf|INF)
					|\\.(?:nan|NaN|NAN))$''', re.X),
					list(u'-+0123456789.'))

				# cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
				cfg = yaml.load(yaml_file, Loader=loader)
				flat_cfg = OptSetup.__flatten_yaml_as_dict(cfg)
				for k, v in flat_cfg.items():
					if hasattr(sys.modules["_opts_"], k):
						if v == "None":
							setattr(sys.modules["_opts_"], k, None)
						else:
							setattr(sys.modules["_opts_"], k, v)

		# override arguments
		override_args = getattr(sys.modules["_opts_"], "override_args", None)
		if override_args:
			for override_k, override_v in override_args.items():
				if hasattr(sys.modules["_opts_"], override_k):
					setattr(sys.modules["_opts_"], override_k, override_v)

	# endregion

	@staticmethod
	def print_arguments_template(model_cls):
		Entry.add_arguments()
		Entry.print_arguments()

	@staticmethod
	def get_arguments(model_cls, parse_args=True):  # configure_file = 'test/configure_file.yaml'
		sys.modules["_parser_"] = argparse.ArgumentParser(description="Training arguments", add_help=True)

		Entry.add_disp_keys()
		Entry.add_arguments()

		sys.argv.append(f"--{Entry.Common}.Common.config-file")
		sys.argv.append(model_cls.configure_file)

		if parse_args:
			sys.modules["_opts_"] = sys.modules["_parser_"].parse_args()
			OptSetup.__load_config_file()
			del sys.modules["_parser_"]

	# @staticmethod
	# def print_arguments():
	# 	Entry.print_arguments()

	# @staticmethod
	# def test_get_arguments(
	# 		add_arguments_handler_list,
	# 		configure_file=None,
	# 		override_args=None,
	# 		parse_args=True):
	# 	sys.modules["_parser_"] = argparse.ArgumentParser(description="Training arguments", add_help=True)
	#
	# 	sys.argv.append(f"--{Entry.Common}.Common.config-file")
	# 	sys.argv.append(configure_file)
	#
	# 	if override_args:
	# 		sys.argv.append("--common.override-kwargs")
	# 		for k, v in override_args.items():
	# 			sys.argv.append(f"{k}={v}")
	#
	# 	for handler in add_arguments_handler_list:
	# 		handler()
	#
	# 	if parse_args:
	# 		sys.modules["_opts_"] = sys.modules["_parser_"].parse_args()
	# 		OptSetup.__load_config_file()
	# 		del sys.modules["_parser_"]
