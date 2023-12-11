import torch.nn as nn
from layer.builtins import LinearLayer
from layer.normalization import BaseNorm

# -------- Helpers to initialize module parameters --------------------


class InitParaUtil:

	supported_conv_inits = ["kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform", "normal", "trunc_normal"]
	supported_fc_inits = ["kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform", "normal", "trunc_normal"]

	@staticmethod
	# assist: helper function to initialize neural network module parameters
	# Note: biases are initialized to zeros
	def __init_layers(module, init_method="kaiming_normal", std_val=None):
		"""
		Helper function to initialize neural network module
		"""
		init_method = init_method.lower()
		if init_method == "kaiming_normal":
			if module.weight is not None:
				nn.init.kaiming_normal_(module.weight, mode="fan_out")
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif init_method == "kaiming_uniform":
			if module.weight is not None:
				nn.init.kaiming_uniform_(module.weight, mode="fan_out")
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif init_method == "xavier_normal":
			if module.weight is not None:
				nn.init.xavier_normal_(module.weight)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif init_method == "xavier_uniform":
			if module.weight is not None:
				nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif init_method == "normal":
			if module.weight is not None:
				std = 1.0 / module.weight.size(1) if std_val is None else std_val
				nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif init_method == "trunc_normal":
			if module.weight is not None:
				std = 1.0 / module.weight.size(1) if std_val is None else std_val
				nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		else:
			raise KeyError(f"Don't support initialization method: {init_method}")

	@classmethod
	# assist: initialize convolution layer parameters
	def initialize_conv_layer(cls, module, init_method="kaiming_normal", std_val=0.01):
		cls.__init_layers(module=module, init_method=init_method, std_val=std_val)

	@classmethod
	# assist: initialize full-connected layer parameters
	def initialize_fc_layer(cls, module, init_method="normal", std_val=0.01):
		"""Helper function to initialize fully-connected layer"""
		if hasattr(module, "layer"):
			cls.__init_layers(module=module.layer, init_method=init_method, std_val=std_val)
		else:
			cls.__init_layers(module=module, init_method=init_method, std_val=std_val)

	@staticmethod
	# assist: initialize normalization layer
	def initialize_norm_layers(module) -> None:
		"""Helper function to initialize normalization layer"""

		def _init_fn(inner_module):
			if hasattr(inner_module, "weight") and inner_module.weight is not None:
				nn.init.ones_(inner_module.weight)
			if hasattr(inner_module, "bias") and inner_module.bias is not None:
				nn.init.zeros_(inner_module.bias)

		_init_fn(inner_module=module.layer) if hasattr(module, "layer") else _init_fn(inner_module=module)

	@classmethod
	# initialize modules parameters by configurations
	def initialize_weights(cls, model) -> None:
		"""Helper function to initialize different layer in a model"""
		# weight initialization
		conv_init_type = model.conv_init
		linear_init_type = model.linear_init

		conv_std = model.conv_init_std_dev
		linear_std = model.linear_init_std_dev
		group_linear_std = model.group_linear_init_std_dev

		modules = model.modules() if hasattr(model, "modules") else model
		if isinstance(modules, nn.Sequential):
			for m in modules:
				if isinstance(m, (nn.Conv2d, nn.Conv3d)):
					cls.initialize_conv_layer(
						module=m, init_method=conv_init_type, std_val=conv_std
					)
				elif isinstance(m, BaseNorm):
					cls.initialize_norm_layers(module=m)
				elif isinstance(m, (nn.Linear, LinearLayer)):
					cls.initialize_fc_layer(
						module=m, init_method=linear_init_type, std_val=linear_std
					)
				# elif isinstance(m, GroupLinear):
				# 	initialize_fc_layer(
				# 		module=m, init_method=linear_init_type, std_val=group_linear_std
				# 	)
		else:
			if isinstance(modules, (nn.Conv2d, nn.Conv3d)):
				cls.initialize_conv_layer(
					module=modules, init_method=conv_init_type, std_val=conv_std
				)
			elif isinstance(modules, BaseNorm):
				cls.initialize_norm_layers(module=modules)
			elif isinstance(modules, (nn.Linear, LinearLayer)):
				cls.initialize_fc_layer(
					module=modules, init_method=linear_init_type, std_val=linear_std
				)
			# elif isinstance(modules, GroupLinear):
			# 	initialize_fc_layer(
			# 		module=modules, init_method=linear_init_type, std_val=group_linear_std
			# 	)

	# -------- End of helper to initialize module parameters --------------
