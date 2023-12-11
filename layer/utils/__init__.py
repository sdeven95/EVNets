from .init_params_util import InitParaUtil

from typing import Union


class LayerUtil:
	@staticmethod
	# split parameters into two groups, one is parameters with decay,
	# other is parameters without decay(bias and norm layer bias and weights)
	def split_parameters(named_parameters, weight_decay=0.0, no_decay_bn_filter_bias=False):
		with_decay = []
		without_decay = []
		if isinstance(named_parameters, list):
			for n_parameter in named_parameters:
				for p_name, param in n_parameter:
					# biases and normalization layer parameters are of one dimension
					if param.requires_grad and len(param.shape) == 1 and no_decay_bn_filter_bias:
						without_decay.append(param)
					else:
						with_decay.append(param)

		else:
			for p_name, param in named_parameters:
				# biases and normalization layer parameters are of one dimension
				if param.requires_grad and len(param.shape) == 1 and no_decay_bn_filter_bias:
					without_decay.append(param)
				else:
					with_decay.append(param)

		param_list = [{"params": with_decay, "weight_decay": weight_decay}]
		if without_decay:
			param_list.append({"params": without_decay, "weight_decay": 0.0})

		return param_list
