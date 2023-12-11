from utils.type_utils import Cfg, Opt, Prf, Dsp
from utils.entry_utils import Entry
from utils.logger import Logger

from ..utils import MdlUtil
from layer.utils import LayerUtil
from layer.utils.init_params_util import InitParaUtil
from layer.builtins import LinearLayer


import torch.nn as nn
from torch import Tensor

from typing import Optional, Dict


class ClassificationModelUtil:
	@staticmethod
	def get_model(entry_name, *args, **kwargs):
		model = Entry.dict_entry[Entry.ClassificationModel][entry_name](*args, **kwargs)

		if model.finetune_pretrained_model:
			model.update_classifier(n_classes=model.n_pretrained_classes)  # change classifier
			if model.pretrained:
				MdlUtil.load_pretrained_model(model, model.pretrained)  # load state dict
			model.update_classifier(n_classes=model.n_classes)  # recover classifier
		elif model.pretrained:
			MdlUtil.load_pretrained_model(model, model.pretrained)

		if model.freeze_batch_norm:
			MdlUtil.freeze_norm_layers(model)

		return model


class BaseEncoder(Cfg, Dsp, Prf, nn.Module):
	__slots__ = [
				"n_classes", "classifier_dropout", "freeze_batch_norm", "global_pool",
				"pretrained",
				"n_pretrained_classes", "finetune_pretrained_model",
				"conv_init", "linear_init", "conv_init_std_dev", "linear_init_std_dev", "group_linear_init_std_dev"
			]
	_disp_ = __slots__
	_keys_ = __slots__
	_defaults_ = [
				1000, 0.0, False, "mean",
				None,
				1000, False,
				"kaiming_normal", "normal", 0.01, 0.01, 0.01
			]
	_types_ = [
				int, float, bool, str,
				str,
				int, bool,
				str, str, float, float, float
			]

	_cfg_path_ = Entry.ClassificationModel

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		nn.Module.__init__(self)

		self.layer_list = None
		self.head = None
		self.body = None
		self.tail = None
		self.classifier = None
		self.round_nearest = 8

		self.model_conf_dict = dict()

		# Segmentation architectures like Deeplab and PSPNet modifies the strides of the backbone
		# We allow that using output_stride and replace_stride_with_dilation arguments
		self.dilation = 1
		output_stride = kwargs.get("output_stride", None)
		self.dilate_l4 = False
		self.dilate_l5 = False
		if output_stride == 8:
			self.dilate_l4 = True
			self.dilate_l5 = True
		elif output_stride == 16:
			self.dilate_l5 = True

	def update_classifier(self, n_classes):
		linear_init_type = Opt.get_by_cfg_path(Entry.ClassificationModel, "linear_init")
		# get the last layer, and change the output number of classes
		if isinstance(self.classifier, nn.Sequential):
			in_features = self.classifier[-1].in_features
			layer = LinearLayer(
				in_features=in_features, out_features=n_classes, bias=True
			)
			InitParaUtil.initialize_fc_layer(layer, init_method=linear_init_type)
			self.classifier[-1] = layer
		else:
			in_features = self.classifier.in_features
			layer = LinearLayer(
				in_features=in_features, out_features=n_classes, bias=True
			)
			InitParaUtil.initialize_fc_layer(layer, init_method=linear_init_type)

			# re-init head, why???
			head_init_scale = 0.001
			layer.weight.data.mul_(head_init_scale)
			layer.bias.data.mul_(head_init_scale)

			self.classifier = layer

	def forward(self, x):
		if self.layer_list:
			self.layer_list = self.layer_list if isinstance(self.layer_list, nn.Sequential) else nn.Sequential(*self.layer_list)
			x = self.layer_list(x)
		else:
			if self.head:
				x = self.head(x)
			if self.body:
				self.body = self.body if isinstance(self.body, nn.Sequential) else nn.Sequential(*self.body)
				x = self.body(x)
			if self.tail:
				x = self.tail(x)
			if self.classifier:
				x = self.classifier(x)

		return x

	def get_training_parameters(self, weight_decay=0.0, no_decay_bn_filter_bias=False):
		param_list = LayerUtil.split_parameters(self.named_parameters(), weight_decay, no_decay_bn_filter_bias)
		return param_list, [1.0] * len(param_list)

	# get computation cost and print model description
	# only print current layer, don't drill down
	@staticmethod
	def _profile_layers(layers, x, overall_params, overall_macs):
		# make sure to be a list
		if not isinstance(layers, list):
			layers = [layers]

		for layer in layers:
			if not layer:
				continue
			# layer may be a single module with profile method or nn.Sequential (in fact, list or tuple is also ok)
			x, layer_param, layer_macs = Prf.profile_list(layer, x)
			overall_params += layer_param
			overall_macs += layer_macs

			# if layer is nn.Sequential, combine every module names
			if isinstance(layer, nn.Sequential):
				module_name = "\n+".join([l_.__class__.__name__ for l_ in layer])
			else:
				module_name = layer.__class__.__name__
			print(
				"{:} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
					module_name,
					"Params",
					round(layer_param / 1e6, 3),
					"MACs",
					round(layer_macs / 1e6, 3),
				)
			)
			Logger.single_dash_line()
		return x, overall_params, overall_macs

	def profile_model(self, x, is_classification=True):
		overall_params, overall_macs = 0.0, 0.0

		x_fvcore = x.clone()

		if is_classification:
			Logger.log("Model statistics for an input of size {}".format(x.size()))
			Logger.double_dash_line(dashes=65)
			print("{:>35} Summary".format(self.__class__.__name__))
			Logger.double_dash_line(dashes=65)

		out_dict = {}

		cnt = 1
		if self.layer_list:
			for layer in self.layer_list:
				x, overall_params, overall_macs = self._profile_layers(layer, x, overall_params, overall_macs)
				out_dict["out_l" + str(cnt)] = x
				cnt += 1

		if self.head:
			x, overall_params, overall_macs = self._profile_layers(self.head, x, overall_params, overall_macs)
			out_dict["out_head"] = x
		if self.body:
			for layer in self.body:
				x, overall_params, overall_macs = self._profile_layers(layer, x, overall_params, overall_macs)
				out_dict["out_l" + str(cnt)] = x
				cnt += 1
		if self.tail:
			x, overall_params, overall_macs = self._profile_layers(self.tail, x, overall_params, overall_macs)
			out_dict["out_tail"] = x
		# for classifier, don't combine module names as layer name, set layer name to 'Classifier'
		if is_classification:
			classifier_params, classifier_macs = 0.0, 0.0
			if self.classifier:
				x, classifier_params, classifier_macs = Prf.profile_list(self.classifier, x)
				print(
					"{:} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
						"Classifier",
						"Params",
						round(classifier_params / 1e6, 3),
						"MACs",
						round(classifier_macs / 1e6, 3),
					)
				)
			overall_params += classifier_params
			overall_macs += classifier_macs
			out_dict["out_classifier"] = x

			Logger.double_dash_line(dashes=65)
			print(
				"{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6)
			)
			overall_params_py = sum([p.numel() for p in self.parameters()])
			print(
				"{:<20} = {:>8.3f} M".format(
					"Overall parameters (sanity check)", overall_params_py / 1e6
				)
			)

			# Counting Addition and Multiplication as 1 operation
			print(
				"{:<20} = {:>8.3f} M".format(
					"Overall MACs (theoretical)", overall_macs / 1e6
				)
			)

			# compute flops using FVCore
			try:
				# compute flops using FVCore also
				from fvcore.nn import FlopCountAnalysis

				flop_analyzer = FlopCountAnalysis(self.eval(), x_fvcore)
				flop_analyzer.unsupported_ops_warnings(False)
				flop_analyzer.uncalled_modules_warnings(False)
				flops_fvcore = flop_analyzer.total()

				print(
					"{:<20} = {:>8.3f} M".format(
						"Overall MACs (FVCore)**", flops_fvcore / 1e6
					)
				)
				print(
					"\n** Theoretical and FVCore MACs may vary as theoretical MACs do not account "
					"for certain operations which may or may not be accounted in FVCore"
				)
			except Exception:
				pass

			print("Note: Theoretical MACs depends on user-implementation. Be cautious")
			Logger.double_dash_line(dashes=65)

		return out_dict, overall_params, overall_macs

	# --------- the following functions for object detection ----------
	# record output tensor for each layer, could be used to check output shapes of layers
	def extract_end_points_all(
		self,
		x: Tensor,
		use_l5: Optional[bool] = True,
		use_tail: Optional[bool] = False,
	) -> Dict[str, Tensor]:
		out_dict = {}  # Use dictionary over NamedTuple so that JIT is happy
		x = self.head(x)  # 112 x112
		x = self.body["layer1"](x)  # 112 x112
		out_dict["out_layer1"] = x

		x = self.self.body["layer2"](x)  # 56 x 56
		out_dict["out_layer2"] = x

		x = self.self.body["layer3"](x)  # 28 x 28
		out_dict["out_layer3"] = x

		x = self.self.body["layer4"](x)  # 14 x 14
		out_dict["out_layer4"] = x

		if use_l5:
			x = self.self.body["layer5"](x)  # 7 x 7
			out_dict["out_layer5"] = x

			if use_tail:
				x = self.tail(x)
				out_dict["out_tail"] = x
		return out_dict

	def extract_end_points_l4(self, x: Tensor) -> Dict[str, Tensor]:
		return self.extract_end_points_all(x, use_l5=False)

	# extract features
	def extract_features(self, x: Tensor) -> Tensor:
		x = self.head(x)
		x = self.body["layer1"](x)
		x = self.body["layer2"](x)
		x = self.body["layer3"](x)

		x = self.body["layer4"](x)
		x = self.body["layer5"](x)
		if self.tail:
			x = self.tail(x)
		return x
