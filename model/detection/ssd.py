from layer.utils import InitParaUtil
# from utils.type_utils import cfg
from utils.entry_utils import Entry
from utils.logger import Logger
from utils import Util

from torch import nn, Tensor
import torch
from torchvision.ops.boxes import batched_nms

from ..classification import BaseEncoder

from detection.anchor import SSDAnchorGenerator
from detection.matcher import SSDMatcher

from . import BaseDetection, DetectionPredTuple
from layer import ConvLayer2D, AdaptiveAvgPool2d
from layer.detections import SSDHead
from model.classification.cvnets.mobilenetv1 import SeparableConv

from typing import Dict, Tuple, Optional, Union, Any, List


@Entry.register_entry(Entry.DetectionModel)
class SingleShotMaskDetector(BaseDetection):
	__slots__ = [
		"proj_channels",
		"min_box_size", "max_box_size", "center_variance", "size_variance", "iou_threshold",   # depreciated
		"conf_threshold", "top_k", "objects_per_image", "nms_iou_threshold",   # inference related arguments
		"fpn_out_channels", "use_fpn",
	]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
				[512, 256, 256, 128, 128, 64],
				0.1, 1.05, None, None, None,
				0.01, 400, 200, 0.5,
				256, False,
			]
	_types_ = [
				(int, ),
				float, float, float, float, float,
				float, int, int, float,
				int, bool,
			]

	coordinates = 4  # 4 coordinates (x1, y1, x2, y2) or (x, y, w, h)

	def __init__(self, encoder: BaseEncoder):
		super().__init__(encoder=encoder)

		# delete layers that are not required in detection network
		self.encoder.tail = None
		self.encoder.classifier = None

		self.anchor: SSDAnchorGenerator = Entry.get_entity(Entry.Anchor)
		self.matcher: SSDMatcher = Entry.get_entity(Entry.Matcher)

		assert len(self.proj_channels) == len(self.anchor.output_strides), \
			f"len(proj_channels) != len(out_strides), Got: {len(self.proj_channels)} != {len(self.anchor.output_strides)}"

		# set up layers
		extra_layers = {}
		enc_channels_list = []
		in_channels = self.enc_l5_channels  # layer_5 out_channels
		# output_strides: [ 16, 32, 64, 128, 256, -1 ]
		# proj_channels: [512, 256, 256, 128, 128, 64]
		for idx, os in enumerate(self.anchor.output_strides):
			out_channels = self.proj_channels[idx]
			if os == 8:
				enc_channels_list.append(self.enc_l3_channels)
			elif os == 16:
				enc_channels_list.append(self.enc_l4_channels)
			elif os == 32:
				enc_channels_list.append(self.enc_l5_channels)
			elif os > 32 and os != -1:
				extra_layers["os_{}".format(os)] = SeparableConv(
					in_channels=in_channels,
					out_channels=out_channels,
					kernel_size=3,
					use_norm=True,
					use_act=True,
					stride=2,
				)
				enc_channels_list.append(out_channels)
				in_channels = out_channels
			elif os == -1:
				extra_layers["os_{}".format(os)] = nn.Sequential(
					AdaptiveAvgPool2d(output_size=1),
					ConvLayer2D(
						in_channels=in_channels,
						out_channels=out_channels,
						kernel_size=1,
						use_norm=False,
						use_act=True,
					),
				)
				enc_channels_list.append(out_channels)
				in_channels = out_channels
			else:
				raise NotImplementedError

		self.extra_layers = None if not extra_layers else nn.ModuleDict(extra_layers)
		if self.extra_layers is not None:
			for layer in self.extra_layers.modules():
				if isinstance(layer, nn.Conv2d):
					InitParaUtil.initialize_conv_layer(module=layer, init_method="xavier_uniform")

		# if we use ffn instead of projection
		self.fpn = None
		if self.use_fpn:
			from layer import FeaturePyramidNetwork
			self.fpn = FeaturePyramidNetwork(
				in_channels=enc_channels_list,
				output_strides=self.anchor.output_strides,
				out_channels=self.fpn_out_channels,
			)
			# update the enc_channels_list
			enc_channels_list = [self.fpn_out_channels] * len(self.anchor.output_strides)
			# for FPN, we do not need to do projections
			self.proj_channels = enc_channels_list

		self.ssd_heads = nn.ModuleList()

		for os, in_dim, proj_dim, n_anchors, step in zip(
			self.output_strides,
			enc_channels_list,
			self.proj_channels,
			self.anchor.num_anchors_per_out_stride,
			self.anchor.steps,
		):
			self.ssd_heads += [
				SSDHead(
					in_channels=in_dim,
					n_classes=self.n_classes,
					n_coordinates=self.coordinates,
					n_anchors=n_anchors,
					proj_channels=proj_dim,
					kernel_size=3 if os != -1 else 1,
					step=step,
				)
			]

	def get_backbone_features(self, x: Tensor) -> Dict[str, Tensor]:
		# extract features from the backbone network
		enc_end_points: Dict = self.encoder.extract_end_points_all(x)

		end_points: Dict = dict()
		for idx, os in enumerate(self.output_strides):
			if os == 8:
				end_points[f"os_{os}"] = enc_end_points.pop("out_layer3")
			elif os == 16:
				end_points[f"os_{os}"] = enc_end_points.pop("out_layer4")
			elif os == 32:
				end_points[f"os_{os}"] = enc_end_points.pop("out_layer5")
			else:
				x = end_points[f"os_{self.output_strides[idx - 1]}"]  # get previous layer output
				end_points[f"os_{os}"] = self.extra_layers[f"os_{os}"](x)

		if self.fpn is not None:
			# apply Feature Pyramid Network
			end_points = self.fpn(end_points)

		return end_points

	def ssd_forward(
		self,
		end_points: Dict[str, Tensor],
		device: Optional[torch.device] = torch.device("cpu"),
	) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]:

		locations = []
		confidences = []
		anchors = []

		for os, ssd_head in zip(self.output_strides, self.ssd_heads):
			x = end_points["os_{}".format(os)]
			fm_h, fm_w = x.shape[2:]
			loc, pred = ssd_head(x)

			locations.append(loc)
			confidences.append(pred)

			anchors_fm_ctr = self.anchor_box_generator(
				fm_height=fm_h, fm_width=fm_w, fm_output_stride=os, device=device
			)
			anchors.append(anchors_fm_ctr)

		locations = torch.cat(locations, dim=1)
		confidences = torch.cat(confidences, dim=1)

		anchors = torch.cat(anchors, dim=0)
		anchors = anchors.unsqueeze(dim=0)

		return confidences, locations, anchors

	def forward(
		self, x: Union[Tensor, Dict]
	) -> Union[Tuple[Tensor, ...], Tuple[Any, ...], Dict]:
		if isinstance(x, Dict):
			input_tensor = x["image"]
		elif isinstance(x, Tensor):
			input_tensor = x
		else:
			raise NotImplementedError(
				"Input to SSD should be either a Tensor or a Dict of Tensors"
			)

		device = input_tensor.device
		backbone_end_points: Dict = self.get_backbone_features(input_tensor)

		if not Util.is_coreml_conversion():  # ???
			confidences, locations, anchors = self.ssd_forward(
				end_points=backbone_end_points, device=device
			)

			output_dict = {"scores": confidences, "locations": locations}

			if not self.training:
				# compute the detection results during evaluation
				scores = nn.Softmax(dim=-1)(confidences)
				boxes = self.match.convert_to_boxes(
					pred_locations=locations, anchors=anchors
				)

				detections = self.post_process_detections(boxes=boxes, scores=scores)
				output_dict["detections"] = detections

			return output_dict
		else:
			return self.ssd_forward(end_points=backbone_end_points)

	@torch.no_grad()
	def predict(self, x: Tensor) -> DetectionPredTuple:
		"""Predict the bounding boxes given an image tensor"""
		bsz, channels, width, height = x.shape
		assert bsz == 1, f"Prediction is supported with a batch size of 1 in {self.__class__.__name__}"

		enc_end_points: Dict = self.get_backbone_features(x)
		confidences, locations, anchors = self.ssd_forward(
			end_points=enc_end_points, device=x.device
		)

		scores = nn.Softmax(dim=-1)(confidences)

		boxes = self.match_prior.convert_to_boxes(
			pred_locations=locations, anchors=anchors
		)
		detections = self.post_process_detections(boxes=boxes, scores=scores)[0]
		return detections

	@torch.no_grad()
	def post_process_detections(
		self, boxes: Tensor, scores: Tensor
	) -> List[DetectionPredTuple]:
		"""Post process detections, including NMS"""
		# boxes [B, N, 4]
		# scores [B, N, n_classes]

		batch_size = boxes.shape[0]
		n_classes = scores.shape[-1]

		device = boxes.device
		box_dtype = boxes.dtype
		scores_dtype = scores.dtype

		results = []
		for b_id in range(batch_size):
			object_labels = []
			object_boxes = []
			object_scores = []

			for class_index in range(1, n_classes):
				probs = scores[b_id, :, class_index]
				mask = probs > self.conf_threshold
				probs = probs[mask]
				if probs.size(0) == 0:
					continue
				masked_boxes = boxes[b_id, mask, :]

				# keep only top-k indices
				num_topk = min(self.top_k, probs.size(0))
				probs, idxs = probs.topk(num_topk)
				masked_boxes = masked_boxes[idxs, ...]

				object_boxes.append(masked_boxes)
				object_scores.append(probs)
				object_labels.append(
					torch.full_like(
						probs, fill_value=class_index, dtype=torch.int64, device=device
					)
				)

			if len(object_scores) == 0:
				output = DetectionPredTuple(
					labels=torch.empty(0, device=device, dtype=torch.long),
					scores=torch.empty(0, device=device, dtype=scores_dtype),
					boxes=torch.empty(0, 4, device=device, dtype=box_dtype),
				)
			else:
				# concatenate all results
				object_scores = torch.cat(object_scores, dim=0)
				object_boxes = torch.cat(object_boxes, dim=0)
				object_labels = torch.cat(object_labels, dim=0)

				# non-maximum suppression
				keep = batched_nms(
					object_boxes, object_scores, object_labels, self.nms_iou_threshold
				)
				keep = keep[: self.objects_per_image]

				output = DetectionPredTuple(
					labels=object_labels[keep],
					scores=object_scores[keep],
					boxes=object_boxes[keep],
				)
			results.append(output)
		return results

	def profile_backbone(self, x: Tensor) -> Tuple[Dict[str, Tensor], float, float]:
		params, macs = 0.0, 0.0
		enc_end_points, p, m = self.encoder.profile_model(x, is_classification=False)
		params += p
		macs += m

		end_points = dict()
		for idx, os in enumerate(self.output_strides):
			if os == 8:
				end_points[f"os_{os}"] = enc_end_points.pop("out_layer3")
			elif os == 16:
				end_points[f"os_{os}"] = enc_end_points.pop("out_layer4")
			elif os == 32:
				end_points[f"os_{os}"] = enc_end_points.pop("out_layer5")
			else:
				x = end_points[f"os_{self.output_strides[idx - 1]}"]
				x, p, m = self.extra_layers[f"os_{os}"].profile(x=x)
				end_points[f"os_{os}"] = x

				params += p
				macs += m

		if self.fpn is not None:
			end_points, p, m = self.fpn.profile(end_points)
			params += p
			macs += m

			enc_str = (
				Logger.text_colors["logs"]
				+ Logger.text_colors["bold"]
				+ "FPN  "
				+ Logger.text_colors["end_color"]
			)
			print("{:>45}".format(enc_str))
			print(
				"{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
					self.fpn.__class__.__name__,
					"Params",
					round(p / 1e6, 3),
					"MACs",
					round(m / 1e6, 3),
				)
			)
			Logger.single_dash_line()
		return end_points, params, macs

	def profile_model(self, x: Tensor) -> None:
		"""
		This function computes layer-wise FLOPs and parameters for SSD

		.. note::
			 Model profiling is for reference only and may contain errors as it relies heavily on user
			 to implement the underlying functions accurately.
		"""
		overall_params, overall_macs = 0.0, 0.0
		x_fvcore = x.clone()

		Logger.log("Model statistics for an input of size {}".format(x.size()))
		Logger.double_dash_line(dashes=65)
		print("{:>35} Summary".format(self.__class__.__name__))
		Logger.double_dash_line(dashes=65)

		# profile encoder
		enc_str = (
			Logger.text_colors["logs"]
			+ Logger.text_colors["bold"]
			+ "Encoder  "
			+ Logger.text_colors["end_color"]
		)
		print("{:>45}".format(enc_str))
		backbone_end_points, encoder_params, encoder_macs = self.profile_backbone(x=x)

		ssd_head_params = ssd_head_macs = 0.0
		for os, ssd_head in zip(self.output_strides, self.ssd_heads):
			_, p, m = ssd_head.profile(x=backbone_end_points[f"os_{os}"])
			ssd_head_params += p
			ssd_head_macs += m

		overall_params += encoder_params + ssd_head_params
		overall_macs += encoder_macs + ssd_head_macs

		ssd_str = (
			Logger.text_colors["logs"]
			+ Logger.text_colors["bold"]
			+ "SSD  "
			+ Logger.text_colors["end_color"]
		)
		print("{:>45}".format(ssd_str))

		print(
			"{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
				self.__class__.__name__,
				"Params",
				round(ssd_head_params / 1e6, 3),
				"MACs",
				round(ssd_head_macs / 1e6, 3),
			)
		)

		Logger.double_dash_line(dashes=65)
		print("{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6))
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
