import sys
from .logger import Logger
from .type_utils import Cfg, Dsp

from typing import Union


class Entry:

	# region Group definitions

	Common = "Common"

	Transform = "Transform"
	VideoReader = "VideoReader"
	VideoTransform = "VideoTransform"
	Sampler = "Sampler"
	Collate = "Collate"
	Dataset = "Dataset"

	DataLoader = "DataLoader"

	Activation = "Activation"
	Norm = "Norm"
	Layer = "Layer"

	Anchor = "Anchor"
	Matcher = "Matcher"

	ClassificationModel = "ClassificationModel"
	DetectionModel = "DetectionModel"
	SegmentationModel = "SegmentationModel"
	VideoClassificationModel = "VideoClassificationModel"

	Model = "Model"

	EMAConfigure = "EMAConfiture"

	Optimizer = "Optimizer"
	Scheduler = "Scheduler"
	Loss = "Loss"
	Metric = "Metric"

	LossLandscape = "LossLandscape"
	Engine = "Engine"

	dict_entry = {
		Common: {},

		Transform: {},
		VideoReader: {},
		Sampler: {},
		Collate: {},
		Dataset: {},

		DataLoader: {},

		Activation: {},
		Norm: {},
		Layer: {},

		Anchor: {},
		Matcher: {},

		ClassificationModel: {},
		DetectionModel: {},
		SegmentationModel: {},
		VideoClassificationModel: {},

		Model: {},
		EMAConfigure: {},

		Optimizer: {},
		Scheduler: {},
		Loss: {},
		Metric: {},

		Engine: {},

		LossLandscape: {},
	}

	_dict_cfg_labels = {
		Collate: ["train", "val", "eval"]
	}

	# endregion

	@staticmethod
	def register_entry(category):
		def func(cls):
			if cls.__name__ in Entry.dict_entry[category]:
				raise ValueError(f"Cannot register duplicate entry: {category}-{cls.__name__} ")
			Entry.dict_entry[category][cls.__name__] = cls
			return cls
		return func

	# region configure related functions

	@staticmethod
	def __add_group_arguments(category):
		if "_parser_" not in sys.modules:
			return
		group = sys.modules["_parser_"].add_argument_group(
			title=category,
			description=category
		)
		if category in Entry._dict_cfg_labels:
			for label in Entry._dict_cfg_labels[category]:
				group.add_argument(
					f"--{category}.{label}",
					type=str,
					default="undefined",
					help=label,
				)
		else:
			group.add_argument(
				f"--{category}.name",
				type=str,
				default="undefined",
				help="name",
			)

	@staticmethod
	def add_arguments():
		for category, entry_dict in Entry.dict_entry.items():
			if category != Entry.Model and len(entry_dict) == 0:
				continue
			Entry.__add_group_arguments(category)
			for entry in entry_dict.values():
				if isinstance(entry, type) and issubclass(entry, Cfg):
					entry.add_arguments()

	@staticmethod
	def __print_group_arguments(category):
		Logger.plain_text(category + ":")
		if category in Entry._dict_cfg_labels:
			for label in Entry._dict_cfg_labels[category]:
				Logger.plain_text(f"  {label}: undefined")
		else:
			Logger.plain_text("  name: undefined")

	@staticmethod
	def print_arguments():
		for category, entry_dict in Entry.dict_entry.items():
			if category != Entry.Model and len(entry_dict) == 0:
				continue
			Entry.__print_group_arguments(category)
			for entry in entry_dict.values():
				if isinstance(entry, type) and issubclass(entry, Cfg):
					entry.print_arguments()

	# endregion

	@staticmethod
	def add_disp_keys():
		for entry_dict in Entry.dict_entry.values():
			for entry in entry_dict.values():
				if isinstance(entry, type) and issubclass(entry, Dsp):
					entry.add_disp_keys()

	@staticmethod
	def get_entity(category, entry_name=None, label_name="name", *args, **kwargs):
		from utils.type_utils import Opt

		# region dataloader entry

		if category == Entry.DataLoader:
			from data import DataLoaderUtil
			if entry_name == "train":
				return DataLoaderUtil.get_train_loader()
			elif entry_name == "val":
				return DataLoaderUtil.get_val_loader()
			elif entry_name == "eval":
				return DataLoaderUtil.get_eval_loader()
			elif entry_name == "train_val":
				return DataLoaderUtil.get_train_val_loader()
			else:
				return None

		# endregion

		# use entry_name or get it by label_name
		entry_name = entry_name or Opt.get(f"{category}.{label_name}")

		# region Model Entry

		if category == Entry.Model:
			if entry_name == "ClassificationModel":
				return Entry.get_entity(Entry.ClassificationModel)
			elif entry_name == "DetectionModel":
				return Entry.get_entity(Entry.DetectionModel)
			elif entry_name == "SegmentationModel":
				return Entry.get_entity(Entry.SegmentationModel)
			elif entry_name == "VideoClassificationModel":
				return Entry.get_entity(Entry.VideoClassificationModel)
			else:
				raise ValueError(f"Unsupported model {entry_name}")

		# endregion

		# region optimizer

		if category == Entry.Optimizer:
			from optim import OptimUtil
			return OptimUtil.get_optimizer(entry_name, *args, **kwargs)

		# endregion

		# region model

		if category == Entry.ClassificationModel:
			from model.classification import ClassificationModelUtil
			return ClassificationModelUtil.get_model(entry_name, *args, **kwargs)

		if category == Entry.DetectionModel:
			from model.detection import DetectionModelUtil
			return DetectionModelUtil.get_model(entry_name, *args, **kwargs)

		if category == Entry.SegmentationModel:
			from model.segmentation import SegmentationModelUtil
			return SegmentationModelUtil.get_model(entry_name, *args, **kwargs)

		if category == Entry.VideoClassificationModel:
			from model.video_classification import VideoClassificationModelUtil
			return VideoClassificationModelUtil.get_model(entry_name, *args, **kwargs)

		# endregion

		rel = Entry.dict_entry[category][entry_name]
		return rel(*args, **kwargs) if isinstance(rel, type) else rel  # return object or function

	@staticmethod
	def get_entity_instance(
			category: str,
			entry_name: Union[str, type, object],
			*args,
			**kwargs
	):
		if entry_name is None or isinstance(entry_name, str):
			return Entry.get_entity(category, entry_name=entry_name, *args, **kwargs)
		elif isinstance(entry_name, type):
			return entry_name(*args, **kwargs)
		else:
			return entry_name


import common
import engine
