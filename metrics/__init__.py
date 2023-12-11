from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Metric)
class Metric(Cfg, Dsp):
	__slots__ = [
				"train_metric_names", "val_metric_names",
				"checkpoint_metric", "checkpoint_metric_max",
				"k_best_checkpoints", "save_all_checkpoints", "terminate_ratio"
	]
	_cfg_path_ = Entry.Metric
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [
		["loss", ], ["loss", "top1", "top5"],
		"top1", True,
		5, False, 100.0
	]
	_types_ = [
		(str, ), (str, ),
		str, bool,
		int, bool, float
	]

# obsoleted
# from .stats import Statistics
# from .coco_evaluator import COCOEvaluator
# from .confusion_matrix import ConfusionMatrix
