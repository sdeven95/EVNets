from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Engine)
class Benchmark(Cfg, Dsp):
	__slots__ = ["batch_size", "warmup_iterations", "n_iterations", "use_jit_model", ]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [1, 10, 100, False]
	_types_ = [int, int, int, bool]

	_cfg_path_ = Entry.Engine


@Entry.register_entry(Entry.Engine)
class Conversion(Cfg, Dsp):
	__slots__ = ["ckpt_path", "coreml_extn", "input_image_path", "bucket_name", "task_id", "viewers"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [None, "mlmodel", None, None, None, None]
	_types_ = [str, str, str, str, str, (str, )]

	_cfg_path_ = Entry.Engine
