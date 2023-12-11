from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry


@Entry.register_entry(Entry.Common)
class Common(Cfg, Dsp):
	__slots__ = [
				"seed", "debug_mode",
				"config_file", "result_loc", "run_label", "exp_loc",
				"resume_loc", "auto_resume", "model_state_loc", "model_ema_state_loc",
				"channel_last", "mixed_precision", "grad_clip", "set_grad_to_none",
				"accum_freq", "accum_after_epoch",
				"log_freq", "tensorboard_logging", "bolt_logging",
				"inference_modality", "enable_coreml_compatible_module",
			]
	_cfg_path_ = Entry.Common
	_keys_ = __slots__
	_defaults_ = [
				0, False,
				None, "results", "run_1", None,
				None, True, None, None,
				False, True, None, True,
				1, 0,
				500, False, False,
				"image", False,
			]
	_types_ = [
				int, bool,
				str, str, str, str,
				str, bool, str, str,
				bool, bool, float, bool,
				int, int,
				int, bool, bool,
				str, bool,
			]

	_disp_ = __slots__

	@classmethod
	def add_arguments(cls):
		group = super().add_arguments()

		if group is None:
			return

		from utils.opt_setup_utils import OptSetup

		group.add_argument(
			"--common.override-kwargs",
			nargs="*",
			action=OptSetup.ParseKwargs,
			help="Override arguments. Example. To override the value of --sampler.vbs.crop-size-width, "
			"we can pass override argument as "
			"--common.override-kwargs sampler.vbs.crop_size_width=512 \n "
			"Note that keys in override arguments do not contain -- or -",
		)

	def __repr__(self):
		return super()._repr_by_line()


@Entry.register_entry(Entry.Common)
class DDP(Cfg, Dsp):
	__slots__ = [
				"enable", "rank", "start_rank", "world_size",
				"use_distributed", "dist_url", "dist_port",
				"spawn", "backend", "find_unused_params",
			]

	_cfg_path_ = Entry.Common
	_keys_ = __slots__
	_defaults_ = [
				True, 0, 0, None,
				True, None, 30786,
				True, "nccl", False
			]
	_types_ = [
				bool, int, int, int,
				bool, str, int,
				bool, str, bool
			]

	_disp_ = __slots__

	def __repr__(self):
		return super()._repr_by_line()


@Entry.register_entry(Entry.Common)
class Dev(Cfg, Dsp):
	__slots__ = ["device", "num_gpus", "device_id", "num_cpus"]

	_cfg_path_ = Entry.Common
	_keys_ = __slots__
	_defaults_ = ["cpu", 0, 0, 1]
	_types_ = [str, int, int, int]

	_disp_ = __slots__
