from utils.opt_setup_utils import OptSetup
from utils.entry_utils import Entry
from utils.type_utils import Opt
from utils.pytorch_to_coreml import CoremlConvertor
from utils.logger import Logger
from engine import Benchmark

from torch.cuda.amp import autocast
import time
import torch


class BenchModel:
	@staticmethod
	def cpu_timestamp(**kwargs):
		return time.perf_counter()

	@staticmethod
	def cuda_timestamp(cuda_sync=False):
		if cuda_sync:
			torch.cuda.synchronize()
		return time.perf_counter()

	@staticmethod
	def step(time_fn, model, example_input, autocast_enable=False):
		start_time = time_fn()
		with autocast(autocast_enable):
			model(example_input)
		end_time = time_fn(cuda_sync=True)
		return end_time - start_time

	@classmethod
	def run(cls, model_cls):
		OptSetup.get_arguments(model_cls)
		bench = Benchmark()

		# not support sync batch normalization
		norm_layer_name = Opt.get_target_subgroup_name(Entry.Norm)
		if "sync" in norm_layer_name.lower():
			Opt.set_target_subgroup_name(Entry.Norm, "BatchNorm2d")

		# if cuda is available
		time_fn = cls.cuda_timestamp if torch.cuda.is_available() else cls.cpu_timestamp
		autocast_enable = torch.cuda.is_available() and Opt.get(f"{Entry.Common}.Common.mixed_precision")

		# prepare model
		model: torch.nn.Module = Entry.get_entity(Entry.Model)
		model.eval()
		model.training = False

		# create random input tensor, print model description
		example_input: torch.Tensor = model.generate_input(batch_size=bench.batch_size)
		if hasattr(model, "profile_model"):
			model.profile_model(example_input)

		time.sleep(5)

		# use jit model
		if bench.use_jit_model:
			converted_models_dict = CoremlConvertor.convert_pytorch_to_coreml(
				pytorch_model=model,
				jit_model_only=True,
			)
			model = converted_models_dict["jit"]

		# move model and input to right device
		if torch.cuda.is_available():
			model.cuda()
			example_input = example_input.cuda(non_blocking=True)

		with torch.no_grad():
			# warmup, don't accumulate time ( making gpu complete some optimization)
			for i in range(bench.warmup_iterations):
				cls.step(
					time_fn=time_fn,
					model=model,
					example_input=example_input,
					autocast_enable=autocast_enable,
				)

			# run benchmark
			n_steps_time, n_samples = 0.0, 0
			for i in range(bench.n_iterations):
				step_time = cls.step(
					time_fn=time_fn,
					model=model,
					example_input=example_input,
					autocast_enable=autocast_enable,
				)
				n_steps_time += step_time
				n_samples += bench.batch_size

		Logger.info(f"Number of samples processed per second: {(n_samples / n_steps_time):.3f}")
