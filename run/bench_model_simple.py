from utils.opt_setup_utils import OptSetup
from utils.entry_utils import Entry
import time
import torch


class BenchModelSimple:
	@staticmethod
	def run(model_cls, batch_size=16, iterations=30, warm_up=10):
		OptSetup.get_arguments(model_cls)

		model: torch.nn.Module = Entry.get_entity(Entry.Model)
		model.eval()
		model.training = False

		cuda_available = torch.cuda.is_available()
		print("\n-----------Benchmark-----------\n")
		input_data = model.generate_input(batch_size=batch_size)
		if cuda_available:
			model.cuda()
			input_data = input_data.cuda(non_blocking=True)
		if warm_up:
			for i in range(warm_up):
				model(input_data)
		tick_start = time.time()
		for i in range(iterations):
			model(input_data)
		if cuda_available:
			torch.cuda.synchronize()
		tick_end = time.time()

		print(f"batch_size {batch_size} throughput {iterations * batch_size / (tick_end - tick_start)}")
