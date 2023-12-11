from fvcore.nn import FlopCountAnalysis, flop_count_table
from utils.entry_utils import Entry
from utils.opt_setup_utils import OptSetup
import torch


class CheckModel:
	@staticmethod
	def run(model_cls):
		OptSetup.get_arguments(model_cls)

		model: torch.nn.Module = Entry.get_entity(Entry.Model)
		model.eval()
		model.training = False

		# print model
		input_data: torch.Tensor = model.generate_input(batch_size=1)
		print("\n-----------Model------------\n")
		print(model)
		print("\n-----------Data Shape ------\n")
		print(f"Input: {input_data.shape}")
		output = model(input_data)
		print(f"Output: {output.shape}")

		# FLOP Analysis
		print("\n------------Flop Analysis-----\n")
		flops = FlopCountAnalysis(model, input_data)
		print(flop_count_table(flops))
