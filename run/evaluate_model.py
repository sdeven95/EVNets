import torch

from common import Dev, DDP, Common
from utils.entry_utils import Entry
from utils.logger import Logger

from engine.eval_engine import Evaluator

from.utils import RunUtil


class EvaluateModel:
	@staticmethod
	def runner():
		dev = Dev()
		ddp = DDP()
		cmm = Common()

		# -------- dataloader --------------
		eval_loader = Entry.get_entity(Entry.DataLoader, "eval")

		# --------- model ------------------
		model = Entry.get_entity(Entry.Model)
		memory_format = torch.channels_last if cmm.channel_last else torch.contiguous_format
		if dev.num_gpus <= 1:
			model = model.to(device=dev.device, memory_format=memory_format)
		elif ddp.use_distributed:
			model = model.to(device=dev.device, memory_format=memory_format)
			model = torch.nn.parallel.DistributedDataParallel(
				module=model,
				device_ids=[dev.device_id],
				output_device=dev.device_id,
			)
			Logger.log("Using DistributedDataParallel for evaluation")
		else:
			model = model.to(memory_format=memory_format)
			model = torch.nn.DataParallel(model)
			model = model.to(dev.device)
			Logger.log("Using DataParallel for evaluation")

		# run evaluation
		eval_engine = Evaluator(model=model, eval_loader=eval_loader)
		eval_engine.run()

	@classmethod
	def run(cls, model_cls):
		RunUtil.start_worker(model_cls, cls.runner, is_training=False)
