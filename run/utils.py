import warnings
import sys
import torch

from common import Dev, DDP, Common
from utils import Util
from utils.opt_setup_utils import OptSetup
from utils.file_utils import File
from utils.type_utils import Opt
from utils.entry_utils import Entry
from utils.logger import Logger


class RunUtil:
	@staticmethod
	def distributed_worker(rank, runner, args):
		if "_opts_" not in sys.modules:
			sys.modules["_opts_"] = args["_opts_"]

		# set device
		dev = Dev()
		dev.device_id = rank
		torch.cuda.set_device(rank)
		dev.device = torch.device(f"cuda:{rank}")

		ddp = DDP()
		ddp.rank = args.get("start_rank", 0) + rank  # set rank

		# init distribute group and get returned rank
		Util.distributed_init(ddp)
		runner()

	@classmethod
	def start_worker(cls, model_cls, runner, is_training=True):
		warnings.filterwarnings("ignore")

		OptSetup.get_arguments(model_cls)
		print(sys.modules["_opts_"])

		# create the directory for saving results
		comm = Common()
		comm.exp_loc = "{}/{}".format(comm.result_loc, comm.run_label)
		File.create_directories(comm.exp_loc)

		dev = Dev()
		# get Dev(device, num_gpus, num_cpus)
		Util.device_setup(comm, dev)

		ddp = DDP()
		if dev.num_gpus <= 1:
			ddp.use_distributed = False

		data_workers = Opt.get_by_cfg_path(Entry.Dataset, "workers")
		if ddp.use_distributed and ddp.spawn and torch.cuda.is_available():
			# set world_size
			if ddp.world_size is None or ddp.world_size < 0 or ddp.world_size != dev.num_gpus:
				ddp.world_size = dev.num_gpus
				Logger.log("Setting --ddp.world_size the same as the number of available gpus")

			# set dataset workers
			if data_workers is None or data_workers < 0:
				Opt.set_by_cfg_path(Entry.Dataset, "workers", dev.num_cpus // dev.num_gpus)

			# set start rank
			args = {}
			start_rank = ddp.rank
			ddp.rank = None  # unset ddp.rank, will be reset in distributed_worker()
			args["start_rank"] = start_rank
			ddp.start_rank = start_rank
			args["_opts_"] = sys.modules["_opts_"]

			# start multiprocessing
			torch.multiprocessing.spawn(
				fn=cls.distributed_worker,
				args=(runner, args),
				nprocs=dev.num_gpus,
			)
		else:
			# set dataset_workers
			if data_workers is None or data_workers < 0:
				Opt.set_by_cfg_path(Entry.Dataset, "workers", dev.num_cpus)

			# adjust normalization layer, for evaluation
			if not is_training:
				norm_name = Opt.get_target_subgroup_name(Entry.Norm)
				if "Sync" in norm_name:
					Opt.set_target_subgroup_name(Entry.Norm, "BatchNorm2d")

			# adjust sampler
			sampler_name = Opt.get_target_subgroup_name(Entry.Sampler)
			if "DDP" in sampler_name:
				Opt.set_target_subgroup_name(Entry.Sampler, sampler_name[:-3])

			# adjust the batch_size
			if is_training:
				train_batch_size = Opt.get_by_cfg_path(Entry.Sampler, "train_batch_size")
				Opt.set_by_cfg_path(Entry.Sampler, "train_batch_size", train_batch_size * max(1, dev.num_gpus))
				val_batch_size = Opt.get_by_cfg_path(Entry.Sampler, "val_batch_size")
				Opt.set_by_cfg_path(Entry.Sampler, "val_batch_size", val_batch_size * max(1, dev.num_gpus))
			else:
				eval_batch_size = Opt.get_by_cfg_path(Entry.Sampler, "eval_batch_size")
				Opt.set_by_cfg_path(Entry.Sampler, "eval_batch_size", eval_batch_size * max(1, dev.num_gpus))

			# unset device_id, don't need if not distributed
			dev.device_id = None

			# call main
			runner()
