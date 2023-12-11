from .logger import Logger
from .type_utils import Opt
from .entry_utils import Entry

from torch.nn.modules.utils import _pair, _triple, _quadruple, _single


class Util:

	# region convert a number to tuple

	@staticmethod
	def single(x):
		if isinstance(x, (tuple, list)):
			return x
		return _single(x)

	@staticmethod
	def pair(x):
		if isinstance(x, (tuple, list)):
			return x
		return _pair(x)

	@staticmethod
	def triple(x):
		if isinstance(x, (tuple, list)):
			return x
		return _triple(x)

	@staticmethod
	def quadruple(x):
		if isinstance(x, (tuple, list)):
			return x
		return _quadruple(x)

	# endregion

	# get actual model
	@staticmethod
	def get_module(model):
		from model.utils.ema_util import EMAWraper
		if isinstance(model, EMAWraper):
			return Util.get_module(model.ema_model)
		else:
			return model.module if hasattr(model, "module") else model

	@staticmethod
	def load_modules(call_file_path, package_path):
		import os
		import importlib
		modules_dir = os.path.dirname(call_file_path)
		for file in os.listdir(modules_dir):
			path = os.path.join(modules_dir, file)
			if (
				not file.startswith("_")
				and not file.startswith(".")
				and (file.endswith(".py") or os.path.isdir(path))
			):
				module_name = file[: file.find(".py")] if file.endswith(".py") else file
				importlib.import_module(f"{package_path}.{module_name}")

	@staticmethod
	def load_dir_modules(call_file_path, package_path):
		import os
		import glob
		import importlib

		outer_dir = os.path.dirname(call_file_path)

		inner_dirs = []

		# get inner directories
		for inner_dir in glob.glob("{}/*".format(outer_dir)):
			if os.path.isdir(inner_dir):
				dir_name = os.path.basename(inner_dir).strip()
				if not dir_name.startswith("_") and not dir_name.startswith("."):
					inner_dirs.append(dir_name)

		for inner_dir in inner_dirs:
			inner_dir_path = os.path.join(outer_dir, inner_dir)
			for file in os.listdir(inner_dir_path):
				path = os.path.join(inner_dir_path, file)
				if (
					not file.startswith("_")
					and not file.startswith(".")
					and (file.endswith(".py") or os.path.isdir(path))
				):
					file_name_without_extension = file[: file.find(".py")] if file.endswith(".py") else file
					importlib.import_module(
						package_path + inner_dir + "." + file_name_without_extension
					)

	@staticmethod
	# check python version
	def check_compatibility():
		import torch
		ver = torch.__version__.split(".")
		major_version = int(ver[0])
		minor_version = int(ver[1])

		if major_version < 1 and minor_version < 7:
			Logger.error(
				"Min pytorch version required is 1.7.0. Got: {}".format(".".join(ver))
			)

	from common import Common, Dev

	@staticmethod
	# set random seed, get the right device and gpu number
	def device_setup(cmm: Common, dev: Dev):
		import multiprocessing
		import torch
		import random
		import numpy as np

		# set random seed
		random_seed = cmm.seed

		random.seed(random_seed)
		np.random.seed(random_seed)
		torch.manual_seed(random_seed)
		torch.cuda.manual_seed(random_seed)
		torch.cuda.manual_seed_all(random_seed)

		# print random seed & torch version
		Logger.log(f"Random seeds are set to {random_seed}")
		Logger.log(f"Using PyTorch version {torch.__version__}")

		n_gpus = torch.cuda.device_count()  # number of gpus
		# if no gpu, set device to cpu
		if n_gpus == 0:
			Logger.warning("No GPUs available. Using CPU")
			device = torch.device("cpu")
			n_gpus = 0
		# set device to cuda
		else:
			Logger.log(f"Available GPUs: {n_gpus}")
			device = torch.device("cuda")
			if torch.backends.cudnn.is_available():
				import torch.backends.cudnn as cudnn

				torch.backends.cudnn.enabled = True
				cudnn.benchmark = False  # close benchmark
				cudnn.deterministic = True  # make sure of be deterministic
				Logger.log("CUDNN is enabled")

		dev.device = device
		dev.num_gpus = n_gpus
		dev.num_cpus = multiprocessing.cpu_count()

	from common import DDP

	@staticmethod
	# distribution operation init
	def distributed_init(ddp: DDP):
		import torch
		import torch.distributed as dist
		import socket

		if not ddp.dist_url:
			hostname = socket.gethostname()
			ddp.dist_url = f"tcp://{hostname}:{ddp.dist_port}"

		if dist.is_initialized():
			Logger.warning("DDP is already initialized and cannot be initialized twice!")
		else:
			Logger.info(f"distributed init (rank {ddp.rank}): {ddp.dist_url}", only_master_node=False)

			if not ddp.backend and dist.is_nccl_available():
				ddp.backend = "nccl"
				Logger.info(f"Using NCCL as distributed backend with version={torch.cuda.nccl.version()}")
			elif not ddp.backend:
				ddp.backend = "gloo"

			# distribution operation manager init
			dist.init_process_group(
				backend=ddp.backend,
				init_method=ddp.dist_url,
				world_size=ddp.world_size,
				rank=ddp.rank,
			)

			# perform a dummy all-reduce to initialize the NCCL communicator
			if torch.cuda.is_available():
				dist.all_reduce(torch.zeros(1).cuda())

		# update rank operation
		ddp.rank = torch.distributed.get_rank()

	@staticmethod
	def channel_last():
		return Opt.get_by_cfg_path(Entry.Common, subgroup_name="Common", key="channel_last")

	@staticmethod
	def is_master_node():
		return Opt.get_by_cfg_path(Entry.Common, subgroup_name="DDP", key="rank") == 0

	@staticmethod
	def is_distributed():
		return Opt.get_by_cfg_path(Entry.Common, subgroup_name="DDP", key="use_distributed")

	@staticmethod
	# is model coreml compatible
	def is_coreml_conversion():
		return Opt.get_by_cfg_path(Entry.Common, key="enable_coreml_compatible_module", subgroup_name="Common")

	@staticmethod
	def get_device():
		return Opt.get_by_cfg_path(Entry.Common, key="device", subgroup_name="Dev")

	@staticmethod
	def get_export_location():
		return Opt.get_by_cfg_path(Entry.Common, key="exp_loc", subgroup_name="Common")
