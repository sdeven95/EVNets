from model.utils.ema_util import EMAWraper
from optim import BaseOptim
from common import Common, Dev
from metrics import Metric
from .logger import Logger
from . import Util

import os
import glob
import math

import torch

from typing import Optional, Union


class CheckPoint:
	ETN = "pt"

	# region load & save model (with ema model) state dict

	@staticmethod
	# load model state from specified location
	# note file location set by common.model_state_loc and common.model_ema_state_loc
	def load_model_state(model, ema_cfg=None):

		dev = Dev()
		cmm = Common()

		# load content from file to specified device
		def load_state(path):
			if dev.device_id is None:
				model_state = torch.load(path, map_location=dev.device)
			else:
				model_state = torch.load(path, map_location="cuda:{}".format(dev.device_id))
			return model_state

		if cmm.model_state_loc is not None and os.path.isfile(cmm.model_state_loc):
			# load model dict
			CheckPoint.load_state_dict(model, load_state(cmm.model_state_loc))

			# load ema dict
			if ema_cfg and ema_cfg.enable and ema_cfg.ema_model and os.path.isfile(cmm.model_ema_state_loc):
				CheckPoint.load_state_dict(ema_cfg.ema_model, load_state(cmm.model_ema_state_loc))

		return model, ema_cfg

	@classmethod
	def save_model_state(cls, path, model, ema_cfg=None):
		model_state = cls.__get_model_state_dict(model)
		loc = os.path.join(path, "model_state_dict." + cls.ETN)
		torch.save(model_state, loc)

		if ema_cfg and ema_cfg.enable and ema_cfg.ema_model:
			ema_model_state = cls.__get_model_state_dict(ema_cfg.ema_model)
			ema_loc = os.path.join(path, "ema_model_state_dict" + cls.ETN)
			torch.save(ema_model_state, ema_loc)

	# endregion

	# region save & load checkpoint

	@classmethod
	# save check point, called on per epoch end
	# !!!!!!checkpoint.pt!!!!!! === main state dict which used to recovery run environment,
	#                   include: [epoch], [iteration], [model], [optimizer], [best_metric], [gradient_scalar]
	#                   could be used to resume model
	# -----------------------------------------------
	# state_dict_last.pt === last [model state dict]
	# state_dict_ema_last.pt  === last [ema model state dict]
	# -------------------------------------------------
	# state_dict_best.pt === best [model state dict]
	# state_dict_ema_best.pt === best [ema model state dict]
	# -------------------------------------------------
	# state_dict_epoch{n}.pt === all [model state dict]
	# state_dict_ema_epoch{n}.pt === all [ema model state dict]
	# ------------------------------------------------------------------
	# state_dict_avg.pt == [average state dict] for k-best model state dicts
	# state_dict_ema_avg.pt == average state dict for [k-best ema model state dicts]
	def save_checkpoint(
		cls,
		save_dir: str,
		iteration: int,
		epoch: int,

		model: torch.nn.Module,
		optimizer: Union[BaseOptim, torch.optim.Optimizer],
		gradient_scalar: torch.cuda.amp.GradScaler,
		is_best: bool,
		best_metric: float,

		ema_cfg: Optional[EMAWraper] = None,
		is_ema_best: Optional[bool] = False,
		ema_best_metric: Optional[float] = None,

		max_metric: Optional[bool] = False,
		k_best_state_dicts: Optional[int] = -1,
		save_all_checkpoints: Optional[bool] = False,
	) -> None:
		model_state = CheckPoint.__get_model_state_dict(model)  # get model state dict

		# create check point dict (for resume model)
		checkpoint = {
			"iteration": iteration,
			"epoch": epoch,
			"model_state_dict": model_state,
			"optim_state_dict": optimizer.state_dict(),
			"best_metric": best_metric,
			"gradient_scalar_state_dict": gradient_scalar.state_dict(),
		}
		state_dict_str = f"{save_dir}/state_dict"

		# save the best model state which later could be used for evaluation or fine-tuning
		if is_best:
			best_model_file_name = f"{state_dict_str}_best.{cls.ETN}"
			if os.path.isfile(best_model_file_name):
				os.remove(best_model_file_name)
			torch.save(model_state, best_model_file_name)  # save [best model] state dict
			Logger.log("Best model state dict with score {:.2f} saved at {}".format(best_metric, best_model_file_name))

			# if the best, update best check point file list
			if k_best_state_dicts > 1:
				CheckPoint.__avg_n_save_k_state_dicts(
					model_state=model_state,
					best_metric=best_metric,
					k_best_state_dicts=k_best_state_dicts,
					max_metric=max_metric,
					state_dict_str=state_dict_str
				)

		# save ema model state
		if ema_cfg and ema_cfg.enable and ema_cfg.ema_model:
			checkpoint["ema_state_dict"] = cls.__get_model_state_dict(ema_cfg)
			ema_file_name = f"{state_dict_str}_ema_last.{cls.ETN}"
			torch.save(checkpoint["ema_state_dict"], ema_file_name)  # save [last ema model] state dict
			if save_all_checkpoints:
				ema_file_name = f"{state_dict_str}_ema_epoch{epoch}.{cls.ETN}"
				torch.save(checkpoint["ema_state_dict"], ema_file_name)  # save [all ema model] state dict

			if is_ema_best:
				ema_best_file_name = f"{state_dict_str}_ema_best.{cls.ETN}"
				if os.path.isfile(ema_best_file_name):
					os.remove(ema_best_file_name)
				torch.save(checkpoint["ema_state_dict"], ema_best_file_name)  # save [best ema model] state dict
				Logger.log("Best EMA state dict with score {:.2f} saved at {}".format(ema_best_metric, ema_best_file_name))

				# if the best, update best check point file list
				if k_best_state_dicts > 1 and ema_best_metric is not None:
					cls.__avg_n_save_k_state_dicts(
						model_state=checkpoint["ema_state_dict"],
						best_metric=ema_best_metric,
						k_best_state_dicts=k_best_state_dicts,
						max_metric=max_metric,
						state_dict_str=f"{state_dict_str}_ema",
					)

		# save state dicts which could be used to recovery run environment
		ckpt_file_name = f"{save_dir}/checkpoint.{cls.ETN}"
		torch.save(checkpoint, ckpt_file_name)

		# save [last model] state dict
		state_dict_file_name = f"{state_dict_str}_last.{cls.ETN}"
		torch.save(model_state, state_dict_file_name)

		# save [all model] state dict
		if save_all_checkpoints:
			state_dict_file_name = f"{state_dict_str}_epoch{epoch}.{cls.ETN}"
			torch.save(model_state, state_dict_file_name)

	@classmethod
	# load check point
	# could be used to resume a training
	def load_checkpoint(
		cls,
		model: torch.nn.Module,
		optimizer: Union[BaseOptim, torch.optim.Optimizer],
		gradient_scalar: torch.cuda.amp.GradScaler,
		ema_cfg: Optional[EMAWraper] = None,
	):
		cmm = Common()
		dev = Dev()
		stat = Metric()

		start_epoch = start_iteration = 0
		best_metric = -math.inf if stat.checkpoint_metric_max else math.inf

		# if auto resume, get location by exp_loc
		resume_loc = cmm.resume_loc
		if resume_loc is None and cmm.auto_resume and cmm.exp_loc is not None:
			resume_loc = f"{cmm.exp_loc}/checkpoint.{cls.ETN}"

		if resume_loc is not None and os.path.isfile(resume_loc):
			# load check point dict from file
			if dev.device is not None:
				checkpoint = torch.load(resume_loc, map_location=dev.device)
			else:
				checkpoint = torch.load(resume_loc, map_location="cuda:{}".format(dev.device_id))

			# recover from dict
			start_epoch = checkpoint["epoch"] + 1
			start_iteration = checkpoint["iteration"] + 1
			best_metric = checkpoint["best_metric"]

			cls.load_state_dict(model, checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optim_state_dict"])
			gradient_scalar.load_state_dict(checkpoint["gradient_scalar_state_dict"])

			# load ema model state dict
			if ema_cfg and ema_cfg.enable and ema_cfg.ema_model and "ema_state_dict" in checkpoint:
				cls.load_state_dict(ema_cfg.ema_model, checkpoint["ema_state_dict"])

			Logger.log("Loaded checkpoint from {}".format(resume_loc))
			Logger.log("Resuming training for epoch {}".format(start_epoch))
		else:
			Logger.log("No checkpoint found at '{}'".format(resume_loc))
		return (
			model,
			optimizer,
			gradient_scalar,
			start_epoch,
			start_iteration,
			best_metric,
			ema_cfg,
		)

	# endregion

	# region  helper functions
	@classmethod
	# calculate average model dict across k best check point record & save it
	# and remove the earliest check point file
	# state dict(checkpoint) file name example: checkpoint_score_72.1234.pt
	# average state dict (checkpoint) file name: checkpoint_avg.pt
	def __avg_n_save_k_state_dicts(
		cls,
		model_state,  # model state dict
		best_metric,  # best metric now
		k_best_state_dicts,  # only keep k-best checkpoint files
		max_metric,  # for the metric, bigger for better?
		state_dict_str,  # file prefix
	):
		try:
			# save current best metric model checkpoint with the file name like state_dict_score_72.1234.pt
			state_dict_file_name = "{}_score_{:.4f}.{}".format(state_dict_str, best_metric, cls.ETN)
			torch.save(model_state, state_dict_file_name)

			# get all file names with score
			best_file_names = glob.glob(f"{state_dict_str}_score_*")
			# get all scores
			current_best_scores = [
				float(f.split("_score_")[-1].replace(f".{cls.ETN}", ""))
				for f in best_file_names
			]

			best_scores_keep = []  # list include files needed to keep
			# Crop Records, that is, remove the file with the smallest score
			if len(current_best_scores) > k_best_state_dicts:
				current_best_scores = sorted(current_best_scores)
				# metric should be reverse, smaller for better, for example loss
				if not max_metric:
					current_best_scores = current_best_scores[::-1]
				# get k best scores from tail
				best_scores_keep = current_best_scores[-k_best_state_dicts:]
				# remove others
				for k in current_best_scores:
					if k in best_scores_keep:
						continue
					rm_state_dict = "{}_score_{:.4f}.{}".format(state_dict_str, k, cls.ETN)
					os.remove(rm_state_dict)
					Logger.log("Deleting state dict: {}".format(rm_state_dict))

			# if we have k best check points
			if len(best_scores_keep) > 1:
				# get file names
				avg_file_names = [
					"{}_score_{:.4f}.{}".format(state_dict_str, k, cls.ETN)
					for k in best_scores_keep
				]
				Logger.log(
					"Averaging state dicts: {}".format(
						[f.split("/")[-1] for f in avg_file_names]
					)
				)
				# compute & save the average model
				avg_model_state = cls.__average_state_dict(state_dict_loc_list=avg_file_names)  # get average values
				state_dict_file_name = f"{state_dict_str}_avg.{cls.ETN}"
				if avg_model_state:
					torch.save(avg_model_state, state_dict_file_name)
					Logger.log("Averaged checkpoint saved at: {}".format(state_dict_file_name))
		except Exception as e:
			Logger.log("Error in k-best-checkpoint")
			print(e)

	@staticmethod
	# loop on k check point files, average values in all k state dicts
	def __average_state_dict(state_dict_loc_list: list):
		avg_state_dict = dict()
		key_count = dict()
		key_dtype = dict()

		for c in state_dict_loc_list:
			if not os.path.isfile(c):
				pass
			ckpt_state_dict = torch.load(c, map_location="cpu")

			for k, v in ckpt_state_dict.items():
				if k not in avg_state_dict:
					key_dtype[k] = v.dtype
					avg_state_dict[k] = v.clone().to(dtype=torch.float64)  # compute in float64 type
					key_count[k] = 1
				else:
					avg_state_dict[k] += v.to(dtype=torch.float64)
					key_count[k] += 1

		for k, v in avg_state_dict.items():
			avg_state_dict[k] = v.div(key_count[k]).to(dtype=key_dtype[k])
		return avg_state_dict

	@staticmethod
	# get state dictionary
	def __get_model_state_dict(model):
		return Util.get_module(model).state_dict()

	@staticmethod
	# load state dictionary
	def load_state_dict(model, state_dict):
		Util.get_module(model).load_state_dict(state_dict)

	@staticmethod
	def load_model_from_state_dict_file(model: torch.nn.Module, path):
		if not os.path.isfile(path):
			raise FileExistsError(f"Can not find state dict file: {path}")
		model_state_dict = torch.load(path, map_location="cpu")
		model.load_state_dict(model_state_dict)

	@staticmethod
	# copy state dict between model
	def copy_weights(
		model_src: Union[torch.nn.Module, EMAWraper], model_tar: torch.nn.Module
		):
		with torch.no_grad():
			model_state = CheckPoint.__get_model_state_dict(model=model_src)
			CheckPoint.load_state_dict(model=model_tar, state_dict=model_state)

	# endregion
