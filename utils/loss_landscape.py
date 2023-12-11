from .type_utils import Cfg
from .entry_utils import Entry

import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict


@Entry.register_entry(Entry.LossLandscape)
class LossLandscape(Cfg):
	__slots__ = ["n_points", "min_x", "max_x", "min_y", "max_y"]
	_keys_ = __slots__
	_disp_ = __slots__
	_defaults_ = [32, -1.0, 1.0, -1.0, 1.0]
	_types_ = [int, float, float, float, float]

	_cfg_path_ = Entry.LossLandscape

	# https://github.com/xxxnell/how-do-vits-work

	@staticmethod
	# set every item's value of dict to random number of standard normalization distribute
	def rand_basis(ws: Dict, device: Optional[str] = torch.device("cpu")):
		return {k: torch.randn(size=v.shape, device=device) for k, v in ws.items()}

	@staticmethod
	# bs[k] * weight => norm_bs[k], weight = norm(ws[k])/ norm(bs[k]), norm: compute matrix norm
	def normalize_filter(bs: Dict, ws: Dict):
		# make sure that every item value data type to float
		bs = {k: v.float() for k, v in bs.items()}
		ws = {k: v.float() for k, v in ws.items()}

		norm_bs = {}
		for k in bs:
			ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
			bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
			norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]

		return norm_bs

	@staticmethod
	# traverse dict items, if size of item value is less than 2, set every element of value to zero
	def ignore_bn(ws: Dict):
		ignored_ws = {}
		for k in ws:
			if len(ws[k].size()) < 2:
				ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
			else:
				ignored_ws[k] = ws[k]
		return ignored_ws

	@classmethod
	# create random dict with model state dict weighting ( refer to normalize_filter function)
	# !!!!!Note: in fact, for every weight parameter list( 2x2 matrix), create two offset matrix
	# for constructing an offset matrix
	# first, generate a matrix with rand normalization function
	# second, normalization in row direction, that is, for per row vector, first computer norm, then every element divided by norm
	# third, weight every element by row, that is, use same weight value to elements in the same row
	# !!!!!Note: the weights come from row vector norms of a parameter matrix of a model dict
	# !!!!!Note: for every one dimension parameter list(vector, generally biases), created offset vectors are zero vectors, that is, no offset
	def create_bases(
		cls,
		model: torch.nn.Module,
		device: Optional[str] = torch.device("cpu"),
		has_module: Optional[bool] = False,
		):
		# get model state dict
		weight_state_0 = (
			copy.deepcopy(model.module.state_dict())
			if has_module
			else copy.deepcopy(model.state_dict())
		)
		# get two random dict of standard normalization distribution which have same size to model state dict
		bases = [cls.rand_basis(weight_state_0, device) for _ in range(2)]  # Use two bases
		# use model state dict to weight each random dict
		bases = [cls.normalize_filter(bs, weight_state_0) for bs in bases]
		# filter dict item value of size < 2
		bases = [cls.ignore_bn(bs) for bs in bases]

		return bases

	@staticmethod
	# generate contour & 3D Protection & 3D Animation ====log plots===== for log transformation and save it
	def generate_plots(xx, yy, zz, model_name, results_loc):
		zz = np.log(zz)  # log transformation

		# plot log contour & save ( static)
		plt.figure(figsize=(10, 10))
		plt.contour(xx, yy, zz)
		plt.savefig(f"{results_loc}/{model_name}_log_contour.png", dpi=100)
		plt.close()

		# 3D plot & save (static)
		fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  # configure
		ax.set_axis_off()  # switch off axes
		ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)  # plot surface
		ax.set_xlim(-1, 1)   # set x axis limitation
		ax.set_ylim(-1, 1)   # set y axis limitation

		# save 3D figure
		plt.savefig(
			f"{results_loc}/{model_name}_log_surface.png",
			dpi=100,
			format="png",
			bbox_inches="tight",
		)
		plt.close()

		# plot & save 3D figure (animation gif)
		fig = plt.figure(figsize=(10, 10))
		ax = Axes3D(fig)  # set axes to 3D mode
		ax.set_axis_off()  # switch off axes display

		# init function --- plot surface, set axes limitations
		def init():
			ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			ax.set_xlim(-1, 1)
			ax.set_ylim(-1, 1)
			return fig

		# animate function ----- view init, set axes limitations
		def animate(i):
			ax.view_init(elev=(15 * (i // 15) + i % 15) + 0.0, azim=i)
			ax.set_xlim(-1, 1)
			ax.set_ylim(-1, 1)
			return fig

		# generate animation
		anim = animation.FuncAnimation(
			fig, animate, init_func=init, frames=100, interval=20, blit=True
		)

		# save animation figure
		anim.save(
			f"{results_loc}/{model_name}_log_surface.gif", fps=15, writer="imagemagick"
		)

	@classmethod
	# plot & save graphs
	def plot_save_graphs(
		cls,
		save_dir: str,
		model_name: str,
		grid_a: np.ndarray,
		grid_b: np.ndarray,
		loss_surface: np.ndarray,
		resolution: int,
		):
		# save numpy arrays
		np.save(f"{save_dir}/{model_name}_xx.npy", grid_a)
		np.save(f"{save_dir}/{model_name}_yy.npy", grid_b)
		np.save(f"{save_dir}/{model_name}_zz.npy", loss_surface)

		# plot grids & contour to save the figure
		plt.figure(figsize=(10, 10))
		plt.contour(grid_a, grid_b, loss_surface)
		plt.savefig(f"{save_dir}/{model_name}_contour_res_{resolution}.png", dpi=100)
		plt.close()

		# generate log plots
		cls.generate_plots(
			xx=grid_a,
			yy=grid_b,
			zz=loss_surface,
			model_name=model_name,
			results_loc=save_dir,
		)
