import av
from PIL import Image
import numpy as np
import torch
import random

from utils.type_utils import Cfg, Dsp, Opt
from utils.entry_utils import Entry

from typing import Optional, Any, List

import data.transform.builtins as T


class BaseVideoReader(Cfg, Dsp):
	__slots__ = [
		"fast_video_decoding", "frame_stack_format",
		"stack_frame_dim", "frame_transforms", "random_erase_transform", "transforms_str", "num_frames_cache"
	]
	_cfg_path_ = Entry.VideoReader
	_keys_ = __slots__[0:2]
	_disp_ = __slots__
	_defaults_ = [False, "sequence_first"]  # frame_stack_format : ["sequence_first", "channel_first"] NCHW or CNHW
	_types_ = [bool, str]

	def __init__(self, is_training: Optional[bool] = False):
		super().__init__()

		self.stack_frame_dim = 1 if self.frame_stack_format == "channel_first" else 0

		self.frame_transforms = self._frame_transform() if is_training else None
		self.random_erase_transform = self._random_erase_transform() if is_training else None

		self.transforms_str += repr(self.frame_transforms) if self.frame_transforms else ""
		self.transforms_str += repr(self.random_erase_transform) if self.random_erase_transform else ""

		self.num_frames_cache = dict()

	@staticmethod
	def _frame_transform():
		auto_augment, rand_augment = Opt.get_argument_values(
			pth_tpl=Entry.Transform + ".{}",
			keys=("AutoAugment.enable", "RandAugment.enable"),
			defaults=(False, False)
		)
		assert not (auto_augment and rand_augment), \
			"AutoAugment and RandAugment are mutually exclusive. Use either of them, but not both"

		if auto_augment:
			return T.AutoAugment()
		elif rand_augment:
			return T.RandAugment()
		return None

	@staticmethod
	def _random_erase_transform():
		random_erase = Opt.get(f"{Entry.Transform}.RandomErasing.enable")
		if random_erase:
			return T.RandomErasing()
		return None

	def check_video(self, filename: str) -> bool:
		try:
			# Adapted from basic demo: https://pyav.org/docs/stable/#basic-demo
			with av.open(filename) as container:
				# Decode the first video channel.
				for frame in container.decode(video=0):
					frame_idx = frame.index
					break
				return True
		except Exception as e:
			return False

	def read_video(self, filename: str) -> Any:
		raise NotImplementedError

	def num_frames(self, filename: str) -> int:
		if filename in self.num_frames_cache:
			return self.num_frames_cache[filename]
		else:
			total_frames = 0
			with av.open(filename) as container:
				total_frames = container.streams.video[0].frames
			self.num_frames_cache[filename] = total_frames
			return total_frames

	# get transformed frame
	def frame_to_tensor(self, frame):
		frame_np = frame.to_ndarray(format="rgb24")
		if self.frame_transforms is not None:
			#
			frame_pil = Image.fromarray(frame_np)
			frame_pil = self.frame_transforms({"image": frame_pil})["image"]
			frame_np = np.array(frame_pil)

		frame_np = frame_np.transpose(2, 0, 1)
		frame_np = np.ascontiguousarray(frame_np)
		# [C, H, W]
		frame_torch = torch.from_numpy(frame_np)

		# normalize the frame between 0 and 1
		frame_torch = frame_torch.div(255.0)

		# apply random erase transform
		if self.random_erase_transform is not None:
			frame_torch = self.random_erase_transform({"image": frame_torch})["image"]

		return frame_torch

	@staticmethod
	def random_sampling(
		desired_frames: int, total_frames: int, n_clips: int
	) -> List:
		# divide the video into K clips
		try:
			interval = (
				desired_frames if total_frames >= desired_frames * (n_clips + 1) else 1
			)
			# The range of start Id is between [0, total_frames - n_desired_frames]
			temp = max(0, min(total_frames - desired_frames, total_frames))
			start_ids = sorted(
				random.sample(population=range(0, temp, interval), k=n_clips)
			)
			# 30 frames and 120 frames in 1s and 4s videos @ 30 FPS, respectively
			# The end_id is randomly selected between start_id + 30 and start_id + 120
			end_ids = [
				min(
					max(s_id + random.randint(30, 120), s_id + desired_frames),
					total_frames - 1,
				)
				for s_id in start_ids
			]
		except:
			# fall back to uniform
			video_clip_ids = np.linspace(
				0, total_frames - 1, n_clips + 1, dtype=int
			).tolist()

			start_ids = video_clip_ids[:-1]
			end_ids = video_clip_ids[1:]

		frame_ids = []
		for start_idx, end_idx in zip(start_ids, end_ids):
			try:
				clip_frame_ids = sorted(
					random.sample(
						population=range(start_idx, end_idx), k=desired_frames
					)
				)
			except:
				# sample with repetition
				clip_frame_ids = np.linspace(
					start=start_idx, stop=end_idx - 1, num=desired_frames, dtype=int
				).tolist()
			frame_ids.extend(clip_frame_ids)
		return frame_ids

	# divide the video to n clips, select desired number of frames in every clip
	@staticmethod
	def uniform_sampling(
		desired_frames: int, total_frames: int, n_clips: int
	):
		video_clip_ids = np.linspace(
			0, total_frames - 1, n_clips + 1, dtype=int
		).tolist()
		start_ids = video_clip_ids[:-1]
		end_ids = video_clip_ids[1:]

		frame_ids = []
		for start_idx, end_idx in zip(start_ids, end_ids):
			clip_frame_ids = np.linspace(
				start=start_idx, stop=end_idx - 1, num=desired_frames, dtype=int
			).tolist()
			frame_ids.extend(clip_frame_ids)
		return frame_ids

	def convert_to_clips(self, video: torch.Tensor, n_clips: int):
		# video is [N, C, H, W] or [C, N, H, W]
		video_clips = torch.chunk(video, chunks=n_clips, dim=self.stack_frame_dim)
		video_clips = torch.stack(video_clips, dim=0)
		# video_clips is [T, n, C, H, W] or [T, C, n, H, W]
		return video_clips

	def process_video(
		self,
		vid_filename: str,
		n_frames_per_clip: Optional[int] = -1,
		clips_per_video: Optional[int] = 1,
		video_transform_fn: Optional = None,
		is_training: Optional[bool] = False,
	):
		raise NotImplementedError

	def dummy_video(
		self, clips_per_video: int, n_frames_to_sample: int, height: int, width: int
	):

		# [K, C, N, H, W] or # [K, N, C, H, W]
		# K --> number of clips, C --> Image channels, N --> Number of frames per clip, H --> Height, W --> Width
		tensor_size = (
			(clips_per_video, 3, n_frames_to_sample, height, width)
			if self.frame_stack_format == "channel_first"
			else (clips_per_video, n_frames_to_sample, 3, height, width)
		)

		input_video = torch.zeros(
			size=tensor_size, dtype=torch.float32, device=torch.device("cpu")
		)
		return input_video


# from .default_video_reader import DefaultVideoReader
# from .key_frame_reader import KeyFrameReader
