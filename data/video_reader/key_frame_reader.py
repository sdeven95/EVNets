
from . import BaseVideoReader

from typing import Optional, List
import torch
import av
from torch import Tensor

from utils.entry_utils import Entry


@Entry.register_entry(Entry.VideoReader)
class KeyFrameReader(BaseVideoReader):
	def __init__(self, is_training: Optional[bool] = False):
		super().__init__(is_training=is_training)

	def read_video(self, filename: str, frame_indices: Optional[List] = None) -> Optional[Tensor]:
		# for key frames, we do not know the indices of key frames.
		# so we can't use frame indices here
		try:
			# Check basic demo for Pyav usage: https://pyav.org/docs/stable/#basic-demo
			with av.open(filename) as container:
				stream = container.streams.video[0]
				stream.codec_context.skip_frame = "NONKEY"

				if self.fast_video_decoding:
					stream.thread_type = "AUTO"

				key_frames = []
				for frame in container.decode(video=0):
					frame = self.frame_to_tensor(frame)
					key_frames.append(frame)

				# [C, H, W] x N --> [N, C, H, W] or [C, N, H, W]
				key_frames = torch.stack(key_frames, dim=self.stack_frame_dim)
				return key_frames

		except av.AVError as ave_error:
			return None

	def process_video(
		self,
		vid_filename: str,
		n_frames_per_clip: Optional[int] = -1,
		clips_per_video: Optional[int] = 1,
		video_transform_fn: Optional = None,
		is_training: Optional[bool] = False,
	):
		# get frames
		# [N, C, H, W] or [C, N, H, W]
		torch_video = self.read_video(filename=vid_filename)

		# apply transformation and convert to clips
		if isinstance(torch_video, torch.Tensor):

			if video_transform_fn is not None:
				# Apply transformation
				torch_video = video_transform_fn({"image": torch_video})
				torch_video = torch_video["image"]

			# if it doesn't need to sample, convert to the clips directly
			# [N, C, H, W] or [C, N, H, W] => [T, n, C, H, W] or [T, C, n, H, W]
			# T: number of clips, n: number of frames per clip
			if n_frames_per_clip == -1:
				return self.convert_to_clips(video=torch_video, n_clips=clips_per_video)

			# select frames (i.e. get sampled frames)
			total_frames = torch_video.shape[self.stack_frame_dim]
			total_desired_frames = clips_per_video * n_frames_per_clip

			if is_training:
				frame_ids = self.random_sampling(
					desired_frames=total_desired_frames, total_frames=total_frames, n_clips=clips_per_video
				)
			else:
				frame_ids = self.uniform_sampling(
					desired_frames=total_desired_frames, total_frames=total_frames, n_clips=clips_per_video
				)

			# [N, C, H, W] or [C, N, H, W]
			torch_video = torch.index_select(
				torch_video, dim=self.stack_frame_dim, index=frame_ids
			)

			# if the number of sampled frames is less than the number of total desired frames
			# add black frames
			if torch_video.shape[self.stack_frame_dim] < total_desired_frames:
				# This is very unlikely but, for a safer side.
				delta = total_desired_frames - torch_video.shape[self.stack_frame_dim]
				clip_height, clip_width = torch_video.shape[-2:]
				if self.stack_frame_dim == 0:
					delta_frames = torch.zeros(size=(delta, 3, clip_height, clip_width))
				else:
					delta_frames = torch.zeros(size=(3, delta, clip_height, clip_width))
				torch_video = torch.cat(
					[torch_video, delta_frames], dim=self.stack_frame_dim
				)
			# if the number of sampled frames is more than the number of total desired frames
			# truncate
			elif torch_video.shape[self.stack_frame_dim] > total_desired_frames:
				if self.stack_frame_dim == 0:
					torch_video = torch_video[:total_desired_frames, ...]
				else:
					torch_video = torch_video[:, :total_desired_frames, ...]

			# keep the number of clips unchanged
			assert torch_video.shape[self.stack_frame_dim] % clips_per_video == 0

			# convert to clips
			# [N, C, H, W] or [C, N, H, W] => [T, n, C, H, W] or [T, C, n, H, W]
			# T: number of clips, n: number of frames per clip
			return self.convert_to_clips(video=torch_video, n_clips=clips_per_video)
		else:
			return None
