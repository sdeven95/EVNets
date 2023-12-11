from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry


class VideoBaseTransform(Cfg, Dsp):
	_cfg_path_ = Entry.VideoTransform


# from .extensions import ToTensor, Compose, Resize, CenterCrop, \
# 	RandomCrop, RandomHorizontalFlip, RandomCropResize, RandomShortSizeResizeCrop
