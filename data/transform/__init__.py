from utils.type_utils import Cfg, Dsp, Par
from utils.entry_utils import Entry

from torchvision import transforms as T

INTERPOLATION_MODE_MAP = {
	"nearest": T.InterpolationMode.NEAREST,
	"bilinear": T.InterpolationMode.BILINEAR,
	"bicubic": T.InterpolationMode.BICUBIC,
	"cubic": T.InterpolationMode.BICUBIC,
	"box": T.InterpolationMode.BOX,
	"hamming": T.InterpolationMode.HAMMING,
	"lanczos": T.InterpolationMode.LANCZOS,
}

AUTOAUGMENT_POLICY_MAP = {
	"imagenet": T.AutoAugmentPolicy.IMAGENET,
	"cifar10": T.AutoAugmentPolicy.CIFAR10,
	"svhn": T.AutoAugmentPolicy.SVHN
}


def pre_handler(para_dict):
	if "interpolation" in para_dict:
		para_dict["interpolation"] = INTERPOLATION_MODE_MAP[para_dict["interpolation"]] if isinstance(para_dict["interpolation"], str) else para_dict["interpolation"]
	if "policy" in para_dict:
		para_dict["policy"] = AUTOAUGMENT_POLICY_MAP[para_dict["policy"]] if isinstance(para_dict["policy"], str) else para_dict["policy"]


class BaseTransform(Cfg, Dsp):

	_cfg_path_ = Entry.Transform

	def __call__(self, data):
		data["image"] = super().forward(data.pop("image"))
		return data

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if self.__class__.mro()[-2] != Dsp:
			Par.init_helper(self, kwargs, super(Dsp, self), pre_handler)


from .builtins import Normalize, RandomCropResize, AutoAugment, RandAugment, ColorJitter, RandomErasing
from .extensions import RandomHorizontalFlip, RandomRotate, Resize, \
	CustomCrop, CenterCrop, RandomResize, RandomShortSizeResize, \
	RandomGaussianBlur, ToTensor, Compose, RandomOrder, \
	SSDCropping, PhotometricDistort, BoxPercentCoords, InstanceProcessor, Identity, RandomCrop
from .torch import RandomMixup, RandomCutmix



