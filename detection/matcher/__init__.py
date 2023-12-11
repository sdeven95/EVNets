from utils.type_utils import Cfg, Dsp
from utils.entry_utils import Entry


class BaseMatcher(Cfg, Dsp):
	_cfg_path_ = Entry.Matcher


from .ssd import SSDMatcher
