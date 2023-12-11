import time
import sys
import os


def check_master_node(func):
	def inner(*args, **kwargs):
		from utils import Util
		if (
			"only_master_node" in kwargs
			and not kwargs["only_master_node"]
			or Util.is_master_node()
		):
			func(*args, **kwargs)

	return inner


class Logger:

	# definitions of colors
	text_colors = {
		"logs": "\033[34m",  # 033 is the escape code and 34 is the color code
		"info": "\033[32m",
		"warning": "\033[33m",
		"debug": "\033[93m",
		"error": "\033[31m",
		"bold": "\033[1m",
		"end_color": "\033[0m",
		"light_red": "\033[36m",
	}

	@staticmethod
	def get_curr_time_stamp():
		return time.strftime("%Y-%m-%d %H:%M:%S")

	@staticmethod
	# --------- print seperator or header or color text or single line -------------
	@check_master_node
	def single_dash_line(dashes=67):
		print("-" * dashes)

	@classmethod
	@check_master_node
	def double_dash_line(cls, dashes=75):
		print(cls.text_colors["error"] + "=" * dashes + cls.text_colors["end_color"])

	@classmethod
	def color_text(cls, in_text):
		return cls.text_colors["light_red"] + in_text + cls.text_colors["end_color"]

	@staticmethod
	def split_line(head_text):
		return "\n" + "-"*5 + head_text + "-"*5 + "\n"

	@classmethod
	@check_master_node
	def print_header(cls, header):
		cls.double_dash_line()
		print(
			cls.text_colors["info"]
			+ cls.text_colors["bold"]
			+ "=" * 50
			+ str(header)
			+ cls.text_colors["end_color"]
		)
		cls.double_dash_line()

	@classmethod
	@check_master_node
	def print_header_minor(cls, header):
		print(
			cls.text_colors["warning"]
			+ cls.text_colors["bold"]
			+ "=" * 25
			+ str(header)
			+ cls.text_colors["end_color"]
		)

	@classmethod
	# ----------- print info/warning/error/log/debug information
	@check_master_node
	def info(cls, message, print_line=False):
		time_stamp = cls.get_curr_time_stamp()
		info_str = (
			cls.text_colors["info"]
			+ cls.text_colors["bold"]
			+ "INFO    "
			+ cls.text_colors["end_color"]
		)
		print(f"{time_stamp} - {info_str} - {message}")
		if print_line:
			cls.double_dash_line(dashes=150)

	@classmethod
	@check_master_node
	def log(cls, message):
		time_stamp = cls.get_curr_time_stamp()
		log_str = (
			cls.text_colors["logs"]
			+ cls.text_colors["bold"]
			+ "LOGS    "
			+ cls.text_colors["end_color"]
		)
		print(f"{time_stamp} - {log_str} - {message}")

	@classmethod
	@check_master_node
	def debug(cls, message):
		time_stamp = cls.get_curr_time_stamp()
		debug_str = (
			cls.text_colors["debug"]
			+ cls.text_colors["bold"]
			+ "DEBUG   "
			+ cls.text_colors["end_color"]
		)
		print(f"{time_stamp} - {debug_str} - {message}")

	@classmethod
	def warning(cls, message):
		time_stamp = cls.get_curr_time_stamp()
		warn_str = (
			cls.text_colors["warning"]
			+ cls.text_colors["bold"]
			+ "WARNING"
			+ cls.text_colors["end_color"]
		)
		print(f"{time_stamp} - {warn_str} - {message}")

	@classmethod
	def error(cls, message):
		time_stamp = cls.get_curr_time_stamp()
		error_str = (
			cls.text_colors["error"]
			+ cls.text_colors["bold"]
			+ "ERROR  "
			+ cls.text_colors["end_color"]
		)
		print(f"{time_stamp} - {error_str} - {message}", flush=True)
		print(f"{time_stamp} - {error_str} - Exiting!!!", flush=True)
		exit(-1)

	@staticmethod
	def plain_text(message):
		print(message)

	# --------- out device select ---------

	__file_device = None

	@staticmethod
	def disable_printing():
		sys.stdout = open(os.devnull, "w")

	@classmethod
	def enable_printing(cls):
		sys.stdout = cls.__file_device if cls.__file_device else sys.__stdout__

	@classmethod
	def open_file_device(cls, path):
		cls.__file_device = open(path, "w")

	@classmethod
	def set_print_to_file(cls):
		cls.enable_printing()

	@classmethod
	def close_file_device(cls):
		if cls.__file_device:
			cls.__file_device.close()
		cls.__file_device = None
		cls.enable_printing()
