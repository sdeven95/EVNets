from .logger import Logger
import os


class File:
	@staticmethod
	# if the directory isn't existed, create it
	def create_directories(dir_path):
		if not os.path.isdir(dir_path):
			os.makedirs(dir_path)
			Logger.log("Directory created at: {}".format(dir_path))
		else:
			Logger.log("Directory exists at: {}".format(dir_path))
