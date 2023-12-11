import time


class MetricLogger:

	log_file = None

	@staticmethod
	def get_time_stamp() -> str:
		return time.strftime('%Y-%m-%d %H:%M:%S')

	@classmethod
	def open_file(cls, file_path: str) -> None:
		cls.log_file = open(file_path, mode='a')

	@classmethod
	def close_file(cls) -> None:
		if cls.log_file is not None:
			cls.log_file.flush()
			cls.log_file.close()

	@classmethod
	# mode: train evl
	# model: general ema
	def print_header(cls) -> None:
		cls.log_file.write("\nlogtime\tepoch\tmode\tmodel\tloss\ttop1\ttop5\n")

	@classmethod
	def log_train(
			cls,
			epoch: int,
			loss: float,
	) -> None:
		cls.log_file.write(f'{cls.get_time_stamp()}\t{epoch}\ttrain\tgeneral\t{loss:.5f}\t0.0\t\0.0\n')
		cls.log_file.flush()

	@classmethod
	def log_val(
			cls,
			epoch: int,
			loss: float,
			top1: float,
			top5: float,
	) -> None:
		cls.log_file.write(f'{cls.get_time_stamp()}\t{epoch}\tval\tgeneral\t{loss:.5f}\t{top1:.5f}\t{top5:.5f}\n')
		cls.log_file.flush()

	@classmethod
	def log_ema_val(
			cls,
			epoch: int,
			loss: float,
			top1: float,
			top5: float,
	) -> None:
		cls.log_file.write(f'{cls.get_time_stamp()}\t{epoch}\tval\tema\t{loss:.5f}\t{top1:.5f}\t{top5:.5f}\n')
		cls.log_file.flush()
