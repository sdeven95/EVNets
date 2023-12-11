# from model.classification.cvnets.resnet import ResNet
from model.classification.mobile_aggregate import MobileAggregate
# from model.classification.mobile_global_shuffle import MobileShuffleV2

from run.check_model import CheckModel
from run.bench_model import BenchModel
from run.bench_model_simple import BenchModelSimple
from run.converse_model import ConverseModel
from run.train_model import TrainModel
from run.evaluate_model import EvaluateModel


def run():
	run_type = "train_model"
	# check_model, bench_model_simple, bench_model, converse_model, train_model, evaluate_model, print_argument_template

	model_cls = MobileAggregate  # MobileAggregate  # set here

	if run_type == "print_argument_template":
		from utils.opt_setup_utils import OptSetup
		OptSetup.print_arguments_template(model_cls)
		exit(0)

	if run_type == "check_model":
		CheckModel.run(model_cls)
		exit(0)

	if run_type == "bench_model_simple":
		BenchModelSimple.run(model_cls, batch_size=16, iterations=30, warm_up=10)
		exit(0)

	if run_type == "bench_model":
		BenchModel.run(model_cls)
		exit(0)

	if run_type == "converse_model":
		ConverseModel.run(model_cls)
		exit(0)

	if run_type == "train_model":
		TrainModel.run(model_cls)
		exit(0)

	if run_type == "evaluate_model":
		EvaluateModel.run(model_cls)
		exit(0)


if __name__ == "__main__":
	run()
