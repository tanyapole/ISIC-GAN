import argparse
import logging
import os
import sys

parser = argparse.ArgumentParser(description="Parse arguments to launch NN")

parser.add_argument("--train_root", help="path to base directory where train dataset storage", required=True)
parser.add_argument("--train_csv", help="path to csv train file", required=True)

parser.add_argument("--validate_root", help="path to base directory where test dataset storage", required=True)
parser.add_argument("--validate_csv", help="path to csv test file", required=True)

parser.add_argument("--epochs", type=int, help="epochs count", required=True)
parser.add_argument("--model_name", help="neural network base model name to use", default="inceptionv4")

parser.add_argument("--batch_size", type=int, help="batch size to process at once", default=32)
parser.add_argument("--num_workers", type=int, help="threads number to load dataset", default=8)
parser.add_argument("--learning_rate", type=float, help="learning step", required=True)

parser.add_argument("--result_dir", help="base path for all experiments", required=True)
parser.add_argument("--experiment_name", help="experiment name where all will be storages", required=True)

parser.add_argument("--start_from_begin", type=bool, help="indicate start train from first epoch", default=True)


def __parse_and_get(args=None):
    return parser.parse_args(args)


def __initialize_logging(path):
    logging.basicConfig(filename=os.path.join(path, "log.txt"), filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def initialize(args=None):
    params = __parse_and_get(args)
    exp_dir = os.path.join(params.result_dir, params.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    __initialize_logging(exp_dir)
    return params


if __name__ == "__main__":
    initialize([
        '--train_root', '/Users/nduginets/Desktop',
        '--train_csv', '/Users/nduginets/PycharmProjects/master-diploma/splits/validation.csv',
        "--validate_root", "/Users/nduginets/Desktop",
        "--validate_csv", "/Users/nduginets/PycharmProjects/master-diploma/splits/validation.csv",
        "--epochs", "10",
        "--learning_rate", "0.001",
        "--result_dir", "/Users/nduginets/Desktop",
        "--experiment_name", "tmp"
    ])
