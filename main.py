from finetuner import finetuner
from utils import AttributeDict
import argparse
import os
import yaml


def load_config(filename):
    with open(filename, 'r') as file:
        return AttributeDict(yaml.safe_load(file))


def finetuner_runner(args, datasets):
    config = load_config(args.config)
    output_dir = config.output_dir

    if os.path.exists(output_dir):
        existing_finetunes = os.listdir(output_dir)
        print(existing_finetunes)
    else:
        os.mkdir(output_dir)

    finetuner(config)


def main():
    parser = argparse.ArgumentParser(description='MoE')
    parser.add_argument('--finetune', action='store_true',
                        help='Finetune? T/F')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    # DATASETS
    datasets = ["timdettmers/openassistant-guanaco"]

    if args.finetune:
        finetuner_runner(args, datasets)


if __name__ == "__main__":
    main()
