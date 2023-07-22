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
    existing_finetunes = os.listdir(output_dir)
    print(existing_finetunes)

    for dataset_name in datasets:
        if f"{args.mode}_{dataset_name.split('/')[-1]}" not in existing_finetunes:
            config.dataset_name = dataset_name
            config.max_steps = 10000
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

    if args.finetune and args.mode:
        finetuner_runner(args, datasets)


if __name__ == "__main__":
    main()
