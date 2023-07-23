from finetuner import finetuner
from utils import AttributeDict
import argparse
import os
import yaml


def load_config(filename):
    with open(filename, "r") as file:
        return AttributeDict(yaml.safe_load(file))


def finetuner_runner(args):
    config = load_config(args.config)
    output_dir = config.output_dir
    config.push_to_hub = args.push_to_hub

    if os.path.exists(output_dir):
        existing_finetunes = os.listdir(output_dir)
        print(existing_finetunes)
    else:
        existing_finetunes = []
        os.makedirs(output_dir)

    if f"qlora_{config.dataset_name.split('/')[-1]}" not in existing_finetunes:
        finetuner(config)


def main():
    parser = argparse.ArgumentParser(description="MoE")
    parser.add_argument("--finetune", action="store_true", help="Finetune? T/F")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="push to huggingface hub.",
    )
    args = parser.parse_args()

    if args.finetune:
        finetuner_runner(args)


if __name__ == "__main__":
    main()
