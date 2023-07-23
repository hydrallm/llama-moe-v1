from finetuner import finetuner
from utils import AttributeDict
import huggingface_hub
import argparse
import os
import yaml
import wandb


def load_config(filename):
    with open(filename, 'r') as file:
        return AttributeDict(yaml.safe_load(file))


def finetuner_runner(args, datasets):
    config = load_config(args.config)
    output_dir = config.output_dir

    if config.wandb_token:
        wandb_setup(config)
    
    if config.hf_token:
        hf_login(config)

    if os.path.exists(output_dir):
        existing_finetunes = os.listdir(output_dir)
        print(existing_finetunes)
    else:
        os.mkdir(output_dir)

    for dataset_name in datasets:
        if f"qlora_{dataset_name.split('/')[-1]}" not in existing_finetunes:
            config.dataset_name = dataset_name
            config.max_steps = 10000
            finetuner(config)

def wandb_setup(config):
    os.environ["WANDB_API_KEY"] = config.wandb_token
    os.environ["WANDB_WATCH"] = "gradients"
    os.environ["WANDB_LOG_MODEL"] = "true"
    wandb.login()

def hf_login(config):
    huggingface_hub.login(token=config.hf_token)

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
