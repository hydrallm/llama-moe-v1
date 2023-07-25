from finetuner import finetuner
from utils import AttributeDict
import argparse
import os
import subprocess
import yaml

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def load_config(config_file):
        with open(config_file, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream, Loader=yaml.FullLoader)
                config = Config(config_dict)
                return config
            except yaml.YAMLError as exc:
                print(exc)

def inference_runner(config_file, dataset, model, adapter):

    config = load_config(config_file)
    model_name = model.split('/')[1]
    dataset_name = dataset.split('/')[1]
    config.dataset_name = dataset
    config.model_name_or_path = model
    config.output_dir = model_name + "_" + dataset_name
    config.checkpoint_dir = adapter
    config.max_steps = 10000
    command = "python inference.py "
    for key, value in vars(config).items():
        command += f"--{key} {value} "
    subprocess.run(command, shell=True)


def finetuner_runner(config_file, dataset, model):
    config = load_config(config_file)
    model_name = model.split('/')[1]
    dataset_name = dataset.split('/')[1]
    config.dataset_name = dataset
    config.model_name_or_path = model
    config.output_dir = model_name + "_" + dataset_name
    config.max_steps = 10000
    command = "python finetuner.py "
    for key, value in vars(config).items():
        command += f"--{key} {value} "
    subprocess.run(command, shell=True)



def main():
    parser = argparse.ArgumentParser(description='MoE')
    parser.add_argument('--finetune', action='store_true',
                        help='Finetune? T/F')
    
    parser.add_argument('--inference', action='store_true',
                        help='Inference? T/F')

    parser.add_argument('--config', type=str, required=False,
                        help='Path to YAML config file')
    args = parser.parse_args()

    model = "meta-llama/Llama-2-7b-hf"
    dataset = "timdettmers/openassistant-guanaco"
    adapter = "HydraLM/camel_science_llama2-7b"
    if not args.config:
        config_file = "default_config.yaml"
    else:
        config_file = args.config

    if args.finetune:
        finetuner_runner(config_file, dataset, model)
    if args.inference:
        inference_runner(config_file, dataset, model, adapter)

if __name__ == "__main__":
    main()
