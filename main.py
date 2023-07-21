from finetuner import finetuner_qlora
from config import ScriptArguments
import argparse
import os


def finetuner_runner(args, datasets):
    output_dir = ScriptArguments.output_dir
    existing_finetunes = os.listdir(output_dir)
    print(existing_finetunes)

    for dataset_name in datasets:
        if f"{args.mode}_{dataset_name.split('/')[-1]}" not in existing_finetunes:
            script_args = ScriptArguments(
                dataset_name=dataset_name,
                max_steps=10000,
            )
            if args.finetune and args.mode == 'qlora':
                finetuner_qlora(script_args)

def main():
    parser = argparse.ArgumentParser(description='MoE')
    parser.add_argument('--finetune', action='store_true', help='Finetune? T/F')
    parser.add_argument('--mode', type=str, required=True, help='Modes: qlora')
    args = parser.parse_args()
    

    #DATASETS
    datasets = ["timdettmers/openassistant-guanaco"] 

    if args.finetune and args.mode:
        finetuner_runner(args, datasets)
    

if __name__ == "__main__":
    main()
