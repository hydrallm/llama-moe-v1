from finetuner import finetuner_qlora
from config import ScriptArguments
import argparse
from peft import LoraConfig
from transformers import (
    HfArgumentParser,
)


def main():
    # parser = argparse.ArgumentParser(description='Finetune or Load QLora Adapters.')
    # parser.add_argument('--finetune', action='store_true', help='Finetune QLora from datasets from scratch')
    # args = parser.parse_args()
    

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    finetuner_qlora(script_args)

if __name__ == "__main__":
    main()