from finetuner import finetuner_qlora
from config import ScriptArguments
import argparse

def main():
    parser = argparse.ArgumentParser(description='Finetune or Load QLora Adapters.')
    parser.add_argument('--finetune', action='store_true', help='Finetune QLora from datasets')
    args = parser.parse_args()

    if args.finetune:
        finetuner_qlora(ScriptArguments)



if __name__ == "__main__":
    main()