from finetuner import finetuner_qlora
from config import ScriptArguments
import argparse
from peft import LoraConfig
from transformers import (
    HfArgumentParser,
)


def main():
    parser = argparse.ArgumentParser(description='Finetune or Load QLora Adapters.')
    parser.add_argument('--finetune', action='store_true', help='Finetune QLora from datasets from scratch')
    args = parser.parse_args()
    

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[])[0]
    script_args = ScriptArguments(
        local_rank=-1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        weight_decay=0.001,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_r=64,
        max_seq_length=512,
        model_name="4bit/Llama-2-7b-chat-hf",
        dataset_name="timdettmers/openassistant-guanaco",
        use_4bit=True,
        use_nested_quant=False,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        num_train_epochs=1,
        fp16=False,
        bf16=False,
        packing=False,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="constant",
        max_steps=10000,
        warmup_ratio=0.03,
        group_by_length=True,
        save_steps=10,
        logging_steps=10,
        merge_and_push=False,
        output_dir="./results",
    )
    if args.finetune:
         finetuner_qlora(script_args)

if __name__ == "__main__":
    main()