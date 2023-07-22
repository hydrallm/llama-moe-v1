# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
import pickle
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from trl import SFTTrainer
from accelerate import init_empty_weights
from pathlib import Path

FORMAT_DICT = {
    'alpaca': lambda x: ''.join([f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {ex['instruction']}

    ### Input:
    {ex['input']}

    ### Output: 
    {ex['output']}
    ''' for ex in x]),

    'default': lambda x: '\n'.join([ex['text'] for ex in x]),
}


def pop_peft(model):
    """
        remove peft from a model, return lora state dict
    """
    with init_empty_weights():
        lora_state, model_state = {}, {}
        for k, v in model.state_dict().items():
            if 'weight_format' in k or 'SCB' in k:
                pass
            elif 'lora' in k:
                lora_state[k] = v
            else:
                print(
                    k,
                )
                model_state[k.split('base_model.model.', 1)[1]] = v
        del model

    return lora_state


def finetuner(script_args):
    """
    Finetune a QLoRA adapter and save it as a .pickle file
    """
    def create_and_prepare_model(args):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

        # Load the entire model on the GPU 0
        # switch to `device_map = "auto"` for multi-GPU
        device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map=device_map,
            use_auth_token=True,
        )

        # check: https://github.com/huggingface/transformers/pull/24906
        model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        return model, peft_config, tokenizer

    training_arguments = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        max_grad_norm=script_args.max_grad_norm,
        max_steps=script_args.max_steps,
        warmup_ratio=script_args.warmup_ratio,
        group_by_length=script_args.group_by_length,
        lr_scheduler_type=script_args.lr_scheduler_type,
    )

    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False
    dataset = load_dataset(script_args.dataset_name, split="train")
    formatting_func = FORMAT_DICT[script_args.data_format]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=script_args.packing,
        formatting_func=formatting_func,
    )

    trainer.train()
    lora_adapter = pop_peft(model)

    output_name = os.path.join(
        trainer.args.output_dir,
        f"lora_{script_args.model_name.split('/')[-1]}",
        f"lora_{script_args.dataset_name.split('/')[-1]}"
    )
    Path(output_name).mkdir(parents=True, exist_ok=True)

    with open(output_name + '/checkpoint.pt', 'w+b') as handle:
        pickle.dump(lora_adapter, handle)
