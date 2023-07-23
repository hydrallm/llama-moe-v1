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
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    LlamaTokenizer,
)
from peft.tuners.lora import LoraLayer
from utils import smart_tokenizer_and_embedding_resize

from trl import SFTTrainer
from accelerate import init_empty_weights

DEFAULT_PAD_TOKEN = "[PAD]"

FORMAT_DICT = {
    "alpaca": lambda x: f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{x['instruction']}\n### Input:\n{x['input']}\n### Output:\n{x['output']}",
    "default": lambda x: x["text"],
    "camel": lambda x: f"### Instruction:\n{x['message_1']}\n### Output:\n{x['message_2']}",
}


def finetuner(script_args):
    """
    Finetune a QLoRA adapter and save it as a .pickle file
    """

    def create_and_prepare_model(args):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            load_in_8bit=args.use_8bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

        # Load the entire model on the GPU 0
        # switch to `device_map = "auto"` for multi-GPU
        device_map = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            use_auth_token=True,
        )

        # check: https://github.com/huggingface/transformers/pull/24906
        model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            target_modules=script_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name, trust_remote_code=True
        )
        tokenizer.pad_token_id = model.config.pad_token_id

        # for name, module in model.named_modules():
        #     if isinstance(module, LoraLayer):
        #         if script_args.bf16:
        #             module = module.to(torch.bfloat16)
        #     if "norm" in name:
        #         module = module.to(torch.float32)
        #     if "lm_head" in name or "embed_tokens" in name:
        #         if hasattr(module, "weight"):
        #             if script_args.bf16 and module.weight.dtype == torch.float32:
        #                 module = module.to(torch.bfloat16)

        if tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

        if "llama" in script_args.model_name or isinstance(tokenizer, LlamaTokenizer):
            # LLaMA tokenizer may not have correct special tokens set.
            # Check and add them if missing to prevent them from being parsed into different tokens.
            # Note that these are present in the vocabulary.
            # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
            print("Adding special tokens.")
            tokenizer.add_special_tokens(
                {
                    "eos_token": tokenizer.convert_ids_to_tokens(
                        model.config.eos_token_id
                    ),
                    "bos_token": tokenizer.convert_ids_to_tokens(
                        model.config.bos_token_id
                    ),
                    "unk_token": tokenizer.convert_ids_to_tokens(
                        model.config.pad_token_id
                        if model.config.pad_token_id != -1
                        else tokenizer.pad_token_id
                    ),
                }
            )

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
        num_train_epochs=script_args.num_train_epochs,
        warmup_ratio=script_args.warmup_ratio,
        group_by_length=script_args.group_by_length,
        lr_scheduler_type=script_args.lr_scheduler_type,
        report_to=script_args.report_to,
        save_strategy=script_args.save_strategy,
    )

    if "science" in script_args.dataset_name:
        dataset_physics = load_dataset("camel-ai/physics", split="train")
        dataset_chemistry = load_dataset("camel-ai/chemistry", split="train")
        dataset_biology = load_dataset("camel-ai/biology", split="train")
        dataset = concatenate_datasets(
            [dataset_physics, dataset_chemistry, dataset_biology]
        )
    else:
        dataset = load_dataset(script_args.dataset_name, split="train")

    print(f"Dataset Length: {len(dataset)}")
    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False

    formatting_func = lambda x: {"text": FORMAT_DICT[script_args.data_format](x)}
    dataset = dataset.map(formatting_func)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=script_args.packing,
        # formatting_func=formatting_func,
    )

    trainer.train()

    # model.save_pretrained("adapters")

    # if script_args.push_to_hub is not None:
    #     model.push_to_hub(script_args.push_to_hub)
