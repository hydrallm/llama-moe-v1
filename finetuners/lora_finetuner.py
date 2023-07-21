import transformers
import argparse
import datasets
import torch
import numpy as np
import pandas as pd
import os
import gc
import pickle
import random

from datasets import load_from_disk
from config import ScriptArguments

from transformers import (
    AutoModel,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaConfig,
    pipeline,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoConfig,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    )

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm

from accelerate import init_empty_weights
from dataclasses import dataclass, field
from multiprocessing import Process

def pop_peft(model):
    """
        remove peft from a model, return lora state dict
    """
    print('removing peft from model...')
    with init_empty_weights():
        lora_state, model_state = {}, {}
        for k, v in model.state_dict().items():
            if 'weight_format' in k or 'SCB' in k:
                pass
            elif 'lora' in k:
                lora_state[k] = v
            else:
                model_state[k.split('base_model.model.', 1)[1]] = v
        del model
    return lora_state


def run_lora_worker(config: ScriptArguments):
    """
    train a single LoRA adapter and save it in pickle format
    """

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 256
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt_instruct(data_point):
        full_prompt = (lambda x: '### Instruction\n' + \
                        x['instruction'] + x['input'] \
                        + '\n###Response\n' + x['output'])\
                        (data_point)

        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point['text']
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    # load dataset
    dataset = load_dataset(config.dataset_name, split='train')
    dataset = dataset.shuffle()
    if config.instruct_format:
        dataset = dataset.map(generate_and_tokenize_prompt_instruct, num_proc=8)
    else:
        dataset = dataset.map(generate_and_tokenize_prompt, num_proc=8)

    # load adapter
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = config.lora_target_modules
    )

    # load peft model with int8 training 
    model = LlamaForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map='auto'
    )
    model = get_peft_model(model, lora_config)
    model = prepare_model_for_int8_training(model)


    # train
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=int(config.warmup_ratio * config.max_steps),
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            optim=config.optim,
            save_strategy="steps",
            output_dir=config.output_dir,
            fp16=config.fp16,
            bf16=config.bf16
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ))

    trainer.train()
    lora_adapter = pop_peft(model)

    output_name = f'run_{config.model_name}_{config.dataset_name}'

    with open(output_name, 'wb') as handle:
        pickle.dump(lora_adapter, handle)
    

