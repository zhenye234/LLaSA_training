import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
)
from dataclasses import dataclass, field
from typing import Optional
import sys
import transformers
import wandb
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np
import random
from datasets import load_dataset
from functools import partial
 
 
@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

 
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the memmap data."})

 
@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(
        default=None, metadata={"help": "The integration to report the results and logs to."}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the run for logging."}
    )
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The learning rate scheduler to use."})
  
class TTSDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):
 
        memmap_path = os.path.join(data_path, f'{split}_input_ids.memmap')
        shape_path = os.path.join(data_path, f'{split}_input_ids_shape.npy')

        self.input_ids = np.memmap(memmap_path, dtype='int32', mode='r', shape=tuple(np.load(shape_path)))
        self.length = self.input_ids.shape[0]
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer

   
        self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
        self.text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
        self.text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
        self.text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
        self.speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
        self.speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
 
        self.max_length = 2048
        self.ignore_index = -100  

    def __len__(self):
        return self.length

    def replace_tagged_token(self, token_list, target_token, new_sequence):
        idx = token_list.index(target_token)
        return token_list[:idx] + list(new_sequence) + token_list[idx+1:]

    def pad_sequence(self, sequence, max_length, value=0):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = torch.full((max_length - len(sequence),), value, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)

 
        labels = torch.full_like(input_ids, self.ignore_index)

 
        speech_gen_positions = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0]
        text_gen_positions = (input_ids == self.text_generation_start_id).nonzero(as_tuple=True)[0]
 
        speech_gen_idx = speech_gen_positions[0].item()
        try:
            speech_gen_end_idx = (input_ids == self.speech_generation_end_id).nonzero(as_tuple=True)[0].item()
        except Exception as e:
            print(f"maybe Error in speech_gen_end_idx: {e}")
            # speech_gen_end_idx = len(input_ids) - 1
            speech_gen_end_idx = 2048
 
        text_sequence = input_ids[:speech_gen_idx]
        speech_sequence = input_ids[speech_gen_idx : speech_gen_end_idx + 1]
 
        chat = [
            {"role": "user", "content": "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True)

    
        ids = self.replace_tagged_token(ids, self.text_understanding_start_id, text_sequence)
        ids = self.replace_tagged_token(ids, self.speech_generation_start_id, speech_sequence)



        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        try:
 
            speech_gen_idx_in_input = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0].item()
    
            
            labels[speech_gen_idx_in_input:] = input_ids[speech_gen_idx_in_input:]
        except Exception as e:
            print(f"maybe Error in speech_gen_idx_in_input: {e}")
            # speech_gen_idx_in_input = len(input_ids) - 1
 
            labels  = input_ids 

 
        attention_mask = (input_ids != self.pad_token_id).long()
 
        labels[input_ids == self.pad_token_id] = self.ignore_index

 
        
        input_ids = self.pad_sequence(input_ids, self.max_length, value=self.pad_token_id)
        attention_mask = self.pad_sequence(attention_mask, self.max_length, value=0)
        labels = self.pad_sequence(labels, self.max_length, value=self.ignore_index)

        return {
            'input_ids': list(input_ids),
            'labels': list(labels),
            'attention_mask': list(attention_mask)
        }
 

def main():
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        # Load arguments from the specified JSON file
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Attempt to load arguments from the default 'config.json' file
        default_config_file = 'config.json'
        if os.path.exists(default_config_file):
            (
                model_args,
                data_args,
                training_args,
            ) = parser.parse_json_file(json_file=os.path.abspath(default_config_file))
        else:
            # If 'config.json' does not exist, parse arguments from the command line
            (
                model_args,
                data_args,
                training_args,
            ) = parser.parse_args_into_dataclasses()

    is_main_process = training_args.local_rank in [-1, 0]
 
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(
            project="llm_tts",  
            config=training_args.to_sanitized_dict(),
            name=training_args.run_name
        )

 
 
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        # Find all checkpoint directories in the output directory
        checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            # Get the most recent checkpoint based on modification time
            last_checkpoint = max(checkpoints, key=os.path.getmtime)

    if last_checkpoint is not None:
        print(f"Loading model and tokenizer from checkpoint {last_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch")
        # Load tokenizer from the initial model path
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
        tokenizer.pad_token = tokenizer.eos_token  # For LLaMa
        original_vocab_size = len(tokenizer)
        print(f"Original tokenizer vocabulary size: {len(tokenizer)}")

        Start_End_tokens = [
            '<|TEXT_GENERATION_START|>',
            '<|TEXT_GENERATION_END|>',
            '<|TEXT_UNDERSTANDING_START|>',
            '<|TEXT_UNDERSTANDING_END|>',
            '<|SPEECH_GENERATION_START|>',
            '<|SPEECH_GENERATION_END|>',
            '<|SPEECH_UNDERSTANDING_START|>',
            '<|SPEECH_UNDERSTANDING_END|>'
        ]

 
        new_speech_tokens = [f'<|s_{i}|>' for i in range(65536)]   
        all_new_tokens = Start_End_tokens + new_speech_tokens
        num_added_tokens = tokenizer.add_tokens(all_new_tokens)
        print(f"Added {num_added_tokens} speech tokens to the tokenizer.")

 
        tokenizer.save_pretrained(training_args.output_dir)
 
        model = AutoModelForCausalLM.from_pretrained(
            model_args.llm_model_name_or_path,
            torch_dtype='auto',
            cache_dir=model_args.cache_dir,
        )

        # Adjust the embedding layer and lm_head
        model.resize_token_embeddings(len(tokenizer))
        model.vocab_size = len(tokenizer)

        # Verify the size of the embedding layer
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        print(f"Model's embedding layer size: {model.model.embed_tokens.weight.size(0)}")
        print(f"Model's lm_head size: {model.lm_head.weight.size(0)}")


 
    train_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='train',
        tokenizer=tokenizer
    )
    train_dataset[0]
    eval_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='val',
        tokenizer=tokenizer
    ) if os.path.exists(os.path.join(data_args.data_path, 'val_input_ids.memmap')) else None
 
    data_collator = default_data_collator

     
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    if is_main_process:
        trainer.add_callback(transformers.integrations.WandbCallback())

 
    trainer.train(resume_from_checkpoint=last_checkpoint)
 
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
