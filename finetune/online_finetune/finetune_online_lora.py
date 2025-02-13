import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    PreTrainedModel
)
import transformers
import wandb
from peft import LoraConfig, get_peft_model
def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

# Directly import your previous dataset and collate function
from tts_online_dataset import WaveDataset, pad_audio_batch


class llm_with_codec_model(PreTrainedModel):
    def __init__(self, config, llm: nn.Module, encoder: nn.Module, tokenizer=None):
        """
        Parameters:
          - config: Model configuration object (should contain or specify the llm's name/path)
          - llm: Causal language model (e.g., AutoModelForCausalLM) used to predict speech tokens
          - encoder: Speech codec model (must implement encode_batch_feats(input_waveform, input_features))
          - tokenizer: Tokenizer
        """
        super().__init__(config)
        self.config = config
        self.llm = llm
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.base_num = 128256 + 8  # length of llama tokenizer + 8 new special tokens
    
    def get_speech_token(self, input_waveform, input_features):
        """
        Extract speech token sequence using the encoder.
        It is assumed that encoder.encode_batch_feats returns a tensor whose shape could be (B, 1, seq_len) or (B, seq_len).
        If the returned shape is (B, 1, seq_len), squeeze out the 1st dimension.
        """
        with torch.no_grad():
            speech_tokens = self.encoder.encode_batch_feats(
                input_waveform=input_waveform,
                input_features=input_features
            )
        if speech_tokens.dim() == 3 and speech_tokens.size(1) == 1:
            speech_tokens = speech_tokens.squeeze(1)
        return speech_tokens 

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, **batch):
        """
        Forward pass implementing the TTS training procedure:
          1. The dataset returns a tokenized text prompt (key "text_tokens") along with its actual length ("text_length").
          2. Use the encoder to extract speech tokens, then truncate based on "audio_length", and add 
             <|SPEECH_GENERATION_START|> and <|SPEECH_GENERATION_END|> tokens at the beginning and end.
          3. For each sample, concatenate the text tokens (only the valid portion) with the processed speech tokens.
             If the total length is less than 2048, pad; otherwise, truncate to 2048.
          4. Construct labels: set the text portion (the first text_len tokens) to ignore_index so that only the speech token part contributes to the loss.
        """
        # Retrieve tensors from the batch
        padded_audios = batch["padded_audios"]         # Tensor, shape (B, 1, T)
        padded_feats = batch["padded_feat_list"]         # Tensor, shape (B, 1, frames, feat_dim)
        audio_length = batch["audio_length"]             # Tensor, shape (B,)
        text_tokens = batch["text_tokens"]               # Tensor, shape (B, L_text_padded)
        text_length = batch["text_length"]               # Tensor, shape (B,)
        
        batch_size = padded_audios.size(0)
        # For the text prompt, extract the actual token list for each sample
        text_length_list = text_length.tolist()
        all_text_tokens = []
        for i in range(batch_size):
            tokens = text_tokens[i, :text_length_list[i]].tolist()
            all_text_tokens.append(tokens)
        
        # Use the encoder to extract the speech token sequence
        speech_tokens_all = self.get_speech_token(
            input_waveform=padded_audios,
            input_features=padded_feats
        )  # Expected shape: (B, seq_len)
        
        processed_speech_tokens = []
        # Get special token ids for speech generation
        speech_gen_start_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        speech_gen_end_id   = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        audio_length_list = audio_length.tolist()
        for i in range(batch_size):
            valid_length = audio_length_list[i]
            # Extract the valid part of the speech tokens and convert to list
            tokens = speech_tokens_all[i, :valid_length] + self.base_num
            tokens = tokens.tolist()
            # Add special tokens at the beginning and end
            tokens = [speech_gen_start_id] + tokens + [speech_gen_end_id]
            processed_speech_tokens.append(tokens)
        
        # Concatenate the text tokens and the processed speech tokens for each sample, ensuring a fixed length of 2048
        combined_tokens = []
        max_total_length = 2048
        for text_tok, speech_tok in zip(all_text_tokens, processed_speech_tokens):
            combined = text_tok + speech_tok
            if len(combined) > max_total_length:
                combined = combined[:max_total_length]
            else:
                pad_len = max_total_length - len(combined)
                combined = combined + [self.tokenizer.pad_token_id] * pad_len
            combined_tokens.append(combined)
        input_ids = torch.tensor(combined_tokens, dtype=torch.long, device=padded_audios.device)
        
        # Construct labels: set the text portion (the first t_len tokens) to ignore_index, keeping the speech tokens unchanged
        labels = input_ids.clone()
        for i, t_len in enumerate(text_length_list):
            labels[i, :t_len] = self.ignore_index
        labels[input_ids == self.tokenizer.pad_token_id] = self.ignore_index

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def freeze_encoder(self):
        freeze_model(self.encoder)        

    # Override state_dict method to return only the llm part's parameters
    def state_dict(self, *args, **kwargs):
        return self.llm.state_dict(*args, **kwargs)

    # Override save_pretrained method to save only the llm part
    def save_pretrained(self, save_directory, **kwargs):
        self.llm.save_pretrained(save_directory, **kwargs)

############################################
# Arguments and Main Function
############################################

@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the data."})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(default=None, metadata={"help": "Integration to report results."})
    run_name: Optional[str] = field(default=None, metadata={"help": "The name of the run for logging."})
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type"})

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        default_config_file = 'config_lora.json'
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(default_config_file))
    
    is_main_process = training_args.local_rank in [-1, 0]
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(project="llm_tts", config=training_args.to_sanitized_dict(), name=training_args.run_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        torch_dtype='auto',
        cache_dir=model_args.cache_dir,
    )
    config = transformers.AutoConfig.from_pretrained(model_args.llm_model_name_or_path)
    device_id = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Import the speech codec model, for example XCodec2Model
    from xcodec2.modeling_xcodec2 import XCodec2Model
    model_path = "HKUST-Audio/xcodec2"
    Codec_model = XCodec2Model.from_pretrained(model_path)



    lora_config = LoraConfig(
        r=8,                      
        lora_alpha=32,           
        target_modules=["q_proj", "v_proj"],   
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
 
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", trainable_params)
    # Load dataset (example using Hugging Face datasets)
    # dataset = load_dataset("shb777/gemini-flash-2.0-speech",
    #                        cache_dir="/aifs4su/data/zheny/opensource/local_data160/data")
    data_split = load_dataset(
        "shb777/gemini-flash-2.0-speech",
        # split="en[:5000]",  # Only load the first 5000 records
        split="en", 
        cache_dir="/aifs4su/data/zheny/opensource/local_data160/data"
    )

    # train_test_split = dataset['en'].train_test_split(test_size=0.005)
    train_test_split = data_split.train_test_split(test_size=0.005)
    train_dataset_raw = train_test_split["train"]
    test_dataset_raw = train_test_split["test"]
    
    # Instantiate custom dataset (pass in tokenizer for prompt construction and tokenization)
    train_dataset = WaveDataset(train_dataset_raw, sampling_rate=16000, tokenizer=tokenizer)
    test_dataset = WaveDataset(test_dataset_raw, sampling_rate=16000, tokenizer=tokenizer)
    
    lwc_model = llm_with_codec_model(config, model, Codec_model, tokenizer)
    lwc_model = lwc_model.to(device)
    lwc_model.freeze_encoder()
    
    trainer = Trainer(
        model=lwc_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=pad_audio_batch,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
