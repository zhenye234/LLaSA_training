import os
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoTokenizer
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
import numpy as np

def pad_audio_batch(batch):
    """
    Collate function for padding each sample in the batch.
    Each sample is formatted as:
       (audio, feat, audio_length, text_tokens, text_length, speaker_id)
       
    - audio has shape [1, time]
    - feat has shape [1, frames, dim]
    - text_tokens is a 1D tensor (variable length, needs padding)
    """
    # Unpack the items (Note: All returned items are tensors or can be converted to tensors)
    audio_list, feat_list, audio_length_list, text_tokens_list, text_length_list = zip(*batch)
    
    # 1. For audio: determine the target length based on the number of frames in feat
    max_length_feat = max(feat.shape[1] for feat in feat_list)
    max_length = max_length_feat * 320  # Calculate target number of audio samples using hop_length=320
    padded_audios = []
    for audio in audio_list:
        padding = max_length - audio.shape[1]
        if padding > 0:
            padded_audio = F.pad(audio, (0, padding), mode='constant', value=0)
        else:
            padded_audio = audio[:, :max_length]
        padded_audios.append(padded_audio)
    padded_audios = torch.stack(padded_audios)
    
    # 2. For auxiliary features: pad the number of frames
    padded_feat_list = []
    for feat in feat_list:
        padding = max_length_feat - feat.shape[1]
        padded_feat = F.pad(feat, (0, 0, 0, padding), mode='constant', value=0)
        padded_feat_list.append(padded_feat)
    padded_feat_list = torch.stack(padded_feat_list)
    
    # 3. For text tokens: pad tokens to the same length since each sample has variable length tokens
    max_text_len = max(t.size(0) for t in text_tokens_list)
    padded_text_tokens = []
    for t in text_tokens_list:
        pad_len = max_text_len - t.size(0)
        # Here, 0 is used as the pad token (you can change it to tokenizer.pad_token_id if needed)
        if pad_len > 0:
            t_padded = F.pad(t, (0, pad_len), value=0)
        else:
            t_padded = t
        padded_text_tokens.append(t_padded)
    padded_text_tokens = torch.stack(padded_text_tokens)
    
    # 4. Convert audio lengths and text lengths to tensors
    audio_length_tensor = torch.tensor(audio_length_list, dtype=torch.long)
    text_length_tensor = torch.tensor(text_length_list, dtype=torch.long)
 
    return {
        "padded_audios": padded_audios,         # Tensor (B, 1, T)
        "padded_feat_list": padded_feat_list,     # Tensor (B, 1, frames, feat_dim)
        "audio_length": audio_length_tensor,      # Tensor (B,)
        "text_tokens": padded_text_tokens,        # Tensor (B, max_text_len)
        "text_length": text_length_tensor,        # Tensor (B,)
 
    }
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor
from torchaudio.transforms import Resample

class WaveDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, sampling_rate, tokenizer, audio_norm_scale: float = 1.0, root_dir: str = ""):
        """
        file_list: A list of data entries, each being a dictionary containing fields such as 'kore', 'puck', and 'text'.
                   Each record is split into two samples (corresponding to 'kore' and 'puck').
        tokenizer: A tokenizer used to convert text into tokens.
        """
        self.data = file_list 
        self.sampling_rate = sampling_rate
        self.audio_norm_scale = audio_norm_scale
        self.hop_length = 320
        self.root_dir = root_dir
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.tokenizer = tokenizer
    
    def __len__(self):
        # Each record corresponds to two samples
        return len(self.data) * 2
    
    def __getitem__(self, index):
        # Map the global index to the specific record and speaker
        row = index // 2
        speaker_index = index % 2
        # Convention: speaker_index==0 indicates 'kore' (corresponding speaker is "puck"),
        #             speaker_index==1 indicates 'puck' (corresponding speaker is "kore")
        # Note: You can adjust this mapping as needed; here we follow the example:
        if speaker_index == 0:
            speaker_key = "kore"
            speaker_str = "kore"
        else:
            speaker_key = "puck"
            speaker_str = "puck"
        
        item = self.data[row]
        text = item.get("text", "")
        speaker_data = item[speaker_key]
        audio_array = speaker_data["array"]
        if audio_array.ndim == 1:
            audio = torch.tensor(audio_array, dtype=torch.float).unsqueeze(0)
        else:
            audio = torch.tensor(audio_array, dtype=torch.float)
        sr = speaker_data["sampling_rate"]
        # Resample (if the sampling rate does not match)
        if sr != self.sampling_rate:
            audio = Resample(sr, self.sampling_rate)(audio)
        if self.audio_norm_scale < 1.0:
            audio = audio * self.audio_norm_scale
        
        # Pad 160 samples on both ends
        audio_pad = F.pad(audio, (160, 160))
        
        # Extract features using feature_extractor
        feat = self.feature_extractor(
            audio_pad,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).data["input_features"]
        
        # Obtain the effective audio length in frames based on hop_length
        audio_length = int(audio.shape[1] / self.hop_length)
        
        # Process the text directly: first add special markers, then include speaker information in the prompt,
        # and finally call tokenizer.apply_chat_template to tokenize.
        text_with_special = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + text_with_special},
            {"role": "assistant", "content": f"Speaker {speaker_str}"}
        ]
        text_tokens = self.tokenizer.apply_chat_template(chat, tokenize=True, continue_final_message=True)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_length = text_tokens.size(0)
        
        # Return: audio, feat, audio_length, text_tokens, text_length
        return audio, feat, audio_length, text_tokens, text_length


# Test code below
if __name__ == '__main__':
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    
    # Load the dataset (example using Hugging Face datasets)
    # dataset = load_dataset("shb777/gemini-flash-2.0-speech",
    #                        cache_dir="/aifs4su/data/zheny/opensource/local_data160/data")
    # data_split = dataset['en']
    data_split = load_dataset(
        "shb777/gemini-flash-2.0-speech",
        # split="en[:5000]",  # Only load the first 5000 records
        split="en",
        cache_dir="/aifs4su/data/zheny/opensource/local_data160/data"
    )

    # Initialize the tokenizer (choose a pre-trained model as needed)
    tokenizer = AutoTokenizer.from_pretrained("HKUSTAudio/Llasa-1B")
    
    # Instantiate the dataset (pass in the tokenizer)
    wave_dataset = WaveDataset(data_split, sampling_rate=16000, tokenizer=tokenizer)
    
    # Construct the DataLoader, specifying collate_fn
    loader = DataLoader(wave_dataset, batch_size=2, collate_fn=pad_audio_batch)
    
    # Retrieve one batch and check the results
    batch = next(iter(loader))
    print("padded_audios shape:", batch["padded_audios"].shape)       # (B, 1, T)
    print("padded_feat_list shape:", batch["padded_feat_list"].shape)   # (B, 1, frames, feat_dim)
    print("audio_length:", batch["audio_length"])                       # (B,)
    print("text_tokens shape:", batch["text_tokens"].shape)             # (B, max_text_len)
    print("text_length:", batch["text_length"])                         # (B,)
    print("speaker_ids:", batch["speaker_ids"])                         # (B,)
