[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.04128)

**Update (2025-02-07):** Our paper has been released! Llasa 1b Multilingual version released!

## Training
```bash
torchrun --nproc_per_node=8 train_tts.py config.json 
```

or 

```bash
sbatch run_slurm.sh
```

## Data

You can download tokenized open-source speech data [here](https://huggingface.co/datasets/HKUST-Audio/Llasa_opensource_speech_data_160k_hours_tokenized/tree/main). This includes LibriHeavy, Emilia (in both Chinese and English), and WenetSpeech4TTS, totaling approximately 160,000 hours of open-source data.

Our models are trained on 250,000 hours of speech data. Of this, 160,000 hours come from the open-source datasets mentioned above, while the remaining 90,000 hours are from internal datasets, which are not yet available for open-source release.


 
## Directly used on Hugging Face

**Codec**: [xcodec2](https://huggingface.co/HKUST-Audio/xcodec2) (Please install new version xcodec2==0.1.3)
 


**Llasa 1b version**: [Llasa-1B](https://huggingface.co/HKUSTAudio/Llasa-1B)

**Llasa 1b Multilingual version**: [Llasa-1B-Multilingual](https://huggingface.co/HKUSTAudio/Llasa-1B-Multilingual) (Not mentioned in the paper)

**Llasa 3b version**: [Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B)

**Llasa 8b version**: [Llasa-8B](https://huggingface.co/HKUSTAudio/Llasa-8B)  
