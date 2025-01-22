# LLaSA_training
LLaSA: Scaling Train-time and Test-time Compute for LLaMA-based Speech Synthesis (Comming Soon!)

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
 
**LLaMa based TTS 3b version**: [Llasa-3B](https://huggingface.co/HKUST-Audio/Llasa-3B)

**LLaMa based TTS 1b version**: [Llasa-1B](https://huggingface.co/HKUST-Audio/Llasa-1B)

**LLaMa based TTS 8b version**: [Llasa-8B](https://huggingface.co/HKUST-Audio/Llasa-8B) (Comming Soon!)
