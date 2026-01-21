[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.04128)


## Directly used on Hugging Face

**Codec**: [xcodec2](https://huggingface.co/HKUST-Audio/xcodec2)  
 


**TTS inference: Llasa-collections**: [Llasa-collections](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44)



**Update (2025-06-09):** Add Llasa finetune instruction.
Awesome work based on Llasa
- [LLaSa GRPO tuning](https://github.com/channel-io/ch-tts-llasa-rl-grpo) : Special thanks to [Seungyoun Shin](https://github.com/SeungyounShin) from [Channel Corp.](https://channel.io/en) for sharing the Llasa GRPO RL-tuning script based on Verl.
You can try the finetuning results here:
- [LLaSA 1B Multi-Speakers (Genshin-zh-en-ja-ko)](https://huggingface.co/spaces/HKUST-Audio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko)
- [LLaSA 1B Finetuned for Two Speakers](https://huggingface.co/spaces/HKUST-Audio/Llasa-1B-finetuned-for-two-speakers)


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

## Data instruction 
 

[Text_sequence](https://github.com/zhenye234/LLaSA_training/blob/5ffcddee243f0aa594ebfc089f4327a24f7cac6f/train_tts.py#L111) is encoded by the  text tokenizer from Llama, for example, [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 

[Speech_sequence](https://github.com/zhenye234/LLaSA_training/blob/5ffcddee243f0aa594ebfc089f4327a24f7cac6f/train_tts.py#L112) is extrated through [X-codec2](https://github.com/zhenye234/X-Codec-2.0)  We change the value of speech tokens by adding  len(text tokenizer) +8 [special tokens](https://github.com/zhenye234/LLaSA_training/blob/1d65cf3e34c0d5b508404d67ff41b3b6fb1ecab7/train_tts.py#L67) thereby forming a unified tokenizer that encompasses both speech and text.

 

 
