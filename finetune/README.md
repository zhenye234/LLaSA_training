
## TTS Models Collection

You can access the collection of Llasa TTS models at [this link](https://huggingface.co/collections/HKUSTAudio/llasa-679b87dbd06ac556cc0e0f44).

We offers two finetuning strategies: **Online Finetuning** and **Offline Finetuning**.

### **Online Finetuning**
In this approach, the codec is used to directly extract the code during the training process. All you need is a dataset containing audio (wav), text, and any additional condition information. Compared to offline finetuning, this strategy requires more GPU memory because the codec model is involved during training.

### **Offline Finetuning**
With this strategy, audio is pre-encoded into codes before training begins, which helps reduce GPU memory usage during training. However, this approach is more complex, as it involves additional steps prior to the actual training process.

### Model Overview
The architecture and training process of our model are designed to be simple: with text input at the left and speech at the right. This simplicity makes it suitable as a base model for finetuning across a variety of scenarios (such as single or multi-speaker, emotion, or voice description). 

- For tasks involving **Chinese** and **English** speech synthesis, you can choose from models LLaSA 1B, 3B, or 8B.

- For **multilingual TTS tasks**, we also offer the LLaSA 1B multilingual model.

### Text Chat Preservation Models
In addition to the standard models, we are releasing two models that preserve text chat abilities. More details on this can be found in [this GitHub issue](https://github.com/zhenye234/LLaSA_training/issues/7).

Although these models are not mentioned in the original paper, they are essentially the same as LLaSA 1B and LLaSA 3B, except they have been fine-tuned with a mixed speech and text SFT dataset, which enables the model to retain text-based conversational abilities.

If you're interested, these models can be used for creating some fun and interesting applications involving text-based conversations!

 
### Fine-tune on Other Languages
If you're interested in fine-tuning on other languages, we recommend reading [this blog post](https://huggingface.co/blog/Steveeeeeeen/llasagna) for more insights.
