 
 
This is a sample code demonstrating how to perform offline fine-tuning on a single speaker, such as LJSpeech. You can refer to the `data_instruction` folder for dataset guidance.

### Full-Parameter Training

To fine-tune using all parameters, run the following command:

```bash
torchrun --nproc_per_node=4 finetune_offline.py
```

### LoRA Training

For fine-tuning using LoRA (Low-Rank Adaptation), use the command below:

```bash
torchrun --nproc_per_node=4 finetune_offline_lora.py
```

> **Note:** If you are using your own dataset, you may need to adjust hyperparameters such as the learning rate, LoRA configuration, etc., to achieve the best results.

 