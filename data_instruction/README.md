 

## Guide: Converting Extracted Code into Llasa Training Format

This guide explains how to convert the extracted code using [X-Codec 2.0](https://github.com/zhenye234/X-Codec-2.0/blob/main/inference_save_code.py) into LLASA training format, using [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) as an example.

### 1. Download the Code

You can download the pre-extracted code [here](https://huggingface.co/datasets/HKUSTAudio/Llasa_opensource_speech_data_160k_hours_tokenized/blob/main/LJSpeech_codes_example.tar.gz).

### 2. Run the Script

Next, run the `get_memmap_from_token.py` script. This will generate the following file structure:

```
ljspeech_bin
    ├── train_input_ids_shape.npy
    ├── train_input_ids.memmap
    ├── val_input_ids_shape.npy
    └── val_input_ids.memmap
```

You may need to modify the `get_memmap_from_token.py` script to match your own data format.

 