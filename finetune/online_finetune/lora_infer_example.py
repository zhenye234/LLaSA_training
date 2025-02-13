from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import soundfile as sf

# Define the path for the base model and LoRA model (replace with your actual paths)
base_model_name = 'HKUSTAudio/Llasa-1B'
lora_model_path = '/path/to/lora_folder'   

# Load the tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the LoRA fine-tuned model: only the LoRA parameters are trainable, but during inference, they are applied to the base model
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()
model.to('cuda')

# Load the speech codec model (remains unchanged)
from xcodec2.modeling_xcodec2 import XCodec2Model
codec_model_path = "HKUSTAudio/xcodec2"
Codec_model = XCodec2Model.from_pretrained(codec_model_path)
Codec_model.eval().cuda()

input_text = 'Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, intending to safeguard some from the harsh truths. One day, I hope you understand the reasons behind my actions. Until then, Anna, please, bear with me.'

def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

# TTS inference process
with torch.no_grad():
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
    
    # Construct the dialogue template
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "Speaker kore <|SPEECH_GENERATION_START|>"}  #choose speaker here! can also be puck
    ]
    
    # Generate input token IDs using the template
    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True
    )
    input_ids = input_ids.to('cuda')
    
    # Get the end token ID
    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    
    # Autoregressive generation of speech token sequence
    outputs = model.generate(
        input_ids,
        max_length=2048,  # Match the training length
        eos_token_id=speech_end_id,
        do_sample=True,
        top_p=1,          # Control diversity of generation
        temperature=0.8,  # Control randomness
        repetition_penalty=1.2,
    )
    
    # Slice the generated token IDs (exclude the input part)
    generated_ids = outputs[0][input_ids.shape[1]:-1]
    
    # Decode the generated token IDs into strings (e.g., "<|s_12345|>")
    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Convert string tokens into integer list
    speech_tokens = extract_speech_ids(speech_tokens)
    
    # Format the tensor to match the Codec_model.decode_code input shape
    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
    
    # Use the speech codec model to convert the tokens into waveform
    gen_wav = Codec_model.decode_code(speech_tokens)
    
# Save the generated speech file
sf.write("gen.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)
