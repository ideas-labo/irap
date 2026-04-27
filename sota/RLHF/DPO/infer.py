import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
from utils.utils import load_pickle, read_md_file

# =================Configuration Area=================
# 1. Base model path (must be consistent with training)
BASE_MODEL_PATH = 'sota/RLHF/models/qwen/Qwen2.5-7B-Instruct/'

# 2. Path to trained LoRA adapter weights
LORA_ADAPTER_PATH = "sota/RLHF/models/DPO_output/checkpoint-1280/"

# 3. Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================================

def load_dpo_model():
    print(f"Loading base model: {BASE_MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16, # bf16 or fp16 is recommended for inference
        trust_remote_code=True
    )

    print(f"Loading DPO LoRA weights: {LORA_ADAPTER_PATH} ...")
    # Attach LoRA adapter to base model
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    # Switch to evaluation mode
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, query, max_new_tokens=1024):
    """
    Generate response
    """
    # 1. Build conversation template (Qwen Instruct models require specific Chat Template)
    messages = [
        {"role": "user", "content": query}
    ]
    
    # apply_chat_template automatically adds special tokens like <|im_start|>user...<|im_end|>
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 2. Encode input
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 3. Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,  # Control randomness, DPO models usually don't need high temperature
            top_p=0.9,        # Nucleus sampling
            do_sample=True,
            repetition_penalty=1.05, # Prevent repetition
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. Decode output
    # generated_ids includes the input prompt, we need to remove the input part and keep only the output
    input_len = model_inputs.input_ids.shape[1]
    generated_ids = [output_ids[input_len:] for output_ids in generated_ids]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == '__main__':
    # 1. Load model
    model, tokenizer = load_dpo_model()
    
    print("\n" + "="*50)
    print("DPO model loaded successfully! Starting conversation test")
    print("="*50 + "\n")

    data_path = 'dataset/pkl/final_synthetic_dataset.pkl'
    prompt_path = 'sota/RLHF/DPO/prompt.md'
    
    if os.path.exists(data_path) and os.path.exists(prompt_path):
        json_datas = load_pickle(data_path)
        base_prompt = read_md_file(prompt_path)
        datas = []
        for json_data in json_datas:
            data = {
                "prompt": base_prompt + json_data.get("sentence", ""), 
                "chosen": str(json_data.get("prefer_label", "")), 
                "rejected": str(json_data.get("label", ""))
            }
            datas.append(data)

    for i in range(10):
        query = datas[i]["prompt"]
        response = generate_response(model, tokenizer, query)
        print("real : ", datas[i]["chosen"])
        print("pred : ",response)
        print("========================\n")