import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from utils.utils import read_md_file, load_pickle
import json

# =================Configuration Area=================
# 1. Base model path (must be consistent with training)
BASE_MODEL_PATH = 'sota/RLHF/models/qwen/Qwen2.5-7B-Instruct/'

# 2. Path to trained LoRA adapter weights
LORA_ADAPTER_PATH = "sota/RLHF/models/DPO_output/checkpoint-1280/"

# 3. Path to general prompt template
PROMPT_PATH = 'sota/RLHF/DPO/prompt.md'

# 4. Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

max_new_tokens=1024

class DPO():
    def __init__(self):
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
        # Attach LoRA adapter to base model
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        
        # Switch to evaluation mode
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        self.base_prompt = read_md_file(PROMPT_PATH)

    def gen_ans(self, obj):
        
        prompt = self.base_prompt + obj["sentence"]
        # 1. Build conversation template (Qwen Instruct models require specific Chat Template)
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # apply_chat_template automatically adds special tokens like <|im_start|>user...<|im_end|>
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 2. Encode input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        # 3. Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,  # Control randomness, DPO models usually don't need high temperature
                top_p=0.9,        # Nucleus sampling
                do_sample=True,
                repetition_penalty=1.05, # Prevent repetition
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 4. Decode output
        # generated_ids includes the input prompt, we need to remove the input part and keep only the output
        input_len = model_inputs.input_ids.shape[1]
        generated_ids = [output_ids[input_len:] for output_ids in generated_ids]
        
        res = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        try:
            res = json.loads(res)
            return res
        except:
            res = obj["label"] # Return unadjusted result if answer format is invalid
            return res



if __name__ == '__main__':
    data_path = 'dataset/pkl/final_synthetic_dataset.pkl'
    prompt_path = 'sota/RLHF/DPO/prompt.md'
    
    if os.path.exists(data_path) and os.path.exists(prompt_path):
        datas = load_pickle(data_path)
    
    dpo = DPO()
    print(dpo.gen_ans(datas[0]))