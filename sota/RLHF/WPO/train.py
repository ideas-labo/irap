import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import os
# Assume utils.utils and sota.RLHF.DPO.dataset paths are correct
from utils.utils import read_md_file, load_pickle 
from sota.RLHF.DPO.dataset import PreferenceDataset 

# --- Import PEFT related modules ---
from peft import LoraConfig, TaskType, get_peft_model


# ==========================================
# WPO Core Logic (Weighted Preference Optimization)
# ==========================================

def get_batch_logps_and_weights(logits, labels):
    """
    Calculate sequence Log Probabilities and key terms in WPO weights.
    
    WPO weight formula:
    w(x, y) = exp( (1/|y|) * sum_{t} [ log(pi_theta(y_t | x, y_<t)) - log(sum_{v} pi_theta(v | x, y_<t)^2) ] )
    
    Returns:
        policy_logps: Total log probabilities of sequences (used for log-ratio)
        normalized_prob_sum: Length-normalized sum of log(pi_theta(y_t)) (WPO weight term 1)
        normalized_expected_prob_sum: Length-normalized sum of -log(sum_{v} pi_theta(v)^2) (WPO weight term 2)
    """
    # Logits and labels should be strictly aligned at this point
    if logits.shape[1] != labels.shape[1]:
        min_len = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :min_len, :]
        labels = labels[:, :min_len]

    # Shift: logits[t] predicts labels[t+1]
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    loss_mask = (labels != -100)
    sequence_lengths = loss_mask.sum(-1).float() 
    
    # Avoid division by zero by setting sequence length to 1 for zero-length sequences 
    # (these sequences have loss_mask sum of 0, so final results won't be affected by division by zero)
    sequence_lengths[sequence_lengths == 0] = 1 
    
    # Replace -100 with 0 for gather operation
    labels[labels == -100] = 0

    log_softmax = logits.log_softmax(-1)
    
    # --- 1. Calculate per_token_logps: log(pi_theta(y_t | x, y_<t)) ---
    per_token_logps = torch.gather(log_softmax, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # 1.1 Policy Logps (for log-ratio numerator): sum of per_token_logps * loss_mask
    policy_logps = (per_token_logps * loss_mask).sum(-1)
    
    # 1.2 WPO weight term 1: Length-normalized sum of log(pi_theta(y_t))
    normalized_prob_sum = (per_token_logps * loss_mask).sum(-1) / sequence_lengths

    # --- 2. Calculate WPO weight term 2: Negative log of expected squared probability sum ---
    
    # Policy probability pi_theta = exp(log_softmax)
    pi_theta = log_softmax.exp()
    
    # Calculate denominator squared sum: sum_{v} pi_theta(v)^2
    sum_pi_theta_sq = (pi_theta ** 2).sum(dim=-1)
    
    # log( 1 / sum_{v} pi_theta(v)^2 ) = -log( sum_{v} pi_theta(v)^2 )
    # Add epsilon to prevent log(0) or log(very small values)
    log_inv_expected_prob_sq = -(sum_pi_theta_sq + 1e-8).log()
    
    # WPO weight term 2: Length-normalized sum of -log(sum(pi_theta(v)^2))
    normalized_expected_prob_sum = (log_inv_expected_prob_sq * loss_mask).sum(-1) / sequence_lengths
    
    return policy_logps, normalized_prob_sum, normalized_expected_prob_sum

def wpo_loss(policy_chosen_logps, policy_rejected_logps, 
             reference_chosen_logps, reference_rejected_logps, 
             chosen_prob_sum, rejected_prob_sum, 
             chosen_expected_prob_sum, rejected_expected_prob_sum, 
             beta=0.1):
    """
    Weighted Preference Optimization (WPO) loss function.
    L_WPO = -E [ w(x, y_w) * w(x, y_l) * log(sigmoid(beta * logits)) ]
    """
    
    # --- 1. Calculate WPO weights w(x, y) ---
    # log(w(x, y)) = term 1 + term 2 
    log_w_chosen = chosen_prob_sum + chosen_expected_prob_sum
    log_w_rejected = rejected_prob_sum + rejected_expected_prob_sum
    
    # WPO weight w(x, y) = exp(log_w)
    w_chosen = log_w_chosen.exp()
    w_rejected = log_w_rejected.exp()
    
    # Combined weight W = w(x, y_w) * w(x, y_l)
    W = w_chosen * w_rejected
    
    # --- 2. Calculate DPO Logits ---
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    
    # --- 3. Calculate weighted loss ---
    # losses = -W * log(sigmoid(beta * logits))
    losses = -W * F.logsigmoid(beta * logits)
    
    return losses.mean()


class WPOTrainer(Trainer): # Replace DPOTrainer with WPOTrainer
    def __init__(self, ref_model, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        concatenated_batch = {}
        # Concatenate chosen and rejected samples
        for k in ["input_ids", "attention_mask", "labels"]:
            concatenated_batch[k] = torch.cat((inputs[f"chosen_{k}"], inputs[f"rejected_{k}"]), dim=0)
            
        len_chosen = inputs["chosen_input_ids"].shape[0]

        # --- 1. Policy Model (Calculate Policy Logps and WPO weight terms) ---
        all_logits = model(
            input_ids=concatenated_batch["input_ids"],
            attention_mask=concatenated_batch["attention_mask"]
        ).logits

        all_logps, all_prob_sum, all_expected_prob_sum = get_batch_logps_and_weights(
            all_logits, concatenated_batch["labels"]
        )

        # Split Policy Logps
        policy_chosen_logps = all_logps[:len_chosen]
        policy_rejected_logps = all_logps[len_chosen:]
        
        # Split WPO weight terms (only need Policy Model outputs)
        chosen_prob_sum = all_prob_sum[:len_chosen]
        rejected_prob_sum = all_prob_sum[len_chosen:]
        chosen_expected_prob_sum = all_expected_prob_sum[:len_chosen]
        rejected_expected_prob_sum = all_expected_prob_sum[len_chosen:]


        # --- 2. Reference Model (Calculate Reference Logps) ---
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=concatenated_batch["input_ids"],
                attention_mask=concatenated_batch["attention_mask"]
            ).logits
            
            # Reference Model only needs Logps (WPO weights are based solely on Policy Model), ignore weight terms
            ref_logps, _, _ = get_batch_logps_and_weights(ref_logits, concatenated_batch["labels"])
            
            reference_chosen_logps = ref_logps[:len_chosen]
            reference_rejected_logps = ref_logps[len_chosen:]

        # --- 3. Calculate WPO loss ---
        loss = wpo_loss(
            policy_chosen_logps, policy_rejected_logps, 
            reference_chosen_logps, reference_rejected_logps, 
            chosen_prob_sum, rejected_prob_sum, 
            chosen_expected_prob_sum, rejected_expected_prob_sum, 
            beta=self.beta
        )
        
        return (loss, all_logits) if return_outputs else loss

# --- Simple Collator (Unchanged) ---
def simple_stack_collator(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([x[key] for x in batch])
    return out

# ==========================================
# Main Program
# ==========================================

if __name__ == '__main__':

    # --- 1. First load Tokenizer (required for data processing) ---
    model_dir = 'sota/RLHF/models/qwen/Qwen2.5-7B-Instruct/'
    print(f"Loading Tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    
    # Fix Tokenizer's Pad Token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 2. Load and process data---
    print("Loading and Formatting Dataset...")
    data_path = 'dataset/pkl/final_synthetic_dataset.pkl'
    prompt_path = 'sota/RLHF/WPO/prompt.md'
    
    # Assume data can be loaded successfully here, otherwise ValueError from previous issue will be triggered
    if os.path.exists(data_path) and os.path.exists(prompt_path):
        json_datas = load_pickle(data_path)
        base_prompt = read_md_file(prompt_path)
        datas = []
        
        # === Debug Point: Check raw data quantity (for locating previous error) ===
        print(f"DEBUG: Loaded {len(json_datas)} raw JSON data items.")
        
        for json_data in json_datas:
            # === Build Prompt with Chat Template ===
            messages = [
                # Use sentence from dataset as User Input
                {"role": "user", "content": base_prompt + json_data.get("sentence", "")}
            ]
            
            # Use tokenizer to automatically convert format
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            data = {
                "prompt": formatted_prompt, 
                "chosen": str(json_data.get("prefer_label", "")), 
                "rejected": str(json_data.get("label", ""))
            }
            datas.append(data)
            
        # Print one sample to check format
        if datas:
            print(f"Sample Formatted Prompt:\n{datas[0]['prompt']}")
        else:
            print("Warning: Processed data list is empty.")
            
    else:
        print("Warning: Data files not found, using dummy data.")
        # Dummy data
        dummy_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Test Query"}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        datas = [{"prompt": dummy_prompt, "chosen": "Good answer", "rejected": "Bad answer"}] * 10

    # --- 3. Load model ---
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64, 
        lora_alpha=64, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    print("Creating Policy Model with LoRA...")
    policy_model = get_peft_model(base_model, lora_config)
    policy_model.print_trainable_parameters()

    print("Loading Reference Model...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    reference_model.eval()
    reference_model.requires_grad_(False)

    # --- 4. Instantiate Dataset ---
    dataset = PreferenceDataset(datas, tokenizer, max_length=1024)
    # === Debug Point: Check Dataset length ===
    print(f"DEBUG: Dataset length: {len(dataset)}")

    output_dir = "sota/RLHF/models/WPO_output/"

    training_args = TrainingArguments(
        output_dir=output_dir, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        num_train_epochs=2, 
        learning_rate=1e-5,
        logging_steps=1, 
        save_steps=50, 
        save_total_limit=2, 
        bf16=True, 
        remove_unused_columns=False, 
        run_name="wpo_lora_run", # Change run_name
        report_to="none" 
    )

    print("Initializing WPOTrainer...")
    # ************************************************
    # *** Key Change: Replace DPOTrainer with WPOTrainer ***
    # ************************************************
    trainer = WPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, 
        data_collator=simple_stack_collator, 
        beta=0.1 
    )

    print("Starting Training...")
    trainer.train()

    print(f"Saving LoRA weights to {output_dir}")
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training Completed.")