from torch.utils.data import Dataset
import torch

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Ensure tokenizer has pad_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=self.max_length // 2)
        chosen_tokens = self.tokenizer.encode(chosen, add_special_tokens=False, truncation=True, max_length=self.max_length // 2)
        rejected_tokens = self.tokenizer.encode(rejected, add_special_tokens=False, truncation=True, max_length=self.max_length // 2)

        # Concatenate sequences
        chosen_sequence = prompt_tokens + chosen_tokens
        rejected_sequence = prompt_tokens + rejected_tokens

        # Add EOS token
        if len(chosen_sequence) == 0 or chosen_sequence[-1] != self.tokenizer.eos_token_id:
            chosen_sequence.append(self.tokenizer.eos_token_id)
        if len(rejected_sequence) == 0 or rejected_sequence[-1] != self.tokenizer.eos_token_id:
            rejected_sequence.append(self.tokenizer.eos_token_id)

        # Truncate sequences
        chosen_sequence = chosen_sequence[:self.max_length]
        rejected_sequence = rejected_sequence[:self.max_length]

        # Create Labels (Mask prompt part with -100)
        chosen_labels = [-100] * len(prompt_tokens) + chosen_sequence[len(prompt_tokens):]
        rejected_labels = [-100] * len(prompt_tokens) + rejected_sequence[len(prompt_tokens):]

        # --- [Key Fix] Manual Padding ---
        # Must ensure input_ids, mask, and labels have exactly the same length (max_length)
        
        def pad_seq(seq, pad_val, max_len):
            pad_len = max_len - len(seq)
            return seq + [pad_val] * pad_len

        # Pad Input IDs (using pad_token_id)
        chosen_input_ids = pad_seq(chosen_sequence, self.pad_token_id, self.max_length)
        rejected_input_ids = pad_seq(rejected_sequence, self.pad_token_id, self.max_length)

        # Pad Labels (using -100)
        chosen_labels = pad_seq(chosen_labels, -100, self.max_length)
        rejected_labels = pad_seq(rejected_labels, -100, self.max_length)

        # Create Masks (1 for non-padding tokens, 0 for padding tokens)
        # Note: Generate mask based on original sequence length first, then pad with 0s
        chosen_mask = [1] * len(chosen_sequence) + [0] * (self.max_length - len(chosen_sequence))
        rejected_mask = [1] * len(rejected_sequence) + [0] * (self.max_length - len(rejected_sequence))

        return {
            'chosen_input_ids': torch.tensor(chosen_input_ids, dtype=torch.long),
            'chosen_attention_mask': torch.tensor(chosen_mask, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_input_ids, dtype=torch.long),
            'rejected_attention_mask': torch.tensor(rejected_mask, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
        }