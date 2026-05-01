# Set up mirror
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Modification 1: Import Auto classes for better generality
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from torch.optim import AdamW
import torch
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------------------------------
# 1. Load data (unchanged)
# -----------------------------------------------------
data_path = "IRAPE/Base_gen/dataset/train.jsonl"
datas = []

try:
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            datas.append(data)
except FileNotFoundError:
    print(f"Error: Data file '{data_path}' not found.")
    print("Will use sample data for demonstration.")
    datas = [
        {"input": "ReqView Desktop shall start in less than 10s.\n", "output": "[[10.0, 1.0], [11.0, 0.0]]", "class": 2, "number": 10.0},
        {"input": "The search function shall return results within 5 seconds.\n", "output": "[[5.0, 1.0], [6.0, 0.0]]", "class": 2, "number": 5.0},
        {"input": "The system must handle 500 concurrent users.\n", "output": "[[500.0, 1.0], [501.0, 0.0]]", "class": 2, "number": 500.0}
    ]

# -----------------------------------------------------
# 2. Define Prompt and hyperparameters (modified)
# -----------------------------------------------------
prompt = "Instruction: Extract the numerical performance threshold from the requirement statement below. Your output MUST be ONLY the bare number, nothing else (no units, no text, no punctuation, no surrounding quotes). Requirement:\n"

# Modification 2: Upgrade model to gpt2-large (774M parameters)
# Compared to gpt2 (124M), it is more intelligent and fully compatible with your existing LogitsProcessor
MODEL_NAME = 'gpt2-large' 

# Modification 3: Adjust Batch Size
# Memory usage increases sharply with larger models, the original 16 may cause OOM (Out of Memory)
# Suggest starting with 2, adjust to 4 or 8 if memory is sufficient
BATCH_SIZE = 2 
LEARNING_RATE = 5e-5
EPOCHS = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print(f"Current model: {MODEL_NAME}")
print(f"Loaded data: {len(datas)} entries")

# -----------------------------------------------------
# 3. Initialize model and tokenizer (using Auto classes)
# -----------------------------------------------------
print(f"Loading model: {MODEL_NAME}...")
# Use Auto interface for automatic matching
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# GPT-2 series has no default pad token, set it to eos token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

model.to(device)
print("Model loaded successfully.")

# -----------------------------------------------------
# 4. Logits Processor definition (unchanged)
# -----------------------------------------------------
class NumberOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        # Define allowed characters
        allowed_chars = set('0123456789.-+/')
        # Convert these characters to token IDs
        self.allowed_token_ids = set()
        for char in allowed_chars:
            # gpt2-large shares the same vocabulary as gpt2, so the logic here is fully universal
            token_ids = tokenizer.encode(char, add_special_tokens=False)
            if len(token_ids) == 1:
                self.allowed_token_ids.add(token_ids[0])
        # Always include EOS token
        self.allowed_token_ids.add(tokenizer.eos_token_id)
        print(f"NumberOnlyLogitsProcessor: Allowed token IDs loaded.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float('inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0.0
        return scores + mask

# -----------------------------------------------------
# 5. Create custom dataset (unchanged)
# -----------------------------------------------------
class CustomGenDataset(Dataset):
    def __init__(self, data, tokenizer, prompt):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.processed_data = []
        self._build_dataset()

    def _build_dataset(self):
        for item in self.data:
            input_text = self.prompt + item["input"]
            target_text = str(item["number"]) + self.tokenizer.eos_token
            input_encoding = self.tokenizer(input_text)
            full_encoding = self.tokenizer(input_text + target_text)
            input_len = len(input_encoding['input_ids'])
            input_ids = full_encoding['input_ids']
            attention_mask = full_encoding['attention_mask']
            labels = list(input_ids)
            labels[:input_len] = [-100] * input_len
            
            self.processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

# -----------------------------------------------------
# 6. Define data collate function (unchanged)
# -----------------------------------------------------
def collate_fn(batch):
    input_ids_list = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask_list = [torch.tensor(item['attention_mask']) for item in batch]
    labels_list = [torch.tensor(item['labels']) for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

# -----------------------------------------------------
# 7. Prepare DataLoader and optimizer
# -----------------------------------------------------
train_dataset = CustomGenDataset(datas, tokenizer, prompt)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn
)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# -----------------------------------------------------
# 8. Training loop
# -----------------------------------------------------
print("--- Starting training ---")
model.train()

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

print("--- Training completed ---")

# -----------------------------------------------------
# 9. Save model (path modified)
# -----------------------------------------------------
# Modification 4: Dynamically generate save path based on model name to avoid overwriting old models
safe_model_name = MODEL_NAME.replace('/', '_') 
output_dir = f"IRAPE/Base_gen/models/{safe_model_name}_finetuned"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to: {output_dir}")

# -----------------------------------------------------
# 10. Test model (unchanged)
# -----------------------------------------------------
print("\n--- Testing model (applying LogitsProcessor) ---")
model.eval()

number_only_processor = NumberOnlyLogitsProcessor(tokenizer)
logits_processor_list = LogitsProcessorList([number_only_processor])

test_data = datas[1]
test_input_text = prompt + test_data["input"]
expected_output = str(test_data["number"])

print(f"Test input:\n{test_input_text}")
print(f"Expected output: {expected_output}")

inputs = tokenizer(test_input_text, return_tensors='pt').to(device)

with torch.no_grad():
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        logits_processor=logits_processor_list
    )

generated_token_ids = output_sequences[0][inputs['input_ids'].shape[1]:]
generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

print(f"Model generated: {generated_text.strip()}")
print("--- Test completed ---")