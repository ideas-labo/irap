import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Use mirror site to speed up download
import torch
from torch.utils.data import Dataset, DataLoader, random_split # Import random_split for validation set splitting
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW # Import AdamW from torch.optim
import numpy as np
import torch.nn.functional as F

class RetrievalClassificationDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length=256):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Group labels by class
        self.class_labels = {}
        for label in labels:
            class_id = label['class']
            if class_id not in self.class_labels:
                self.class_labels[class_id] = []
            self.class_labels[class_id].append(label['label'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        true_class = item['class']
        
        return {
            'input_text': input_text,
            'true_class': true_class
        }

class RetrievalBERTModel(torch.nn.Module):
    def __init__(self, model_name='roberta-base', num_classes=3):
        super(RetrievalBERTModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        return pooled_output

# MODIFIED FUNCTION
def get_label_embeddings(model, tokenizer, labels, device):
    """
    Get embedding representations for all labels.
    """
    label_embeddings = {} # {class_id: [list of embeddings]}
    
    # Remove with torch.no_grad():
    for label_item in labels:
        class_id = label_item['class']
        label_text = label_item['label']
        
        inputs = tokenizer(
            label_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        # Gradients will flow here
        label_embedding = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        if class_id not in label_embeddings:
            label_embeddings[class_id] = []
            
        # Store embedding vector for each label
        # Remove .cpu()
        # label_embedding is [1, hidden_size], .squeeze(0) converts to [hidden_size]
        label_embeddings[class_id].append(label_embedding.squeeze(0)) 
    
    # Convert embedding list of each class to tensor
    for class_id in label_embeddings:
        # Now label_embeddings[class_id] is [num_labels, hidden_size]
        label_embeddings[class_id] = torch.stack(label_embeddings[class_id]) 
    
    return label_embeddings

def retrieval_classify(model, input_embedding, label_embeddings):
    """Retrieval-based classification"""
    similarities = {}

    for class_id, class_label_embeddings in label_embeddings.items():
        # input_embedding: [1, hidden_size] or [hidden_size]
        # class_label_embeddings: [num_labels_in_class, hidden_size]
        
        # Ensure input_embedding has shape [1, hidden_size]
        if len(input_embedding.shape) == 1:
            input_emb_expanded = input_embedding.unsqueeze(0) # [1, hidden_size]
        else:
            input_emb_expanded = input_embedding # [1, hidden_size] or [batch_size, hidden_size]
        
        # Calculate similarity between input and all labels of this class
        # class_label_embeddings: [num_labels_in_class, hidden_size]
        # input_emb_expanded: [1, hidden_size] -> broadcast -> [num_labels_in_class, hidden_size]
        # cosine_similarity requires the last dimension to be the feature dimension, so transposition is needed
        # Calculate cosine similarity, result is [num_labels_in_class]
        class_similarities = torch.cosine_similarity(
            input_emb_expanded, # [1, hidden_size] 
            class_label_embeddings, # [num_labels_in_class, hidden_size]
            dim=1 # Calculate similarity on the last dimension (hidden_size)
        )
        
        # Take the maximum similarity as the score for this class
        max_similarity = torch.max(class_similarities).item()
        similarities[class_id] = max_similarity
    
    # Return the class with the highest similarity
    predicted_class = max(similarities, key=similarities.get)
    return predicted_class, similarities


def contrastive_loss(input_embeddings, true_classes, label_embeddings, margin=0.5):
    """
    InfoNCE Loss implementation (All-Pairs Contrastive Loss).

    Args:
        input_embeddings: Tensor of shape [batch_size, hidden_size]
        true_classes: Tensor of shape [batch_size] containing ground truth class IDs
        label_embeddings: Dict mapping class_id to Tensors of shape [num_labels_per_class, hidden_size]
        margin: Kept for signature compatibility, but unused in InfoNCE logic.
    """
    # Temperature hyperparameter (tau) as per your formula
    # Common values are 0.07 or 0.1.
    tau = 0.07
    batch_size = input_embeddings.size(0)
    total_loss = 0.0
    
    # 1. Flatten all label embeddings into a single matrix for the denominator
    # all_label_embeddings shape: [total_num_labels (e.g., 30), hidden_size]
    all_label_embeddings = torch.cat(list(label_embeddings.values()), dim=0)
    
    # 2. Build a map to locate specific class indices within the flattened matrix
    class_indices_map = {}
    current_idx = 0
    for class_id, embeds in label_embeddings.items():
        count = embeds.size(0)
        class_indices_map[class_id] = (current_idx, current_idx + count)
        current_idx += count

    # 3. Process each sample in the batch
    for i in range(batch_size):
        input_emb = input_embeddings[i].unsqueeze(0) # [1, hidden_size]
        true_class = true_classes[i].item()
        
        # Calculate cosine similarity between input and ALL labels (A)
        # logits shape: [total_num_labels]
        logits = F.cosine_similarity(input_emb, all_label_embeddings, dim=1) / tau
        
        # Calculate the denominator using LogSumExp for numerical stability
        # log_sum_exp_all corresponds to log(sum(exp(sim(s_i, a)/tau)))
        log_sum_exp_all = torch.logsumexp(logits, dim=0)
        
        # Identify indices of positive samples (P_i)
        start_idx, end_idx = class_indices_map[true_class]
        pos_logits = logits[start_idx:end_idx] # [num_pos_labels] (e.g., 10)
        
        # Calculate -log(prob) for each positive sample
        # -log(exp(pos)/sum(exp_all)) = -(pos - log(sum(exp_all)))
        per_positive_loss = -(pos_logits - log_sum_exp_all)
        
        # Average loss over all positive samples for this input (1/|P_i|)
        sample_loss = per_positive_loss.mean()
        
        total_loss += sample_loss
    
    # Return mean loss over the batch
    return total_loss / batch_size

# Load data
data_path = "IRAPE/Base_gen/dataset/train.jsonl"
label_path = "IRAPE/Base_gen/dataset/label.jsonl"

labels = []
datas = []

with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip()) # Parse single-line JSON
        datas.append(data)

with open(label_path, 'r', encoding='utf-8') as f:
    for line in f:
        label = json.loads(line.strip()) # Parse single-line JSON
        labels.append(label)

# Initialize model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download model and tokenizer using mirror site
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RetrievalBERTModel().to(device)

# *** EARLY STOPPING SETUP ***
# --- Add validation set ---
# Split part of the training data as validation set
val_ratio = 0.1  # Validation set ratio
val_size = int(len(datas) * val_ratio)
train_size = len(datas) - val_size

# Split dataset using random_split
train_dataset_full = RetrievalClassificationDataset(datas, labels, tokenizer)
train_dataset, val_dataset = random_split(
    train_dataset_full, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42) # For reproducibility
)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False) # No need to shuffle validation set

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# *** EARLY STOPPING VARIABLES ***
patience = 3  # Number of epochs to tolerate if validation loss does not improve
best_val_loss = float('inf')  # Record the best validation loss
patience_counter = 0  # Record consecutive epochs without improvement
best_model_state = None # Record the best model state

num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train() # Ensure model is in training mode
    total_train_loss = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        # Get label embeddings (may be updated during training)
        # This function now retains gradients
        label_embeddings = get_label_embeddings(model, tokenizer, labels, device)
        
        # Process input texts
        input_texts = batch['input_text']
        true_classes = batch['true_class'].to(device)
        
        # Encode input texts
        inputs = tokenizer(
            input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        input_embeddings = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Calculate contrastive loss
        loss = contrastive_loss(
            input_embeddings, 
            true_classes, 
            label_embeddings
        )
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
        if batch_idx % 10 == 0: # Print loss every 10 batches
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Train Loss: {loss.item():.4f}')
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}')

    # Validation phase
    model.eval() # Switch to evaluation mode
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad(): # Do not calculate gradients during validation
        # Calculate all label embeddings at once with no_grad before validation
        label_embeddings = get_label_embeddings(model, tokenizer, labels, device)
        
        for val_batch in val_dataloader:
            input_texts = val_batch['input_text']
            true_classes = val_batch['true_class'].to(device)

            # Encode input texts
            inputs = tokenizer(
                input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            input_embeddings = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # Calculate validation contrastive loss
            val_loss = contrastive_loss(
                input_embeddings, 
                true_classes, 
                label_embeddings
            )
            
            total_val_loss += val_loss.item()
            num_val_batches += 1
    
    avg_val_loss = total_val_loss / num_val_batches
    print(f'Epoch {epoch+1}/{num_epochs}, Average Val Loss: {avg_val_loss:.4f}')

    # *** EARLY STOPPING LOGIC ***
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0 # Reset counter
        # Save current best model state
        best_model_state = model.state_dict().copy() 
        print(f"  -> New best validation loss! Saving model state.")
    else:
        patience_counter += 1
        print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
        # Load best model weights
        model.load_state_dict(best_model_state)
        break # Exit training loop

# *** EVALUATION AFTER TRAINING ***
# Calculate all label embeddings at once with no_grad before evaluation
with torch.no_grad():
    label_embeddings = get_label_embeddings(model, tokenizer, labels, device)

# Evaluation loop also needs to disable gradients
print("\nStarting final evaluation on training set...")
model.eval() # Ensure model is in evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for item in datas:
        input_text = item['input']
        true_class = item['class']
        
        # Get embedding of input text
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        input_embedding = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Perform retrieval classification
        predicted_class, similarities = retrieval_classify(
            model, input_embedding, label_embeddings
        )
        
        if predicted_class == true_class:
            correct += 1
        total += 1

accuracy = correct / total
print(f'Final Training Set Accuracy: {accuracy:.4f}')

# Save trained model (using best model state)
model_save_path = "IRAPE/Base_gen/models/roberta/"
model.roberta.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

print(f"Training completed. Best validation loss: {best_val_loss:.4f}")