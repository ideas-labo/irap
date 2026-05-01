# Set up mirror
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import RobertaTokenizer, RobertaModel
from torch.nn import Module

class BaseGenerator:
    NUM_EXTRA_MODEL_PATH = "IRAPE/Base_gen/models/gpt2" 
    CLASSIFY_MODEL_PATH = "IRAPE/Base_gen/models/roberta"
    LABEL_PATH = "IRAPE/Base_gen/dataset/label.jsonl"
    
    # Prompt for extracting numerical thresholds
    PROMPT_TEMPLATE = "Instruction: Extract the numerical performance threshold from the requirement statement below. Your output MUST be ONLY the bare number, nothing else (no units, no text, no punctuation, no surrounding quotes). Requirement:\n"

    # Automatically select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Static member variables for storing models and tokenizers
    num_extra_model = None
    num_extra_tokenizer = None
    classify_model = None
    classify_tokenizer = None
    label_embeddings = None
    
    # -----------------------------------------------------
    # Define model class
    # -----------------------------------------------------
    class RetrievalBERTModel(Module):
        def __init__(self, model_name_or_path='roberta-base'):
            """
            During initialization, load the pre-trained roberta model from the specified path.
            """
            super(BaseGenerator.RetrievalBERTModel, self).__init__()
            # This will load your fine-tuned roberta weights from MODEL_PATH
            self.roberta = RobertaModel.from_pretrained(model_name_or_path)
            # Dropout is automatically disabled in model.eval() mode
            self.dropout = torch.nn.Dropout(0.1) 
            
        def forward(self, input_ids, attention_mask):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output # [batch_size, hidden_size]
            pooled_output = self.dropout(pooled_output)
            return pooled_output

    @classmethod
    def get_label_embeddings(cls):
        """
        Load labels and calculate embeddings for all labels.
        """
        # Load labels
        labels = []
        try:
            with open(cls.LABEL_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    labels.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Error: Label file '{cls.LABEL_PATH}' not found.")
            return None
        print(f"Successfully loaded {len(labels)} labels from '{cls.LABEL_PATH}'.")
        
        print("Precomputing embeddings for all labels...")
        cls.classify_model.eval() # Ensure the model is in evaluation mode
        label_embeddings = {}
        
        # Disable gradient calculation to save memory and speed up
        with torch.no_grad():
            for label_item in labels:
                class_id = label_item['class']
                label_text = label_item['label']
                
                inputs = cls.classify_tokenizer(
                    label_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(cls.device)
                
                # Model forward pass
                label_embedding = cls.classify_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                ) # [1, hidden_size]
                
                if class_id not in label_embeddings:
                    label_embeddings[class_id] = []
                    
                # .squeeze(0) converts [1, hidden_size] to [hidden_size]
                label_embeddings[class_id].append(label_embedding.squeeze(0))
        
            # Stack the embedding list of each category into a tensor
            for class_id in label_embeddings:
                label_embeddings[class_id] = torch.stack(label_embeddings[class_id])
                
        print("Label embedding calculation completed.")
        return label_embeddings

    @classmethod
    def initialize(cls):
        """Initialize all models, tokenizers, and label embeddings"""
        print(f"--- Preparing to load all models ---")
        print(f"Using device: {cls.device}")

        # 1. Load number extraction model and tokenizer
        print(f"Loading number extraction model and tokenizer from '{cls.NUM_EXTRA_MODEL_PATH}'...")
        
        try:
            # Load from local file system
            cls.num_extra_tokenizer = GPT2Tokenizer.from_pretrained(cls.NUM_EXTRA_MODEL_PATH)
            cls.num_extra_model = GPT2LMHeadModel.from_pretrained(cls.NUM_EXTRA_MODEL_PATH)

            # Move model to specified device
            cls.num_extra_model.to(cls.device)
            
            # !! Critical: Set model to evaluation mode !!
            # This disables layers like Dropout that are only used during training
            cls.num_extra_model.eval() 
            
            print("Number extraction model loaded successfully.")

        except OSError:
            print(f"Error: Number extraction model files not found. Please ensure the model is saved in '{cls.NUM_EXTRA_MODEL_PATH}'.")
            # In practical applications, you may want to exit the script here
            # exit()
            # To prevent program crash, we use the base 'gpt2' model for demonstration
            print(f"Warning: Using untuned 'gpt2' model for demonstration.")
            cls.num_extra_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            cls.num_extra_model = GPT2LMHeadModel.from_pretrained('gpt2')
            cls.num_extra_tokenizer.pad_token = cls.num_extra_tokenizer.eos_token
            cls.num_extra_model.config.pad_token_id = cls.num_extra_model.config.eos_token_id
            cls.num_extra_model.to(cls.device)
            cls.num_extra_model.eval()

        # 2. Load classification model and tokenizer
        print(f"Loading classification model and tokenizer from '{cls.CLASSIFY_MODEL_PATH}'...")
        
        try:
            # 1. Load tokenizer
            cls.classify_tokenizer = RobertaTokenizer.from_pretrained(cls.CLASSIFY_MODEL_PATH)
            
            # 2. Load model
            #    We pass MODEL_PATH, and RetrievalBERTModel will internally call
            #    RobertaModel.from_pretrained(MODEL_PATH) to load weights
            cls.classify_model = cls.RetrievalBERTModel(model_name_or_path=cls.CLASSIFY_MODEL_PATH).to(cls.device)
            cls.classify_model.eval() # !! Critical: Set to evaluation mode !!
            
            # 3. Load labels and calculate embeddings
            cls.label_embeddings = cls.get_label_embeddings()
            if cls.label_embeddings is None:
                raise FileNotFoundError("Failed to load labels or calculate embeddings, inference cannot proceed.")
                
            print(f"--- Classification model '{cls.CLASSIFY_MODEL_PATH}' loaded successfully and ready ---")

        except Exception as e:
            print(f"\n--- Error: Failed to load classification model or labels ---")
            print(f"Please ensure '{cls.CLASSIFY_MODEL_PATH}' contains model and tokenizer files,")
            print(f"and '{cls.LABEL_PATH}' path is correct.")
            print(f"Error details: {e}")
            # In practical applications, you may want to exit here
            # exit()
            cls.classify_model = None # Set to None for later checking
            cls.label_embeddings = None

    @classmethod
    def extract_threshold(cls, input_statement: str) -> str:
        """
        Receive a requirement statement, call the fine-tuned GPT-2 model, and return the extracted numeric string.

        Parameters:
        input_statement (str): Original requirement text, e.g., "The system... 500 users."

        Returns:
        str: Numeric string generated by the model, e.g., "500.0"
        """
        # Ensure the model is initialized
        if cls.num_extra_model is None or cls.num_extra_tokenizer is None:
            cls.initialize()
        
        # 1. Construct complete input
        # Must use exactly the same format as during training
        full_input_text = cls.PROMPT_TEMPLATE + input_statement

        # 2. Encode/tokenize
        # return_tensors='pt' converts output to PyTorch Tensors
        inputs = cls.num_extra_tokenizer(full_input_text, return_tensors='pt').to(cls.device)

        # 3. Perform inference using model.generate()
        # We do not calculate gradients to save memory and speed up
        with torch.no_grad():
            output_sequences = cls.num_extra_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10, # Limit generation length, 10 tokens are enough for "500.0"
                pad_token_id=cls.num_extra_tokenizer.eos_token_id,
                eos_token_id=cls.num_extra_tokenizer.eos_token_id,
                do_sample=False, # Use greedy search to ensure consistent results
                num_beams=1      # Same as above
            )

        # 4. Decode output
        
        # inputs['input_ids'].shape[1] is the token length of the input text
        # We only decode the newly generated part after this
        input_token_len = inputs['input_ids'].shape[1]
        generated_token_ids = output_sequences[0][input_token_len:]
        
        # skip_special_tokens=True automatically removes special tokens like <|endoftext|>
        generated_text = cls.num_extra_tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # 5. Return cleaned result
        return generated_text.strip()

    @classmethod
    def retrieval_classify(cls, input_text: str):
        """
        Receive a requirement statement, call the fine-tuned RoBERTa model, and return the predicted category.

        Parameters:
        input_text (str): Original requirement text.

        Returns:
        int: Predicted category ID.
        """
        if cls.classify_model is None or cls.classify_tokenizer is None or cls.label_embeddings is None:
            cls.initialize()
        
        if cls.classify_model is None or cls.label_embeddings is None:
            print("Error: Model not loaded successfully, cannot perform classification.")
            return None

        # Disable gradients
        with torch.no_grad():
            # 1. Encode input text
            inputs = cls.classify_tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            ).to(cls.device)
            
            # 2. Get embedding of input text
            input_embedding = cls.classify_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            ) # [1, hidden_size]
            
            # 3. Calculate similarities
            similarities = {}
            for class_id, class_label_embeddings in cls.label_embeddings.items():
                
                # Calculate cosine similarity
                class_similarities = torch.cosine_similarity(
                    input_embedding,          # [1, hidden_size]
                    class_label_embeddings,   # [num_labels_in_class, hidden_size]
                    dim=1                     # Calculate on hidden_size dimension
                ) # [num_labels_in_class]
                
                # Take the maximum similarity as the score for this category
                # .item() converts Tensor to python float
                max_similarity = torch.max(class_similarities).item()
                similarities[class_id] = max_similarity
            
            # Return the category with the highest similarity
            predicted_class = max(similarities, key=similarities.get)
            
        return predicted_class

    @classmethod
    def safe_convert_to_number(cls, s):
        # 1. 预处理：去掉空格和奇怪的下划线
        s = str(s).strip().replace('_', '')
        
        try:
            # 尝试直接转换（效率最高）
            return float(s)
        except ValueError:
            # 2. 正则提取：匹配第一个出现的数字（包括整数、小数、负数）
            # 匹配规则：可选负号 + 数字 + 可选的小数点和数字
            match = re.search(r"[-+]?\d*\.?\d+", s)
            
            if match:
                extracted = match.group()
                try:
                    # 针对 ".10.0" 这种情况再次处理，只保留第一个小数点
                    if extracted.count('.') > 1:
                        parts = extracted.split('.')
                        extracted = f"{parts[0]}.{parts[1]}"
                    
                    return float(extracted)
                except ValueError:
                    pass
            
            # 3. 兜底方案：打印更有意义的错误并返回 0.0
            print(f"警告: 无法从字符串 '{s}' 中解析数字，已设为默认值 0.0")
            return 0.0

    @classmethod
    def base_gen(cls, sentence: str):
        num = cls.extract_threshold(sentence) # Convert string to float
        num = cls.safe_convert_to_number(num)
        tp = cls.retrieval_classify(sentence)
        res = []
        if tp == 1:
            res.append([num * 0.9, 0.0])
            res.append([num, 1.0])
        elif tp == 2:
            res.append([num, 1.0])
            res.append([num * 1.1, 0.0])
        else:
            res.append([num * 0.9, 0.0])
            res.append([num, 1.0])
            res.append([num * 1.1, 0.0])

        return res

# -----------------------------------------------------
# 6. Example calls
# -----------------------------------------------------
if __name__ == "__main__":
    print("\n--- Starting test of number extraction function ---")

    # Example 1
    statement1 = "ReqView Desktop shall start in less than 10s."
    output1 = BaseGenerator.extract_threshold(statement1)
    print(f"Input: {statement1.strip()}")
    print(f"Model output: {output1}")
    print(f"Expected output: 10.0")

    print("---")

    # Example 2
    statement2 = "The system must handle 500 concurrent users."
    output2 = BaseGenerator.extract_threshold(statement2)
    print(f"Input: {statement2.strip()}")
    print(f"Model output: {output2}")
    print(f"Expected output: 500.0")

    print("---")

    # Example 3 (new data)
    statement3 = "The API response time must be under 20.2 seconds."
    output3 = BaseGenerator.extract_threshold(statement3)
    print(f"Input: {statement3.strip()}")
    print(f"Model output: {output3}")
    print(f"Expected output: 20.2")
    
    print("\n--- Starting test of classification function ---")
    
    # Test classification function
    statement4 = "The API response time must be under 20.5 seconds." 
    statement5 = "The system must handle 500 concurrent users." 
    statement6 = "The system shall maintain a **99.5% accuracy rate** in real-time parking space occupancy detection." 

    # Test 1
    print(f"\nInput: {statement4}")
    pred_class = BaseGenerator.retrieval_classify(statement4)
    print(f"Predicted category: {pred_class}")

    # Test 2
    print(f"\nInput: {statement5}")
    pred_class = BaseGenerator.retrieval_classify(statement5)
    print(f"Predicted category: {pred_class}")

    # Test 3
    print(f"\nInput: {statement6}")
    pred_class = BaseGenerator.retrieval_classify(statement6)
    print(f"Predicted category: {pred_class}")

    print(BaseGenerator.base_gen(statement4))
    print(BaseGenerator.base_gen(statement5))
    print(BaseGenerator.base_gen(statement6))