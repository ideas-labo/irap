import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import BertTokenizer, BertModel
import pickle
import numpy as np
import pickle

from IRAPE.KM import KM

class Retriever :

    def __init__(self, database_path):
        # Load pre-trained model and tokenizer
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval() # Set to evaluation mode
        self.database = self.load_database(database_path)

    def embedding(self, sentence: str) -> np.ndarray:
        """
        Encode a single sentence using the specified BERT model and return its embedding vector.

        Args:
            sentence (str): A single sentence to be encoded.
            model_name (str): HuggingFace model name, default is 'bert-base-uncased'.

        Returns:
            np.ndarray: Embedding vector of the sentence with shape (768,).
                        Returns None if the input sentence is empty or processing fails.
        """
        if not sentence or not isinstance(sentence, str):
            print("The input sentence is empty or not a string type.")
            return None

        try:
            
            # Tokenize and encode a single sentence
            inputs = self.tokenizer(
                sentence, # Input a single sentence
                return_tensors="pt", # Return PyTorch tensors
                padding=True, # Apply padding
                truncation=True, # Apply truncation
                max_length=128 # Maximum length limit
            )

            with torch.no_grad(): # Disable gradient calculation to save memory and speed up
                outputs = self.model(**inputs)

            # Get the embedding vector of the [CLS] token (batch_size=1, sequence_length, hidden_size)
            # Take the vector of the first token ([CLS]) of the first sequence, move to CPU, and convert to numpy
            # outputs.last_hidden_state.shape: [1, seq_len, 768]
            # [:, 0, :] -> [1, 768]
            # [0] -> [768,]
            embedding_vector = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

            return embedding_vector

        except Exception as e:
            print(f"Error processing sentence '{sentence}': {e}")
            return None


    def load_database(self, database_path: str):
        """
        Load historical interaction data from a local .pkl file to a global variable
        
        Args:
            database_path (str): Path to the .pkl file
        """
        try:
            with open(database_path, 'rb') as f:
                database = pickle.load(f)
            print(f"Successfully loaded {len(database)} data entries")
            return database
        except FileNotFoundError:
            print(f"File {database_path} not found")
        except Exception as e:
            print(f"Error loading file: {e}")

    def update_database(self, new_database):
        """Update the database"""
        self.database = new_database

    def save_database(self, database_path : str):
        """Save the current data state to a file"""
        try:
            with open(database_path, 'wb') as f:
                pickle.dump(self.database, f)
            print("Successfully saved the current interaction data state")
        except Exception as e:
            print(f"Error saving file: {e}")


    def cos_score(self, vec1 : np.ndarray, vec2 : np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors and linearly map it to the range [0, 1].

        Args:
            vec1 (np.ndarray): The first vector.
            vec2 (np.ndarray): The second vector.

        Returns:
            float: Cosine similarity between the two vectors, ranging from [0, 1].
                1 indicates perfect positive correlation (same direction), 0 indicates perfect negative correlation (opposite direction),
                and 0.5 indicates orthogonality (no correlation).
        """
        # Calculate the original cosine similarity in the range [-1, 1]
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0 # Similarity between zero vector and any vector is 0 (after mapping)

        cosine_similarity_raw = dot_product / (norm_vec1 * norm_vec2)
        cosine_similarity_raw = np.clip(cosine_similarity_raw, -1.0, 1.0) # Ensure it is within [-1, 1]

        # Linear mapping: map [-1, 1] to [0, 1]
        # Formula: (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        # Here old_min=-1, old_max=1, new_min=0, new_max=1
        # Simplified: (raw_value + 1) / 2
        cosine_similarity_mapped = (cosine_similarity_raw + 1.0) / 2.0

        # Ensure the result is within [0, 1] again (handle possible floating-point errors)
        return np.clip(cosine_similarity_mapped, 0.0, 1.0)
    
    def KM_score(self, points1 : list, points2 : list) -> float:
        score, _ = KM(points1, points2)
        return score
    
    def retrieval(self, obj, k=1):
        """
        Input a JSON object and return the top k most similar JSON objects from the database

        Args:
            obj (Dict[str, Any]): Input JSON object, which should contain 'vector' and 'label' fields.
            k (int): Return the top k most similar objects, default is 1

        Returns:
            List[Dict[str, Any]]: List of the top k most similar JSON objects from the database, sorted in descending order of similarity.
        """
        if not self.database:
            print("Database is empty, cannot perform retrieval.")
            return []

        if k <= 0:
            return []

        query_vector = obj['vector']
        query_label = obj['label']

        if query_vector is None or query_label is None:
            print(f"Input object missing 'vector' or 'label' field: {obj}")
            return []

        # 1. Calculate cos_score list
        cos_scores = []
        for db_item in self.database:
            db_vector = db_item['vector']
            if db_vector is not None and len(obj['label']) == len(db_item['label']): # Ensure the number of points in the initial state is the same
                score = self.cos_score(query_vector, db_vector)
                cos_scores.append(score)
            else:
                # Assign a very low score if the database item has no vector field
                cos_scores.append(0.0)

        # 2. Calculate KM_score list
        km_scores = []
        for db_item in self.database:
            db_label = db_item['label']
            if db_label is not None:
                score = self.KM_score(query_label, db_label)
                km_scores.append(score)
            else:
                # Assign a very low score if the database item has no label field
                km_scores.append(0.0) # Or set to 0 or other values based on the characteristics of the KM function

        # 3. Quantify the two lists (using ranking scores)
        # Using scipy.stats.rankdata is more convenient for ranking, but manual implementation is used here to avoid additional dependencies
        # Higher ranking means higher score (e.g., the most similar item gets a score of N, the least similar gets 1)
        def get_rank_scores(scores_list):
            # Create a list of tuples (score, original index)
            indexed_scores = [(score, i) for i, score in enumerate(scores_list)]
            # Sort by score (ascending)
            sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[0])
            # Create ranking list, the lowest score gets rank 1, the highest score gets rank len(list)
            rank_scores = [0] * len(scores_list)
            for rank, (score, original_idx) in enumerate(sorted_indexed_scores, start=1):
                rank_scores[original_idx] = rank
            return rank_scores

        cos_rank_scores = get_rank_scores(cos_scores)
        km_rank_scores = get_rank_scores(km_scores)

        # 4. Calculate total score (sum of ranking scores)
        # total_scores = [crs + kms for crs, kms in zip(cos_rank_scores, km_rank_scores)]
        total_scores = [crs for crs, kms in zip(cos_rank_scores, km_rank_scores)]

        # 5. Get indices sorted by total score in descending order
        # Use negative sign to achieve descending order
        sorted_indices = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)

        # 6. Return the top k most similar data items
        k = min(k, len(self.database))  # Ensure k does not exceed the database size
        result = [self.database[idx] for idx in sorted_indices[:k]]
        
        return result