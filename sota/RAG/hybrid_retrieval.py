import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import BertTokenizer, BertModel
import pickle
import numpy as np
from typing import List, Dict, Any
import json
import pickle

class HybridRetriever:

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
        """
        if not sentence or not isinstance(sentence, str):
            print("The input sentence is empty or not a string type.")
            return None

        try:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            embedding_vector = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            return embedding_vector

        except Exception as e:
            print(f"Error processing sentence '{sentence}': {e}")
            return None


    def load_database(self, database_path: str):
        """Load historical interaction data from local .pkl file"""
        try:
            with open(database_path, 'rb') as f:
                database = pickle.load(f)
            print(f"Successfully loaded {len(database)} data entries")
            return database
        except FileNotFoundError:
            print(f"File {database_path} not found")
            return []
        except Exception as e:
            print(f"Error loading file: {e}")
            return []

    def update_database(self, new_database):
        """Update the database"""
        self.database = new_database

    def save_database(self, database_path : str):
        """Save current data state to file"""
        try:
            with open(database_path, 'wb') as f:
                pickle.dump(self.database, f)
            print("Successfully saved current interaction data state")
        except Exception as e:
            print(f"Error saving file: {e}")


    def cos_score(self, vec1 : np.ndarray, vec2 : np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors and linearly map it to the range [0, 1].
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        cosine_similarity_raw = dot_product / (norm_vec1 * norm_vec2)
        cosine_similarity_raw = np.clip(cosine_similarity_raw, -1.0, 1.0)

        cosine_similarity_mapped = (cosine_similarity_raw + 1.0) / 2.0
        return np.clip(cosine_similarity_mapped, 0.0, 1.0)
    
    def KM_score(self, points1 : list, points2 : list) -> float:
        """Calculate similarity between two knowledge graph labels (using external KM function)"""
        score, _ = KM(points1, points2)
        return score

    def _keyword_match_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword matching score in text. 
        Score is defined as: number of matched keywords / total number of keywords.
        """
        if not keywords or len(keywords) == 0:
            return 0.0
        
        text_lower = text.lower()
        matched_count = 0
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched_count += 1
                
        score = matched_count / len(keywords)
        return score

    def hybrid_retrieval(self, obj: Dict[str, Any], keywords: List[str], k=1, 
                            weight_cos: float = 0.5, weight_keyword: float = 0.5) -> List[Dict[str, Any]]:
            """
            [Modified Hybrid Retrieval] Combine vector similarity (Cos Score) and keyword matching similarity (Keyword Score),
            calculate total score using **weighted sum of original scores**.

            Args:
                obj (Dict[str, Any]): Input JSON object, should contain 'vector' and 'sentence' fields.
                keywords (List[str]): List for keyword matching.
                k (int): Return top k most similar objects, default is 1.
                weight_cos (float): Weight for vector similarity, default is 0.5.
                weight_keyword (float): Weight for keyword matching similarity, default is 0.5.

            Returns:
                List[Dict[str, Any]]: List of top k most similar JSON objects from database, 
                sorted in descending order of hybrid score.
            """
            if not self.database:
                print("Database is empty, cannot perform retrieval.")
                return []
            if k <= 0:
                return []

            query_vector = obj.get('vector')
            query_sentence = obj.get('sentence', "") 

            if query_vector is None or not query_sentence:
                print(f"Input object missing 'vector' or 'sentence' field. Cannot perform hybrid retrieval: {obj}")
                return []

            # --- 1. Calculate score list and total score list ---
            total_scores = []
            
            # Ensure sum of weights is close to 1 (though not enforced in actual use)
            if abs(weight_cos + weight_keyword) < 1e-6:
                print("Warning: Sum of weights is close to zero, results may be inaccurate.")

            for db_item in self.database:
                db_vector = db_item.get('vector')
                db_sentence = db_item.get('sentence', "") # Get sentence from database
                
                # A. Vector similarity (Cos Score), range [0, 1]
                if db_vector is not None:
                    cos_score = self.cos_score(query_vector, db_vector)
                else:
                    cos_score = 0.0

                # B. Keyword matching score (Keyword Score), range [0, 1]
                if db_sentence:
                    keyword_score = self._keyword_match_score(db_sentence, keywords)
                else:
                    keyword_score = 0.0
                
                # C. Calculate weighted total score: direct sum (Cos Score * W_cos + Keyword Score * W_key)
                # Since both Cos Score and Keyword Score are in [0, 1] range, direct weighted sum is reasonable.
                hybrid_score = (weight_cos * cos_score) + (weight_keyword * keyword_score)
                total_scores.append(hybrid_score)


            # --- 2. Get indices sorted by total score in descending order ---
            if not total_scores:
                return []
                
            # Use negative sign or reverse=True to achieve descending order
            sorted_indices = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)

            # --- 3. Return top k most similar data items ---
            k = min(k, len(self.database))
            result = [self.database[idx] for idx in sorted_indices[:k]]
            
            return result