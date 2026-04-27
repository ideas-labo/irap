import pickle
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Import your custom modules
from expriments.metrix import p_to_p_dis, chebyshev_dis, RMSE_dis, area_dis
from sota.RAG.native_rag import NativeRag
from sota.RAG.hybrid_retrieval_rag import HybridRetrievalRag
from sota.RAG.ASTUTE_rag import AstuteRag

def save_data_to_jsonl(results, filepath):
    """Save results to jsonl file"""
    # Create target folder (if it doesn't exist)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            # Convert result to dictionary format
            result_dict = {
                'index': result[0],  # Sequence number
                'p2p_distance': result[1],
                'chebyshev_distance': result[2], 
                'rmse': result[3],
                'area_difference': result[4]
            }
            f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

def save_data_to_json(data, filename):
    """Saves a Python object (list or dict) to a JSON file."""
    try:
        # Create target folder (if it doesn't exist)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 'w' mode for writing
        # indent=4 makes the output readable (pretty-print)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ Data successfully saved to '{filename}'")
        print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")
        
    except IOError as e:
        print(f"❌ Error saving file: {e}")

# Define list of available model names (for reference or Argparse use)
AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V3.1-Terminus", 
    "Qwen/Qwen3-Coder-480B-A35B-Instruct", 
    "moonshotai/Kimi-K2-Instruct-0905"
]

# ================= Configuration Parameters =================
model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
API_KEY = "sk-xxvbjcjvfmyhiuhkycltltsaxvtjducvlslnxdxjspbbflru"
database_path = 'dataset/pkl/final_synthetic_dataset.pkl'
source_pre = 'dataset/pkl/'
files = ['Promise', 'Functional-Quality', 'PURE', 'Shaukat_et_al']
k = 5

# Define list of all methods to be tested
METHODS = ["native_rag", "hybrid_retrieval_rag", "ASTUTE_rag"]

# ================= Main Execution Logic =================

if __name__ == "__main__":
    print(f"Starting experiments with model: {model_name}")
    
    # First layer loop: iterate over all RAG methods
    for method in METHODS:
        print(f"\n{'#'*30}")
        print(f"Running Method: {method}")
        print(f"{'#'*30}")

        # 1. Instantiate RAG object based on current method
        if method == "native_rag":
            rag = NativeRag(api_key=API_KEY, model_name=model_name, database_path=database_path)
        elif method == "hybrid_retrieval_rag":
            rag = HybridRetrievalRag(api_key=API_KEY, model_name=model_name, database_path=database_path)
        elif method == "ASTUTE_rag":
            rag = AstuteRag(api_key=API_KEY, model_name=model_name, database_path=database_path)
        else:
            print(f"Unknown method: {method}, skipping.")
            continue

        # Second layer loop: iterate over all dataset files
        for filename in files:
            data_path = f"{source_pre}final_{filename}.pkl"
            
            # Load dataset
            try:
                with open(data_path, 'rb') as f:
                    datas = pickle.load(f)
            except FileNotFoundError:
                print(f"Error: File {data_path} not found. Skipping {filename}.")
                continue

            results = []
            
            # Third layer loop: process single data entry (with progress bar)
            progress_bar = tqdm(
                enumerate(datas), 
                total=len(datas), 
                desc=f"[{method}] {filename}", 
                unit="item"
            )
            
            for i, data in progress_bar:
                try:
                    # Generate answer using current RAG method
                    list1 = rag.gen_ans(data, k) 
                    list2 = data["prefer_label"]
                    
                    # Calculate four difference metrics
                    p2_p = p_to_p_dis(list1, list2)      # Point-to-point distance
                    cheby = chebyshev_dis(list1, list2)  # Maximum deviation (Chebyshev distance)
                    rms = RMSE_dis(list1, list2)         # Root mean square error
                    area = area_dis(list1, list2)        # Integral area difference
                    
                    results.append([
                        i + 1,
                        p2_p,
                        cheby,
                        rms,
                        area
                    ])
                except Exception as e:
                    # Record error and continue to avoid script termination due to single API call failure
                    print(f"\nError at {filename} index {i}: {e}")
                    continue

            # Save experimental results of current dataset under current method
            save_path = f'expriments/main_exp/res/tmp/{filename}/{method}_k={k}.jsonl'
            save_data_to_jsonl(results, save_path)
            
            print(f"\n[Done] Saved {len(results)} results to {save_path}")

DIR_PRE = "expriments/main_exp/res/tmp/"

DIRS = [
    "Functional-Quality",
    "Promise",
    "PURE",
    "Shaukat_et_al"
]

METHODS = [
    "RAG",
    "hybridRAG",
    "ASTUTE RAG",
]

FILES = [
    "native_rag_k=1.jsonl",
    "hybrid_retrieval_rag_k=1.jsonl",
    "ASTUTE_rag_k=5.jsonl",
]

# --- Configuration ---
metric_keys = {
    "p2p_distance": "P2P-Dist",
    "chebyshev_distance": "Chebyshev",
    "rmse": "RMSE",
    "area_difference": "Area-Diff"
}

final_res = []
for dir in DIRS:
    res = {}
    res["dataset"] = dir
    res["result"] = {}
    for file in FILES:
        method = METHODS[i]
        
        FILE_NAME = DIR_PRE + dir + '/' + file

        # List to store all values for each metric
        metric_values = defaultdict(list)

        # 1. Read JSONL data from file
        try:
            print(f"--- Reading data from {FILE_NAME} ---")
            with open(FILE_NAME, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        for json_key in metric_keys.keys():
                            if json_key in data:
                                metric_values[json_key].append(data[json_key])
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {line}")
                        continue
        except FileNotFoundError:
            print(f"Error: The file '{FILE_NAME}' was not found. Please create it and add the JSONL data.")
            exit()

        # 2. Calculate mean and standard deviation
        results = {}
        for json_key, output_key in metric_keys.items():
            values = metric_values[json_key]
            
            if values:
                # Use numpy for calculations
                avg = np.mean(values)
                std = np.std(values)
                
                # Format output string: avg(std), keep 4 decimal places
                formatted_avg = f"{avg:.4f}"
                formatted_std = f"{std:.4f}"
                
                results[output_key] = f"{formatted_avg} ({formatted_std})"
            else:
                results[output_key] = "N/A"
        # 3. Print final results
        print("\n--- Statistical Results ---")
        print(json.dumps(results, indent=4))
        res["result"][method] = results
    final_res.append(res)
save_path = "expriments/main_exp/res/RAG.json"
save_data_to_json(final_res, save_path)