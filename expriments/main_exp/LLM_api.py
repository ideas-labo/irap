import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import os  # Added missing import for os module

from expriments.metrix import *
from sota.LLM_api.LLM_api import LLM_api

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

# Define list of available model names
AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V3.1-Terminus", 
    "Qwen/Qwen3-Coder-480B-A35B-Instruct", 
    "moonshotai/Kimi-K2-Instruct-0905",
    "gpt-5-mini",
    "llama-4-maverick-17b-128e-instruct"
]

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM comparison with specified model')
    parser.add_argument('--model', 
                       type=str, 
                       default="ByteDance-Seed/Seed-OSS-36B-Instruct",
                       choices=AVAILABLE_MODELS,
                       help='Model name to use for LLM_api (default: ByteDance-Seed/Seed-OSS-36B-Instruct)')
    return parser.parse_args()


# Parse command line arguments
args = parse_args()
model_name = args.model

print(f"Using model: {model_name}")

source_pre = 'dataset/pkl/'
files = ['Promise', 'Functional-Quality', 'PURE', 'Shaukat_et_al']
# files = ['PURE', 'Shaukat_et_al']

for filename in files:

    data_path = source_pre + 'final_' + filename + '.pkl'
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)


    # List to store results
    results = []

    # Add progress bar to the first main calculation loop
    for i, data in tqdm(enumerate(datas), 
                       total=len(datas), 
                       desc=f"Processing Results1 - {filename}", 
                       unit="data"):
        list1 = data["label"]
        list2 = data["prefer_label"]
        
        # Calculate four types of differences
        p2p_distance = p_to_p_dis(list1, list2)  # Point-to-point distance difference
        chebyshev_distance = chebyshev_dis(list1, list2)  # Maximum deviation (Chebyshev distance)
        rmse = RMSE_dis(list1, list2)  # Root mean square error
        area_difference = area_dis(list1, list2)  # Integral area difference
        
        # Add results to the result list
        results.append([
            i + 1,  # Sequence number
            p2p_distance,
            chebyshev_distance,
            rmse,
            area_difference
        ])


    # List to store results
    results = []

    # Add progress bar to the second main calculation loop
    for i, data in tqdm(enumerate(datas), 
                       total=len(datas), 
                       desc=f"Processing Results3 - {filename}", 
                       unit="data"):
        list1 = LLM_api(data, model_name=model_name)  # Use model name passed from command line arguments
        list2 = data["prefer_label"]
        
        # Calculate four types of differences
        p2p_distance = p_to_p_dis(list1, list2)  # Point-to-point distance difference
        chebyshev_distance = chebyshev_dis(list1, list2)  # Maximum deviation (Chebyshev distance)
        rmse = RMSE_dis(list1, list2)  # Root mean square error
        area_difference = area_dis(list1, list2)  # Integral area difference
        
        # Add results to the result list
        results.append([
            i + 1,  # Sequence number
            p2p_distance,
            chebyshev_distance,
            rmse,
            area_difference
        ])

    # Save experimental results to jsonl file
    # Modify the following paths according to your actual path requirements
    model_name_safe = model_name.replace("/", "_")
    save_data_to_jsonl(results, f'expriments/main_exp/res/tmp/{filename}/{model_name_safe}.jsonl')
    print(f"Results saved for {filename} using model {model_name}")

DIR_PRE = "expriments/main_exp/res/tmp/"

DIRS = [
    "Functional-Quality",
    "Promise",
    "PURE",
    "Shaukat_et_al"
]

METHODS = [
    "DeepSeek-V3",
    "Qwen3-Coder",
    "gpt-5-mini",
    "llama-4"
]

FILES = [
    "deepseek-ai_DeepSeek-V3.1-Terminus.jsonl",
    "Qwen_Qwen3-Coder-480B-A35B-Instruct.jsonl",
    "gpt-5-mini.jsonl",
    "llama-4-maverick-17b-128e-instruct.jsonl",
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
save_path = "expriments/main_exp/res/LLM_api.json"
save_data_to_json(final_res, save_path)