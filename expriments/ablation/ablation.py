import pickle
from expriments.metrix import *
from IRAPE.Anal_reason.diff_trans import Transfer
from IRAPE.Query.interaction_simulate import query
import json
import os

source_pre = 'dataset/pkl/'
files = ['Promise', 'Functional-Quality', 'PURE', 'Shaukat_et_al']
database_path = 'dataset/pkl/final_synthetic_dataset.pkl'
transfer = Transfer(database_path)

def save_results_to_jsonl(results, filepath):
    """Save results to jsonl file"""
    # Get the directory where the file is located
    directory = os.path.dirname(filepath)
    
    # Recursively create directory if it does not exist
    if directory: # Ensure filepath is not just a filename
        os.makedirs(directory, exist_ok=True)

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

for filename in files:

    data_path = source_pre + 'final_' + filename + '.pkl'
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)


    # List to store results
    results = []

    for i, data in enumerate(datas):
        list1 = data["label"]
        list2 = data["prefer_label"]
        
        # Calculate four types of differences
        p2p_distance = p_to_p_dis(list1, list2)  # Point-to-point distance difference
        chebyshev_distance = chebyshev_dis(list1, list2)  # Maximum deviation
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
    results2 = []

    for i, data in enumerate(datas):
        list1 = transfer.trans(data) # Get recommendation results
        list2 = data["prefer_label"]
        
        # Calculate four types of differences
        p2p_distance = p_to_p_dis(list1, list2)  # Point-to-point distance difference
        chebyshev_distance = chebyshev_dis(list1, list2)  # Maximum deviation
        rmse = RMSE_dis(list1, list2)  # Root mean square error
        area_difference = area_dis(list1, list2)  # Integral area difference
        
        # Add results to the result list
        results2.append([
            i + 1,  # Sequence number
            p2p_distance,
            chebyshev_distance,
            rmse,
            area_difference
        ])

    # List to store results
    results3 = []

    for i, data in enumerate(datas):
        list2 = data["prefer_label"]
        _, list1 = query(data["label"], 5)
        
        # Calculate four types of differences
        p2p_distance = p_to_p_dis(list1, list2)  # Point-to-point distance difference
        chebyshev_distance = chebyshev_dis(list1, list2)  # Maximum deviation
        rmse = RMSE_dis(list1, list2)  # Root mean square error
        area_difference = area_dis(list1, list2)  # Integral area difference
        
        # Add results to the result list
        results3.append([
            i + 1,  # Sequence number
            p2p_distance,
            chebyshev_distance,
            rmse,
            area_difference
        ])

    # List to store results
    results4 = []

    for i, data in enumerate(datas):
        list2 = data["prefer_label"]
        _, list1 = query(transfer.trans(data), 5)
        
        # Calculate four types of differences
        p2p_distance = p_to_p_dis(list1, list2)  # Point-to-point distance difference
        chebyshev_distance = chebyshev_dis(list1, list2)  # Maximum deviation
        rmse = RMSE_dis(list1, list2)  # Root mean square error
        area_difference = area_dis(list1, list2)  # Integral area difference
        
        # Add results to the result list
        results4.append([
            i + 1,  # Sequence number
            p2p_distance,
            chebyshev_distance,
            rmse,
            area_difference
        ])

    # Save experimental results to jsonl files
    # Modify the following paths according to your actual path requirements
    save_results_to_jsonl(results, f'expriments/ablation/res/{filename}/base.jsonl')
    save_results_to_jsonl(results2, f'expriments/ablation/res/{filename}/wo_query.jsonl') 
    save_results_to_jsonl(results3, f'expriments/ablation/res/{filename}/wo_anal.jsonl')
    save_results_to_jsonl(results4, f'expriments/ablation/res/{filename}/XXX.jsonl')
    print(f"Results saved for {filename}")