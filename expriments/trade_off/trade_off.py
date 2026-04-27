import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import math

from expriments.metrix import *
from IRAPE.Anal_reason.diff_trans import Transfer
from IRAPE.Query.interaction_simulate import query

source_pre = 'dataset/pkl/'
files = ['Promise', 'Functional-Quality', 'PURE', 'Shaukat_et_al'] 
database_path = 'dataset/pkl/final_synthetic_dataset.pkl'
transfer = Transfer(database_path)

def save_data_to_json(data, filename):
    """Saves a Python object (list or dict) to a JSON file."""
    try:
        # 'w' mode for writing
        # indent=4 for readable formatting of objects
        # separators=(',', ': ') removes extra whitespace after separators,
        # which helps keep lists compact without newlines for each item
        # Note: This will make lists compact within the indented structure
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))
        
        print(f"✅ Data successfully saved to '{filename}'")
        print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")
        
    except IOError as e:
        print(f"❌ Error saving file: {e}")

# Calculate statistical information for each result set
def calculate_statistics(results_list, name):
    if len(results_list) == 0:
        return
    
    p2p_distances = [r[1] for r in results_list]
    chebyshev_distances = [r[2] for r in results_list]
    rmse_values = [r[3] for r in results_list]
    area_differences = [r[4] for r in results_list]
    
    metrics = {
        'p2p_distance': p2p_distances,
        'chebyshev_distance': chebyshev_distances,
        'rmse': rmse_values,
        'area_difference': area_differences
    }
    
    print(f"\n{name} Statistics:")
    print(f"Total data pairs: {len(results_list)}")
    
    stats = []
    for metric_name, values in metrics.items():
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        # Calculate variance
        variance_val = sum((x - mean_val) ** 2 for x in values) / len(values)
        
        print(f"{metric_name} - Mean: {mean_val:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}, Variance: {variance_val:.3f}")
        stats.append((round(mean_val, 3), round(variance_val, 3)))
    
    return stats

# List to store results
results = {
    'p2p_distance' : {},
    'chebyshev_distance' : {},
    'rmse': {},
    'area_difference': {}
}

for filename in files:

    data_path = source_pre + 'final_' + filename + '.pkl'
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)

    results['p2p_distance'][filename] = []
    results['chebyshev_distance'][filename] = []
    results['rmse'][filename] = []
    results['area_difference'][filename] = []

    # # List to store results
    # results = {
    #     'p2p_distance' : [],
    #     'chebyshev_distance' : [],
    #     'rmse': [],
    #     'area_difference': [] 
    # }
    for k in range(1, 10):
        results3 = []
        for i, data in enumerate(datas):
            list2 = data["prefer_label"]
            _, list1 = query(transfer.trans(data), k) # Obtained by further adjusting recommendation results, with k query times
            
            # Calculate four types of differences
            p2p_distance = p_to_p_dis(list1, list2)  # Point-to-point distance difference
            chebyshev_distance = chebyshev_dis(list1, list2)  # Maximum deviation (Chebyshev distance)
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

        

        stats3 = calculate_statistics(results3, "Results3")
        results['p2p_distance'][filename].append((k, stats3[0]))
        results['chebyshev_distance'][filename].append((k, stats3[1]))
        results['rmse'][filename].append((k, stats3[2]))
        results['area_difference'][filename].append((k, stats3[3]))

save_data_to_json(results, 'expriments/trade_off/res/trade_off.json')