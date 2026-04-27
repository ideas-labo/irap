# This module encapsulates various utility functions

import numpy as np
from datetime import datetime
from random import randint
from static_data import *
import pandas as pd
import os
from datetime import datetime

# Negative word list
negative_words = open('sota/LQPR/negative_word.txt', 'r', encoding='utf-8').read().split('\n')

# Modal verbs for qualification
qualifier = ["shall", "must", "should", "will", "be able to", "have", "can"]

word_num = {
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

# Split into training and test sets
def random_move(path, path1, path2):
    train_sentences = []
    test_sentences = []
    # Open file
    with open(path, 'r', encoding='ansi') as f:  # Note the encoding format of the txt
        # Read each line of the file
        for line in f:
            sign = randint(0, 2)
            if sign < 2:
                train_sentences.append(line)
            else:
                test_sentences.append(line)

    with open(path1, 'w', encoding='ansi') as file1:
        for line in train_sentences:
            file1.write(line)

    with open(path2, 'w', encoding='ansi') as file2:
        for line in test_sentences:
            file2.write(line)

# Check if a word is a number
def is_num(x):
    return x[0].isdigit()

# Check if two strings are equal
def equal(a, b):
    # Convert to lowercase
    a = a.lower()
    b = b.lower()
    
    # Convert number words to digits
    if a in word_num:
        a = word_num[a]
    if b in word_num:
        b = word_num[b]
    
    if is_num(a) and is_num(b):  # Both are numbers
        return True
    if a in qualifier and b in qualifier:  # Both are modal verbs
        return True
    if a == b:
        return True
    return False

# Get the specific content of the longest common subsequence
def get_lcs_content(sequence, l, r, trans, LCS):
    if l == 0 or r == 0:
        return
    if trans[l][r] == 1:
        get_lcs_content(sequence, l-1, r-1, trans, LCS)
        LCS.append(sequence[l])
    elif trans[l][r] == 2:
        get_lcs_content(sequence, l-1, r, trans, LCS)
    else:
        get_lcs_content(sequence, l, r-1, trans, LCS)

# Get the position indices of each word in the longest common subsequence
def get_lcs_location(sequence, l, r, trans, LCS):
    if l == 0 or r == 0:
        return
    if trans[l][r] == 1:
        get_lcs_location(sequence, l-1, r-1, trans, LCS)
        LCS.append(l)
    elif trans[l][r] == 2:
        get_lcs_location(sequence, l-1, r, trans, LCS)
    else:
        get_lcs_location(sequence, l, r-1, trans, LCS)

# Find the longest common subsequence (LCS) between two sentences
def lcs(list1, list2):
    n = len(list1)
    m = len(list2)
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
    trans = [[0 for _ in range(m+1)] for _ in range(n+1)]

    for i in range(n):
        dp[i][0] = 1
    for i in range(m):
        dp[0][i] = 1
    for i in range(1, n):
        for j in range(1, m):
            if equal(list1[i], list2[j]):
                dp[i][j] = dp[i-1][j-1] + 1
                trans[i][j] = 1
            else:
                if dp[i-1][j] >= dp[i][j-1]:
                    trans[i][j] = 2
                else:
                    trans[i][j] = 3
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    x = 0
    y = 0
    for i in range(n):
        for j in range(m):
            if dp[i][j] > dp[x][y]:
                x = i
                y = j

    LCS = []
    get_lcs_location(list1, x, y, trans, LCS)
    
    return LCS, (dp[n-1][m-1] - 1) / (len(list2) - 1) if len(list2) > 1 else 0

# Split a sentence into a list of words
def get_word(s):
    res = [' ']
    for word in s.split(' '):
        if word == '':
            continue
        if is_num(word):  # Replace all numbers with '100'
            if '%' in word:  # Special handling for numbers with percentage signs
                res.append('100')
                res.append('percent')
            else:
                res.append('100')
            continue
        res.append(word)
    return res

# Apply penalty based on the compactness of the longest common subsequence
def get_punishment_score(loc_delta, len_pattren, score):  # Penalty for long-distance matches
    if loc_delta <= len_pattren:
        k = 1
    else:
        k = len_pattren / loc_delta
    return k * score

# Sorting function
def key_function(result):
    LCS = result[0]
    pattern = result[1]
    if len(LCS):
        loc_delta = LCS[-1] - LCS[0] + 1
    else:
        loc_delta = 0
    return (result[3], get_punishment_score(loc_delta, len(pattern) - 1, result[3]), len(pattern))
    # Sort by LCS score, penalty score, and pattern size

# Find the numeric threshold in a requirement description
def get_number(sequence):
    for word in sequence:
        if word[0].isdigit():
            return word
    return "none"

# Brute-force matching (can be optimized with KMP) whether a sentence contains a phrase
def match(sentence1, sentence2):
    n = len(sentence1)
    m = len(sentence2)
    for i in range(1, m):
        j = 1
        k = i
        while j < n and k < m and equal(sentence1[j], sentence2[k]):
            j += 1
            k += 1
        if j == n:
            return True
    return False

# Detect semantic inversion
def is_passive(text):  # text is a list of words
    text = get_word(text)
    for keyword in negative_words:
        if match(get_word(keyword), text):
            return keyword
    return 'no negative word'

# Get formatted current time
def get_formatted_time():
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
    return formatted_time

# Join a list of words into a string
def list2str(ls):
    res = ''
    for word in ls:
        res = res + ' ' + word
    return res.strip()

def load_data(data_path):
    full_sentences = open(data_path, 'r', encoding='utf-8').read().split('\n')
    sentences = []
    real_labels = []
    for sentence in full_sentences:
        try:
            sentences.append(sentence.split('@')[1])
            label = sentence.split('@')[2]
            real_labels.append(label)
        except:
            continue
    return sentences, real_labels

def write_to_excel(path, data):
    if not os.path.exists(path):
        new_file = pd.DataFrame()
        new_file.to_excel(path, header=True, index=False)
    
    result = []
    df = pd.read_excel(path)
    h, w = df.shape
    for i in range(h):
        row = []
        for j in range(w):
            row.append(df.iloc[i, j])
        result.append(row)
    result.append(data)
    result_df = pd.DataFrame(result)
    header = ["approach", "w_p", "w_r", "w_f"]
    result_df.to_excel(path, header=header, index=False)

def format_number(number):
    rounded_number = round(number, 3)
    formatted_number = "{:.3f}".format(rounded_number)
    return formatted_number

def result_report(approach, save_path, predicted_labels, real_labels):
    TP = [0 for _ in range(9)]  # True positives: number of samples actually in class i correctly predicted as i
    FN = [0 for _ in range(9)]  # False negatives: number of samples actually in class i mispredicted as other classes
    FP = [0 for _ in range(9)]  # False positives: number of samples not in class i but predicted as i
    num = [0 for _ in range(9)]  # Number of samples in each class
    
    n = len(real_labels)
    correct = 0  # Number of correctly predicted samples
    
    for i in range(n):
        predicted_label = predicted_labels[i]
        real_label = real_labels[i]
        if predicted_label == real_label:
            correct += 1
            class_id = label_encode[predicted_label]
            TP[class_id] += 1
        else:
            actual_class_id = label_encode[real_label]
            FN[actual_class_id] += 1  # Actual class i mispredicted
            predicted_class_id = label_encode[predicted_label]
            FP[predicted_class_id] += 1  # Predicted class i incorrectly
    
    recalls = []
    precisions = []
    F1_scores = []
    
    for i in range(9):
        if TP[i] + FN[i] == 0:
            recall = 0.0
        else:
            recall = TP[i] / (TP[i] + FN[i])
        if TP[i] + FP[i] == 0:
            precision = 0.0
        else:
            precision = TP[i] / (TP[i] + FP[i])
        
        if recall + precision == 0:
            F1_score = 0.0
        else:
            F1_score = 2 * (precision * recall) / (precision + recall)
        
        precisions.append(precision)
        recalls.append(recall)
        F1_scores.append(F1_score)
    
    label_distribution = {}
    total_samples = 0
    for label in real_labels:
        if label in label_distribution:
            label_distribution[label] += 1
        else:
            label_distribution[label] = 1
        total_samples += 1
    
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f = 0.0
    for label, count in label_distribution.items():
        class_id = label_encode[label]
        weighted_p += precisions[class_id] * (count / total_samples)
        weighted_r += recalls[class_id] * (count / total_samples)
        weighted_f += F1_scores[class_id] * (count / total_samples)
    
    macro_recall = sum(recalls) / len(recalls)
    macro_precision = sum(precisions) / len(precisions)
    macro_f1 = sum(F1_scores) / len(F1_scores)
    
    micro_TP = sum(TP)
    micro_FN = sum(FN)
    micro_FP = sum(FP)
    micro_recall = micro_TP / (micro_TP + micro_FN) if (micro_TP + micro_FN) != 0 else 0.0
    micro_precision = micro_TP / (micro_TP + micro_FP) if (micro_TP + micro_FP) != 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) != 0 else 0.0
    
    print(f"* Weighted Precision: {weighted_p:.4f}")
    print(f"* Weighted Recall: {weighted_r:.4f}")
    print(f"* Weighted F1-score: {weighted_f:.4f}")
    print("-----------------------------------")
    
    row_data = [approach, format_number(weighted_p), format_number(weighted_r), format_number(weighted_f)]
    write_to_excel(save_path, row_data)

def get_date_time():
    current_date = datetime.now()
    return current_date.strftime('%Y%m%d')

def write_log(log_path, log_msg):
    with open(log_path, 'a', encoding="utf-8") as f:
        for line in log_msg:  # Write log line by line
            f.write(line + '\n')
        f.write("---------------------------" + '\n')
        f.write("\n")
    f.close()