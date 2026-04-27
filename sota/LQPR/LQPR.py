from tools import *
from static_data import *
import matplotlib.pyplot as plt
import random
import spacy
import numpy as np

languagePatterns = []
labels = []
pattern_vecs = []

# Word vector table
word2vec = {}
# Load spaCy's English model
nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

def cosine_similarity(vector1, vector2):
    # L2 normalize the vectors
    vector1_normalized = vector1 / np.linalg.norm(vector1)
    vector2_normalized = vector2 / np.linalg.norm(vector2)
    
    # Calculate the dot product of the normalized vectors (cosine similarity)
    cosine_sim = np.dot(vector1_normalized, vector2_normalized)
    return (cosine_sim + 1) / 2 # Map cosine similarity to [0, 1] interval

def sentence_vector(sentence):
    doc = nlp(sentence)
    vectors = [token.vector for token in doc if token.has_vector]
    if not vectors:
        # Return a default vector with the same shape as normal results
        default_vector = np.ones((96,))
        return default_vector
    return np.mean(vectors, axis=0)

# Calculate semantic similarity
def semantic_similarity(vecs, vec2):
    res = 0
    for vec1 in vecs:
        res = max(res, cosine_similarity(vec1, vec2))
    return res

def get_split_sentence_vec(word_list):
    res = []
    for substr_len in range(2, 6):  # Extract substrings of different lengths
        for i in range(1, len(word_list)):
            if i + substr_len > len(word_list):
                break
            sub_str = ' '.join(word_list[i : i+substr_len])
            res.append(sentence_vector(sub_str))
    return res

def init_pattern_vecs():
    full_languagePatterns = open('sota/LQPR/pattern.txt', 'r', encoding='utf-8').read().split('\n')
    for pattern in full_languagePatterns:
        languagePatterns.append(get_word(pattern.split('$')[0]))
        labels.append(pattern.split('$')[1])
        
    for pattern in languagePatterns:
        print(' '.join(pattern[1: ]))
        pattern_vecs.append(sentence_vector(' '.join(pattern[1: ])))

    print("pattern_vecs initialization completed")
        
# Match sentence with all patterns in both syntactic and semantic ways
def matching(sentence, config):
    use_sync = config['use_sync']
    use_semantic = config['use_semantic']
    weight = config['weight']
    possible_result = []
    num = len(languagePatterns)
    word_list = get_word(sentence)
    for i in range(num):
        pattern = languagePatterns[i]
        LCS, score = lcs(word_list, pattern)
        if len(pattern) == 3:
            if len(LCS) > 0:
                score = get_punishment_score(LCS[-1] - LCS[0] + 1, 2, score)
                
        # Use only semantic matching
        if not use_sync:
            lcs_str = ' '.join(word_list[index] for index in LCS)
            lsc_vec = sentence_vector(lcs_str)
            score = cosine_similarity(lsc_vec, pattern_vecs[i])
        
        # Syntactic similarity + semantic similarity
        if use_semantic and use_sync:
            lcs_score = score
            lcs_str = ' '.join(word_list[index] for index in LCS)
            lsc_vec = sentence_vector(lcs_str)
            sem_score = cosine_similarity(lsc_vec, pattern_vecs[i])
            score = weight * lcs_score + (1 - weight) * sem_score
            
        possible_result.append((LCS, pattern, labels[i], score, i))
        possible_result = sorted(possible_result, key=key_function, reverse=True)

    return possible_result

# Core function to get various prediction information
def predict(big_sentence, config):
    use_inv = config['use_inv']
    score = 0
    Changing_trend = ''
    matched_pattern = ''
    negative_word = 'no negative word'
    big_sentence = big_sentence.lower()
                    
    possible_result = matching(big_sentence, config) 
    
    if possible_result[0][4] == 0:  # Matched with "100 % of"
        if possible_result[0][3] == possible_result[1][3]:
            if possible_result[0][0][0] == possible_result[1][0][-1]:  # Modifying "100 % of"
                possible_result[0] = possible_result[1]  # Modifier indicates large

    result = possible_result[0]
    Changing_trend = result[2]
    negative_word = is_passive(big_sentence)
    
    if use_inv:
        if negative_word != 'no negative word':  # Semantic inversion
            if result[4] != 0:  # "100 % of" does not require semantic inversion
                Changing_trend = label_inv[Changing_trend] 

    score = result[3]                 
    matched_pattern = result[1]
    
    return (Changing_trend, list2str(matched_pattern[1:]), score, negative_word)

def LQPR(sentence):
    config = {  
        'use_inv': True,
        'use_sync': True,
        'use_semantic': True,
        'weight': 0.7
    }
    Changing_trend, matched_pattern, score, negative_word = predict(sentence, config)
    num = get_number(matched_pattern)
    res = []
    if Changing_trend == "1 0":
        res.append([num * 0.9, 0.0])
        res.append([num, 1.0])
    elif Changing_trend == "0 -1":
        res.append([num, 1.0])
        res.append([num * 1.1, 0.0])
    else:
        res.append([num * 0.9, 0.0])
        res.append([num, 1.0])
        res.append([num * 1.1, 0.0])

    return res

