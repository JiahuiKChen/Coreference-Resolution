### Given an answer and corresponding input file,
### featurizes pairs into negative and positive label features
# RUN (1): extract-featurize-pairs.py <answer_file> <input_file>
# OR (2): extract-featurize-pairs.py <list of answer files> <list of input files>
# OR PASS IN FILES BY NAME TO featurize_file() METHOD
import sys
import re
import numpy as np
import spacy
from util import get_input_file_text


# Returns a dictionary mapping initial reference to all mentions
# from answer file of true mentions
def parse_true_mentions(key_file):
    # Regexes to extract words
    init_reg = '>(.*)<'
    ref_reg = '{(.*?)}'
    
    # Fill map with all intial clusters mapped to their references
    mention_clusters = {}
    with open(key_file) as file:
        init_ref = ''
        for line in file:
            # Initial references are keys in the answer map
            if line[0] == '<':
                init_ref = re.split(init_reg, line)[1]
                mention_clusters[init_ref] = []
            elif line[0] == '{':
                ref_line = re.split(ref_reg, line)
                min_ref = ref_line[5]            
                max_ref = ref_line[3]
                mention_clusters[init_ref].append(min_ref)
                # Only make 2 separate coreferent entries if min and max are different
                if min_ref != max_ref: 
                    mention_clusters[init_ref].append(max_ref)
    return mention_clusters


# FEATURE VALUES ##########################################################################################

# Cosine similarity of 2 spacy noun phrase chunks/spans
def similarity(p1, p2):
    return p1.similarity(p2)


# 1 if either noun phrase is substring of the other, 0 otherwise
def substring(p1, p2):
    text1 = p1.text
    text2 = p2.text
    if text1 in text2:
        return 1
    elif text2 in text1:
        return 1
    else: 
        return 0


# Returns count of phrases that match in number/plurality 
# implemented by counting matching lemmas of all tokens in each phrase
def plurality(p1, p2):
    min_len = min(len(p1), len(p2))
    match_count = 0
    for i in range(min_len):
        lemma1 = p1[i].lemma_
        lemma2 = p2[i].lemma_
        if lemma1 == lemma2:
            match_count += 1
    return match_count


# Returns number of matching Named Entities 
def ner(p1, p2):
    ner1 = list(p1.ents)
    ner2 = list(p2.ents)
    min_len = min(len(ner1), len(ner2))
    match_count = 0
    for i in range(min_len):
        n1 = ner1[i]
        n2 = ner2[i]
        if n1.label == n2.label:
            match_count += 1
    return match_count


# Returns number of differently capitalized words in noun phrases
def cap_diffs(p1, p2):
    words1 = p1.text.split()
    words2 = p2.text.split()
    min_len = min(len(words1), len(words2))
    diff_count = 0
    for i in range(min_len):
        w1 = words1[i]
        w2 = words2[i]
        if w1[0].isupper() != w2[0].isupper():
            diff_count += 1
    return diff_count


# Given 2 Spacy-processed phrases (noun phrases that are either Spans or Docs), returns their feature vector
def get_pair_features(p1, p2):
    features = [ similarity(p1, p2), substring(p1, p2) ] #plurality(p1, p2), ner(p1, p2), cap_diffs(p1, p2) ] 
    return np.array(features)


# Given a map of all initial references to their mentions and the input text file,
# returns all negative and positive features for the given references.
# Positive Label Pairs: All pairs of intial references with each of their mentions
# Negative Label Pairs: All pairs of extracted noun phrases from all sentences with each initial reference
# 
# Returns list of positive features and list of negative features
def gen_features(mention_map, input_file, nlp_model):
    pos_features = []
    neg_features = []

    # Getting input file's text with all tags removed
    input_txt, s_map = get_input_file_text(input_file) 

    # Passing input text through spacy model (for similarity vectors, noun phrases, etc)
    processed_input = nlp_model(input_txt)

    # Getting noun chunks/spans using spacy from the input text
    np_chunks = list(chunk for chunk in processed_input.noun_chunks)

    # Positive labelled features gen
    # Generating features for each initial reference/cluster of mentions
    for initial in mention_map:
        processed_initial = nlp_model(initial)
        # Positive pairs are initial ref with each of their mentions
        references = mention_map[initial]
        for ref in references:
            processed_ref = nlp_model(ref)
            pos_feature = get_pair_features(processed_initial, processed_ref)

            #print(f"Initial Ref: {initial}    Mention: {ref}")
            #print(pos_feature)

            pos_features.append(pos_feature)

        # Negative labelled features gen
        # Generaing features for all non-reference pairs (each initial reference with each non-reference word)
        for non_ref in np_chunks:
            for initial in mention_map:
                processed_initial = nlp_model(initial)
                neg_feature = get_pair_features(non_ref, processed_initial)

                #print(f"Initial Ref: {initial}    Non-reference noun phrase: {non_ref}")
                #print(neg_feature)

                neg_features.append(neg_feature)

    return pos_features, neg_features


def featurize_file(answer_file, input_file, nlp_model, save_dir=''):
    mention_map = parse_true_mentions(answer_file)
    pos_features, neg_features = gen_features(mention_map, input_file, nlp_model)
    file_name = input_file.split('/')[-1].split('.')[0]

    np.save(save_dir + file_name + ".pos_features", pos_features)
    np.save(save_dir + file_name + ".neg_features", neg_features)


# Given a python list of input and output files (indices must correspond)
# Featurizes each pair, outputting features to save_dir 
def featurize_list(in_files, key_files, nlp_model, save_dir):
    if len(in_files) != len(key_files):
        raise Exception("MISTMATCHED NUMBER OF INPUT AND KEY FILES")
        return
    for i in range(len(in_files)):
        input_file = in_files[i].strip()
        key_file = key_files[i].strip()
        featurize_file(key_file, input_file, nlp_model, save_dir)


######################################## RUN FEATURIZATION OF KEY FILE
## Load in large English model from spacy
#nlp = spacy.load("en_core_web_lg")
#
## RUN (1)
## Load in input and key files, corresponding to same text
#args = sys.argv
#key_file = args[1]
#input_file = args[2]
#featurize_file(key_file, input_file, nlp)
#
# RUN (2)
# Load in list of input files and ouptut files
#args = sys.argv
#key_list = args[1]
#input_list = args[2]
#with open(key_list) as keys:
#    key_files = keys.readlines()
#with open(input_list) as inputs:
#    in_files = inputs.readlines()
#featurize_list(in_files, key_files, nlp, "features/")
