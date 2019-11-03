### Given an answer and corresponding input file,
### featurizes pairs into negative and positive label features
# RUN WITH : extract-featurize-pairs.py <answer_file> <input_file>
# OR PASS IN FILES BY NAME TO featurize_file() METHOD
import sys
import re
import numpy as np


# Returns a dictionary mapping initial reference to all mentions
# Reads in answer file as the first command line argument
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


# Given 2 words, returns their feature vector
def get_pair_features(w1, w2):



# Given a map of all initial references to their mentions,
# featurizes every positive pair and all negative noun phrase pairs between each mention
# Positive Label Pairs: All pairs of intial references with each of their mentions
# Negative Label Pairs: All pairs of extracted noun phrases from all sentences with each initial reference
# 
# Returns list of positive features and list of negative features
def gen_features(mention_map, input_file):
    pos_features = []
    neg_features = []

    # Generating positive features
    for initial in mention_map:
        references = mention_map[initial]


def featurize_file(answer_file, input_file):
    mention_map = parse_true_mentions(answer_file)
    features = gen_features(mention_map, input_file)


######################################## RUN FEATURIZATION OF KEY FILE
# Load in key file
args = sys.argv
key_file = args[1]
input_file = args[2]

featurize_file(key_file, input_file)
