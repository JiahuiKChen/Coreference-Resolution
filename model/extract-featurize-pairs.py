### Given an answer and corresponding input file,
### featurizes pairs into negative and positive label features
# RUN WITH : extract-featurize-pairs.py <answer_file> <input_file>
# OR PASS IN FILES BY NAME TO featurize_file() METHOD
import sys
import re
import numpy as np


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


# Given 2 phrases (noun phrases), returns their feature vector
def get_pair_features(p1, p2):



# Given a map of all initial references to their mentions and the input text file,
# returns all negative and positive features for the given references.
# Positive Label Pairs: All pairs of intial references with each of their mentions
# Negative Label Pairs: All pairs of extracted noun phrases from all sentences with each initial reference
# 
# Returns list of positive features and list of negative features
def gen_features(mention_map, input_file):
    pos_features = []
    neg_features = []

    # TODO: GET INPUT FILE'S TEXT WITH ALL COREF AND SENTENCE ID TAGS REMOVED

    # TODO: GET NOUN CHUNKS FROM THE INPUT TEXT

    # Generating features for each initial reference/cluster of mentions
    for initial in mention_map:
        # Positive pairs are initial ref with each of their mentions
        references = mention_map[initial]
        for ref in references:
            pos_feature = get_pair_features(initial, ref)
            pos_features.append(pos_feature)
            # TODO: FINISH POS FEATURE GEN

        # TODO: GEN NEGATIVE FEATURES
        # BY ALL GOING THROUGH ALL NP CHUNKS/SPANS IN INPUT TEXT (WITH REMOVED TAGS)
        # THEN FEATURIZE EACH NP CHUNK PAIR THAT DOES NOT INCLUDE ANY VALID REFERENCE 


######################################## RUN FEATURIZATION OF KEY FILE
# Load in key file
args = sys.argv
key_file = args[1]
input_file = args[2]

featurize_file(key_file, input_file)
