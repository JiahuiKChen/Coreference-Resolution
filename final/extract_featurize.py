### Given an answer and corresponding input file,
### featurizes pairs into negative and positive label features
# RUN WITH : extract-featurize-pairs.py <answer_file> <input_file>
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


# Given 2 Spacy-processed phrases (noun phrases that are either Spans or Docs), returns their feature vector
#
# FOR NOW: RETURNS THEIR SIMILARITY SCORE
# TODO???: RETURN A BUNCH OF FEATURES (NOT JUST SIMILARITY SCORE)
def get_pair_features(p1, p2):
    return p1.similarity(p2)
    #return 'yeet'


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

    # Generating features for each initial reference/cluster of mentions
    for initial in mention_map:
        processed_initial = nlp_model(initial)
        # Positive pairs are initial ref with each of their mentions
        references = mention_map[initial]
        for ref in references:
            processed_ref = nlp_model(ref)
            pos_feature = get_pair_features(processed_initial, processed_ref)

            print(f"Initial Ref: {initial}    Mention: {ref}")
            print(pos_feature)
            pos_features.append(pos_feature)

        # TODO: GEN NEGATIVE FEATURES
        # BY ALL GOING THROUGH ALL NP CHUNKS/SPANS IN INPUT TEXT (WITH REMOVED TAGS)
        # THEN FEATURIZE EACH NP CHUNK PAIR THAT DOES NOT INCLUDE ANY VALID REFERENCE 

    return pos_features, neg_features


def featurize_file(answer_file, input_file, nlp_model):
    mention_map = parse_true_mentions(answer_file)
    pos_features, neg_features = gen_features(mention_map, input_file, nlp_model)


######################################## RUN FEATURIZATION OF KEY FILE
## Load in large English model from spacy
#nlp = spacy.load("en_core_web_lg")
#
## Load in input and key files, corresponding to same text
#args = sys.argv
#key_file = args[1]
#input_file = args[2]
#
#featurize_file(key_file, input_file, nlp)
