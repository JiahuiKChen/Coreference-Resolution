import sys
import re


# Returns a dictionary mapping initial reference to all mentions
# Reads in answer file as the first command line argument
def parse_true_mentions():
    # Regexes to extract words
    init_reg = '>(.*)<'
    ref_reg = '{(.*?)}'
    
    # Load in key file
    args = sys.argv
    filename = args[1]
    # Fill map with all intial clusters mapped to their references
    mention_clusters = {}
    with open(filename) as file:
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


# Given a map of all initial references to their mentions,
# featurizes every positive pair and all negative noun phrase pairs between each mention
# TODO: HOW TO GET NEGATIVE PAIRS!!!
def gen_features(mention_map):


######################################## RUN FEATURIZATION OF KEY FILE
mention_map = parse_true_mentions()
