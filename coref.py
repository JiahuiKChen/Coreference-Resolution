import sys
import spacy
import re
from extract_featurize import get_pair_features
from util import get_input_file_text


# Finds and removes initial references given a sentence
#
# Returns:  map of <initial reference (just the noun phrase), initial reference with surrounding reference ID tags>
#           list of initial references, in the order they were found (later is higher ind)
#           text of sentence with intial references REMOVED (so np extraction gets only other NPs)
def extract_initial_refs(sentence_txt):
    #coref_end_reg = '(</COREF>)'
    #coref_start_reg = '(<COREF ID.*?>)'
    coref_extract_reg = '<COREF ID.*?>(.*?)</COREF>'
    coref_with_tags_extract_reg = '(<COREF ID.*?>.*?</COREF>)'
    ref_map = {}
    refs = []

    # Trying to find initial coreferences/tags
    inits = re.findall(coref_extract_reg, sentence_txt)
    if len(inits) > 0:
        refs = inits
        ref_fulls = re.findall(coref_with_tags_extract_reg, sentence_txt)
        if (len(refs) != len(ref_fulls)):
            raise Exception("Number of initial references found in sentence doesn't match number of reference IDs found!\
                    \n FROM extract_initial_refs in coref run")
        # Mapping each inital reference to its ID
        for ref_ind in range(len(refs)):
            ref_map[refs[ref_ind]] = ref_fulls[ref_ind]

        # Removing inital references entirely, including tags 
        no_inits = re.sub(coref_with_tags_extract_reg, '', sentence_txt)
        return ref_map, refs, no_inits 
    else:
        return ref_map, refs, sentence_txt


# Runs coreference resolution on all sentences in the given file
def run_coref(input_file, nlp_model):
    # Get input text 
    input_txt, sentence_map = get_input_file_text(input_file)
    # Tracks possible initial references (as an initial coref is encountered, it's added)
    possible_initials = []
    # Holds all initial references mapped to their coreference ID tag string
    initials_map = {}

    # For each sentence, run best-first coreference resolution
    for s in sentence_map:
        # Gets sentence with initial coref ID tags in it
        s_txt = sentence_map[s]
        # Get initial coreferences in sentence (if there are any), and sentence text with all initial references removed
        ref_map, new_initials, sentence = extract_initial_refs(s_txt)
        # Add new initial coreferences to possible initials, map of all initials
        possible_initials.extend(new_initials)
        initials_map.update(ref_map)

        # Get all np chunks/spans from sentence (excluding initial references, so all these NPs should be checked for coreference)
        parsed_sentence = nlp_model(sentence)
        np_chunks = list(chunk for chunk in parsed_sentence.noun_chunks)
        print(possible_initials, np_chunks)



###################################### RUNNING COREFERENCE ON INPUT FILES
# Load in large English model from spacy
nlp_model = spacy.load("en_core_web_lg")

# Parse in the given list_file (of input file names to run coreference on)
# and output directory
try:
    arg_list = sys.argv
    if len(arg_list) < 3:
        raise Exception("Invalid command line arguments")
    list_file = arg_list[1].strip()
    output_dir = arg_list[2].strip()
    
    # Running coreference resolution on each input file 
    with open(list_file) as in_file:
        for input_name in in_file:
            run_coref(input_name.strip(), nlp_model)
except Exception as e: 
   print("\nException thrown:\n")
   print(str(e), '\n')
