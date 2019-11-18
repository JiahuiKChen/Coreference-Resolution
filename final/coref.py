import sys
import spacy
import re
from extract_featurize import get_pair_features
from util import get_input_file_text
from joblib import dump, load


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


# Creates string of coreference extractions in specified format
def format_response(coref_map, initial_refs_map, ordered_initials):
    #print(initial_refs_map, ordered_initials)
    response_str = ""
    for initial in ordered_initials:
        if initial not in coref_map:
            continue
        corefs = coref_map[initial]
        full_tag = initial_refs_map[initial]
        response_str += full_tag + '\n'
        for coref in corefs:
            coref_txt = coref[0]
            sentence_id = coref[1]
            response_str += "{" + sentence_id + "} " + "{" + coref_txt + "}\n" 
        response_str += '\n'
    #print(response_str)
    return response_str

# Runs coreference resolution on all sentences in the given file
#
# Returns:      Responses as map of initial references to a list of (coreference text, sentence ID)
#               Map of all initial references (text only) to their full coreference tag wrapping
#               List of initial references IN ORDER IN WHICH THEY APPEAR IN INPUT
def run_coref(input_file, nlp_model, model_file):
    # Get input text 
    input_txt, sentence_map = get_input_file_text(input_file)
    # Tracks possible initial references (as an initial coref is encountered, it's added)
    possible_initials = []
    # Holds all initial references mapped to their coreference ID tag string
    initials_map = {}
    # Tracks found coreferences: <intial reference, (NP that matched, sentence_id)
    found_corefs = {}

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

        # MENTION PAIR MODEL APPROACH
        # Pre-trained mention-pair model loaded in
        #model = load(model_file)

        # For each np, pair with the most recent intiial reference - run pair through the mention-pair classifier 
        for np in np_chunks:

            # Try mention pairs with initial references (trying most recent first), until all are tried OR positive match is found
            for init_ind in range(len(possible_initials)-1, -1, -1):
                init_ref = possible_initials[init_ind]
                potential_ref = nlp_model(init_ref)

                #print(f"Trying NP {np.text} with Initial Reference: {init_ref}")
                
                # THE MENTION-PAIR MODEL APPROACH (doesn't work as well as just using similarity) :( 
                #pair_features = get_pair_features(potential_ref, np).reshape(1, -1)
                #prediction = model.predict(pair_features)
                #if prediction > 0:
                #    if init_ref not in found_corefs:
                #        found_corefs[init_ref] = []
                #    found_corefs[init_ref].append((np.text, s))
                #    break

                # Primitive approach that just uses similarity...
                pair_features = get_pair_features(potential_ref, np)
                sim_score = pair_features[0]
                contains = pair_features[1] 
                #match_lemmas = pair_features[2]
                #match_ners = pair_features[3]
                #match_caps = pair_features[4]

                ## Number of matching lemmas is a plus 
                #sim_score += 0.5 * match_lemmas
    
                ## Containment is a huge plus
                #sim_score += contains 

                ## Matching NER is a huge plus
                #sim_score += 0.8 * match_ners

                ## Same number of capitals is (minor) plus
                #if match_caps == 0:
                #    sim_score += 0.1
                #if contains == 1:
                #    if init_ref not in found_corefs:
                #        found_corefs[init_ref] = []
                #    found_corefs[init_ref].append((np.text, s))
                #    break 
                # Special case where there's for sure containment
                if sim_score == 10 or contains == 1:
                    if init_ref not in found_corefs:
                        found_corefs[init_ref] = []
                    # Add in only the matching word to avoid capturing too much of a phrase
                    for w in np:
                        for t in potential_ref:
                            np_match = w.text.lower()
                            ref_word = t.text.lower()
                            if np_match == ref_word:
                                found_corefs[init_ref].append((w.text, s))
                if sim_score > 0.75:
                    if init_ref not in found_corefs:
                        found_corefs[init_ref] = []
                    found_corefs[init_ref].append((np.text, s))
                    break 

    print(found_corefs)

    # Generating formatted coreference responses
    response_str = format_response(found_corefs, initials_map, possible_initials)
    return response_str


# Writes given response coreference string to output dir
def write_response(response_str, input_file, output_dir):
    # Getting name of input file
    input_name = input_file.split('/')[-1]
    prefix = input_name.split('.')[0].strip()
    output_file = output_dir + prefix + ".response"
    out_file = open(output_file, "w")
    out_file.write(response_str)


###################################### RUNNING COREFERENCE ON INPUT FILES
# Load in large English model from spacy
nlp_model = spacy.load("en_core_web_lg")

# The saved pre-trained mention pair models
#tree_model = "3x-neg-mention-pair-tree.joblib"
svm_model = "svm-mention-pair-model.joblib"

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
            response = run_coref(input_name.strip(), nlp_model, svm_model)
            write_response(response, input_name.strip(), output_dir)        
except Exception as e: 
   print("\nException thrown:\n")
   print(str(e), '\n')
