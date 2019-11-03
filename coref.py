import sys
import os


# Runs coreference resolution on all sentences in the given file
def run_coref(input_file):
    with open(input_file) as text_file:
        # Get noun phrases for each sentence
        # From left to right, pass noun phrase pairs into mention-pair classifier
        for sentence in text_file:
            print(sentence)


###################################### RUNNING COREFERENCE ON INPUT FILES
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
            run_coref(input_name.strip())
except Exception as e: 
    print("\nException thrown:\n")
    print(str(e), '\n')
