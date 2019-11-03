import sys
import re


# Get text from an input file, remove the tags from it 
# 
# Returns input_text:           string of just text (without any sentence ID and reference ID tags)
# Returns input_sentences:      map of <id, sentence> where the sentence is a string with sentence ID tags removed 
#                               (but coreference ID tag of initial references still inside)
def get_input_file_text(input_file):
    sentence_regex = '(<S ID.*?>.*?</S>)'

    with open(input_file) as in_file:
        raw_text = in_file.read()
    # Split raw text into sentences
    sentences = re.split(sentence_regex, raw_text)
    print(sentences) 

get_input_file_text(sys.argv[1])
