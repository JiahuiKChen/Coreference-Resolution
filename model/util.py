import sys
import re


# Get text from an input file, remove the tags from it 
# 
# Returns input_text:           string of just text (without any sentence ID and reference ID tags)
# Returns input_sentences:      map of <id, sentence> where the sentence is a string with sentence ID tags removed 
#                               (but coreference ID tag of initial references still inside)
def get_input_file_text(input_file):
    sentence_splits_regex = '(<S ID.*?>.*?</S>)'
    sentence_txt_regex = '<S ID.*?>(.*?)</S>'
    coref_end_reg = '(</COREF>)'
    coref_start_reg = '(<COREF ID.*?>)'
    sentence_map = {}
    sentences_text = ""

    with open(input_file) as in_file:
        raw_text = in_file.read()
    # Split raw text into sentences
    sentences = re.split(sentence_splits_regex, raw_text)
    for s in sentences:
        if len(s) > 0 and s[0] == '<':
            sentence_id = s[7]
            sentence_text = re.split(sentence_txt_regex, s)[1]
            sentence_map[sentence_id] = sentence_text

            # Removing coref tags 
            pure_txt = re.sub(coref_start_reg, '', sentence_text)
            pure_txt = re.sub(coref_end_reg, '', pure_txt)
            #print(sentence_text)
            #print(pure_txt)
            sentences_text += ' ' + pure_txt


    #print(sentence_map)
    #print(sentences_text)
    return sentences_text, sentence_map

#get_input_file_text(sys.argv[1])
