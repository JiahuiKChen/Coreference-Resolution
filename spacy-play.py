import spacy
import sys


# NP parsing... other things
nlp = spacy.load("en_core_web_sm")


# Load in a file
args = sys.argv
filename = args[1]
text = ""
with open(filename) as file:
    for line in file:
        text += line 
#print(text)


# Getting the model...
parsed = nlp(text)
noun_chunks = list(chunk.text for chunk in parsed.noun_chunks)
for noun in noun_chunks:
    print(noun)
