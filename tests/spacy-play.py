import spacy
import sys


# Large English model that has lots of word similarity vectors...
nlp = spacy.load("en_core_web_lg")


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
#noun_chunks = list(chunk.text for chunk in parsed.noun_chunks)
noun_chunks = list(chunk for chunk in parsed.noun_chunks)
# Each item in noun_chunks is a "slice" of "tokens" (segment of words that are a noun phrase)
first_np = noun_chunks[0]
for noun_span in noun_chunks:
    print(f"First NP: {first_np.text}    -    Other NP: {noun_span.text}")
    sim = first_np.similarity(noun_span)
    print(f"Similarity: {sim}")
