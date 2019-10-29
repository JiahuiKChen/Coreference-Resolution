from textblob import TextBlob 
import sys


# Load in a file
args = sys.argv
filename = args[1]
text = ""
with open(filename) as file:
    for line in file:
        text += line 
#print(text)


# Getting the model...
blob = TextBlob(text)
print(blob.tags)

#noun_chunks = list(chunk.text for chunk in parsed.noun_chunks)
#for noun in noun_chunks:
#    print(noun)
