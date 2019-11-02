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
#print(blob.tags)
for noun in blob.noun_phrases:
    print(noun)
