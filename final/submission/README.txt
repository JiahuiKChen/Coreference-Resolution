Jiahui Chen
u0980890

**** RUN MY CODE ****
MUST USE VENV TO RUN:

source bin/activate.csh
python coref.py <list_file> <response_dir>

NOTES: 
    'python' in this venv can also run the scorer.py
    all_inputs is a list of all input files (original development set, test1 and test2 included in scorer folder, and test set for midpoint evaluation)
    all_keys is list of all keys (corresponding to all files in all_inputs)
    keys/ is the directory where all keys are located
    inputs/ is the directory where all inputs are located
    full_names is a list of all file names (without .input or .key) used for running scorer.py on all files

**** EXTERNAL RESOURCES ****
spacy:      https://spacy.io/
nltk:       https://www.nltk.org/
numpy:      https://numpy.org/
I implemented a modified version of the method and got some of my mention-pair features from this paper:
     https://www.cs.cornell.edu/home/cardie/papers/acl2002.pdf
Some features for mention-pair vetors were taken from slide 45 of this deck:    
    https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/handouts/cs224n-lecture10-coreference.pdf


**** TIME ESTIMATE FOR ONE DOCUMENT ****
It takes about 20 seconds to run coref.py on only a10.input
This is perhaps longer than usual, since loading in spacy's large english model takes some time
and only occurs at the start of the program.


**** CONTRIBUTIONS ****
I worked alone, all the code and work is solely mine.


**** PROBLEMS ****
none...


**** OTHER INFO ****
All code was developed and tested on lab1-17
I used Spacy for noun phrase chunking, tokenizing, and word vector cosine similarity

I implement a modified version of the Best Match algorithm 
(presented in "Improving Machine Learning Approaches to Coreference Resolution" by Vincent Ng and Claire Cardie),
that only tries pairing NPs with initial references.

I use cosine similarity on word vectors and a similarity threshold to determine if a pair is a mention.
When the word vectors are empty, I use a weighted combination of features as the similarity value.
I also consider all substrings/word matches as coreferences when they match the head noun of an initial reference.

