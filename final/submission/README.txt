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


**** TIME ESTIMATE FOR ONE DOCUMENT ****


**** CONTRIBUTIONS ****


**** PROBLEMS ****
none...


**** OTHER INFO ****
All code was developed and tested on lab1-17
I used Spacy for noun phrase chunking, tokenizing, and word vector cosine similarity

I implement a modified Best Match algorithm 
(presented in "Improving Machine Learning Approaches to Coreference Resolution" by Vincent Ng and Claire Cardie),
that only tries pairing NPs with initial references.

I use cosine similarity on word vectors and a similarity threshold to determine if a pair is a mention.
