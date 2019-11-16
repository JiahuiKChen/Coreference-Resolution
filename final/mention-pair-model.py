# Decision Tree classifier to determine if pairs of words 
# are coreference pairs
# RUN BY: python mention-pair-model.py <feature-list> - where feature-list is a text file with a feature file on each line

from sklearn.tree import DecisionTreeClassifier 
import sys
import numpy as np


# Loading in features 
args = sys.argv
feature_list = args[0]
pos_features = []
neg_features = []

# Load in each np array of features, based on if they're pos or neg label features
with open(feature_list) as feature_file:
    for f_file in feature_file:
        features = np.load(f_file.strip())
        if "neg_features" in f_file.strip():
            neg_features.append(features)
        elif "pos_features" in f_file.strip():
            pos_features.append(features)
        else:
            raise Exception("UNKNOWN FEATURE FILE FOUND")
            return

pos_data = np.concatenate(pos_features, axis=0)
neg_data = np.concatenate(neg_features, axis=0)

######################################Train model

#####################################Test Model

#####################################Save Model...
