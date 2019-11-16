# Decision Tree classifier to determine if pairs of words 
# are coreference pairs
# RUN BY: python mention-pair-model.py <feature-list> - where feature-list is a text file with a feature file on each line

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score
import numpy as np

import sys
from joblib import dump, load


# Loading in features 
args = sys.argv
feature_list = args[1]
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

pos_data = np.concatenate(pos_features, axis=0)
neg_data = np.concatenate(neg_features, axis=0)

#all_data = np.concatenate((pos_data, neg_data), axis=0)
#all_labels = np.concatenate((np.ones((pos_data.shape[0])), np.zeros(neg_data.shape[0])))

# Using same amounts of neg and pos vectors in training
np.random.shuffle(neg_data)
neg_data = neg_data[:(pos_data.shape[0])*1000]
all_data = np.concatenate((pos_data, neg_data), axis=0)
all_labels = np.concatenate((np.ones((pos_data.shape[0])), np.zeros(neg_data.shape[0])))


######################################Train model
# Cross validate on tree depth and minimum samples needed to split a node
depths = [2, 4, 6, 8]
min_samples = [2, 4, 6, 8]

best_acc = 0
best_params = [0, 0] # (best depth, best min sample split)
for depth in depths: 
    for min_sample in min_samples:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_sample)
        cv_scores = cross_val_score(tree, all_data, all_labels, cv=5) 
        cv_mean = np.mean(cv_scores)
        if cv_mean > best_acc:
            best_acc = cv_mean
            best_params[0] = depth
            best_params[1] = min_sample

print(f"BEST CROSS VALIDATION MEAN ACC: {best_acc}")

# Train model on entire dataset with best parameters
tree = DecisionTreeClassifier(max_depth=best_params[0], min_samples_split=best_params[1])
tree.fit(all_data, all_labels)

#####################################Save Model...
dump(tree, "1000x-neg-mention-pair-tree.joblib")
