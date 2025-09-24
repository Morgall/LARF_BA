# running tree on taxa easy with increasing feature subsets until solver throws malloc because of contraint matrix size

import pandas as pd
from tree_refactored.class_tree import DecisionTree_rollOCT
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.model_selection import train_test_split

"""
name_dataset_dict = {
    'data_test': 'test',
    'data_breast_cancer': 'breast+cancer+wisconsin+diagnostic',
    'data_car_eval' : 'car_evaluation',
    'data_mushroom' : 'mushroom',
    'data_nursery' : 'nursery',
    'data_seismic' : 'seismic',
    'data_spambase' : 'spambase',
    'data_wine' : 'wine',
    'data_adult' : 'adult'
     }
"""

target_label = "y"
depth_rolling_tree = 2
criterion_loss = "gini"

subset_size_list = [0.1, 0.25, 0.5, 0.75] # percentages

#folds_cross_val = 10

to_do_dict = dict() # add datasets to be run into this dict and choose key as dataset name

data_microbiome_taxa_easy = pd.read_csv("datasets/microbiome_taxa_counts_easy/microbiome_taxa_counts_easy_bin.csv")
to_do_dict['microbiome_taxa_easy'] = data_microbiome_taxa_easy

for dataset_name, data in to_do_dict.items(): #.items() gives key, values

    features = data.drop(columns=['y'])
    targets = data['y']

    num_columns = features.shape[1] #number of all features


    for perc in subset_size_list:

        subset_int = int(perc * num_columns)
        subset = features.sample(n=subset_int, axis=1, random_state=42) #subset of features the tree is to be trained on

        X = subset
        y = targets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #75% training, 25% testing; default in scikit-learn
    
        dir_path = f'microbiome_data/taxa_easy/const_mat_test/subset_size_{subset_int}_d{depth_rolling_tree}'

        # Create the directory if it doesn't exist
        os.makedirs(f'{dir_path}', exist_ok=True)
        
        with open(f'{dir_path}/time.txt', 'w') as f:
            pass  # This just creates/truncates the file

        tree = DecisionTree_rollOCT(max_depth=depth_rolling_tree, max_features = None)
        tree.fit(X_train, y_train)
        
        r_test = tree.predict(y_test)
        result_test = pd.concat([y_test, r_test['prediction']], axis=1)

        r_train = tree.predict(y_train)
        result_train = pd.concat([y_train, r_train['prediction']], axis=1)

        with open(f'{dir_path}/time.txt', 'a') as f:
            f.write(str(tree.fit_time))

        with open(f'{dir_path}/result_test.csv', 'w') as f:
            f.write(str(result_test.to_csv()))

        with open(f'{dir_path}/result_train.csv', 'w') as f:
            f.write(str(result_train.to_csv()))



