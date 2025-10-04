import pandas as pd
from tree_refactored.class_tree import DecisionTree_rollOCT
from sklearn.model_selection import StratifiedKFold
import os
import pickle
import time

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
depth_rolling_tree = 8
criterion_loss = "gini"

folds_cross_val = 10

to_do_dict = dict() # add datasets to be run into this dict and choose key as dataset name

data_test = pd.read_csv("datasets/example_datasets/stacked.csv")
to_do_dict['test'] = data_test

data_breast_cancer = pd.read_csv("datasets/breast+cancer+wisconsin+diagnostic/wdbc_bin.csv")
to_do_dict['breast+cancer+wisconsin+diagnostic'] = data_breast_cancer

data_car_eval = pd.read_csv("datasets/car_evaluation/car_bin.csv")
to_do_dict['car_evaluation'] = data_car_eval

data_mushroom = pd.read_csv("datasets/mushroom/agaricus_lepiota_bin.csv")
to_do_dict['mushroom'] = data_mushroom

data_nursery = pd.read_csv("datasets/nursery/nursery_bin.csv")
to_do_dict['nursery'] = data_nursery

data_seismic = pd.read_csv("datasets/seismic/seismic_bin.csv")
to_do_dict['seismic'] = data_seismic

data_spambase = pd.read_csv("datasets/spambase/spambase_bin.csv")
to_do_dict['spambase'] = data_spambase

data_wine = pd.read_csv("datasets/wine/wine_bin.csv")
to_do_dict['wine'] = data_wine

data_adult = pd.read_csv("datasets/adult/stacked.csv")
to_do_dict['adult'] = data_adult

data_banknote = pd.read_csv("datasets/banknote+authentication/banknote_bin.csv")
to_do_dict['banknote+authentication'] = data_banknote

data_chess = pd.read_csv("datasets/chess+king+rook+vs+king+pawn/kr-vs-kp_bin.csv")
to_do_dict['chess+king+rook+vs+king+pawn'] = data_chess

data_monk1 = pd.read_csv("datasets/monk1/monk1_bin.csv")
to_do_dict['monk1'] = data_monk1

data_monk2 = pd.read_csv("datasets/monk2/monk2_bin.csv")
to_do_dict['monk2'] = data_monk2

data_monk3 = pd.read_csv("datasets/monk3/monk3_bin.csv")
to_do_dict['monk3'] = data_monk3

"""
data_microbiome_taxa_easy = pd.read_csv("datasets/microbiome_taxa_counts_easy/microbiome_taxa_counts_easy_bin.csv")
to_do_dict['microbiome_taxa_easy'] = data_microbiome_taxa_easy
"""

for dataset_name, data in to_do_dict.items(): #.items() gives key, values

    dir_path = f'results/{dataset_name}_refactored_tree_pulp'

    features = data.drop(columns=['y'])
    targets = data['y']

    skf = StratifiedKFold(n_splits=folds_cross_val, shuffle=True, random_state=42)

    i=1 # index for fold number

    for train_idx, test_idx in skf.split(features, targets): #gives row indices

        # Create the directory if it doesn't exist
        os.makedirs(f'{dir_path}/fold{i}', exist_ok=True)
        
        with open(f'{dir_path}/fold{i}/fold{i}_time_{dataset_name}.txt', 'w') as f:
            pass  # This just creates/truncates the file

        with open(f'{dir_path}/fold{i}/fold{i}_tree_fit_time_{dataset_name}.txt', 'w') as f:
            pass  # This just creates/truncates the file

        print(dataset_name)
        print(i)
        
        features_train = features.iloc[train_idx]
        features_test = features.iloc[test_idx]
        targets_train = targets.iloc[train_idx]
        targets_test = targets.iloc[test_idx]

        start_time_pulp = time.time()

        tree = DecisionTree_rollOCT(max_depth=depth_rolling_tree, max_features = None)
        tree.fit(features_train, targets_train)
        
        r_test = tree.predict(features_test)
        result_test = pd.concat([targets_test, r_test['prediction']], axis=1)

        r_train = tree.predict(features_train)
        result_train = pd.concat([targets_train, r_train['prediction']], axis=1)

        end_time_pulp = time.time()

        with open(f'{dir_path}/pulp/fold{i}/fold{i}_time_{dataset_name}.txt', 'a') as f:
                f.write(f"{end_time_pulp - start_time_pulp}")

        with open(f'{dir_path}/fold{i}/fold{i}_tree_fit_time_{dataset_name}.txt', 'a') as f:
            f.write(str(tree.fit_time))

        with open(f'{dir_path}/fold{i}/{dataset_name}_result_test.csv', 'w') as f:
            f.write(str(result_test.to_csv()))

        with open(f'{dir_path}/fold{i}/{dataset_name}_result_train.csv', 'w') as f:
            f.write(str(result_train.to_csv()))

        #os.makedirs(f'refactored_tree_pickle/cvals_tree/{dataset_name}', exist_ok=True)

        #with open(f'refactored_tree_pickle/cvals_tree/{dataset_name}/fold{i}_tree_class.pkl', 'wb') as f: # 'wb' for write binary, rb for read binary
                #pickle.dump(tree, f)
        
        i+=1


