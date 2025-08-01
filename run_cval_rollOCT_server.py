import pandas as pd
from rolling_lookahead_dt_pulp import rollo_oct_pulp
from sklearn.model_selection import StratifiedKFold
import time
import os

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
#criterion_loss = "misclassification"

folds_cross_val = 10

#dataset_name = 'adult' # folder in 'results' will be named after it, results/dataset_name contains result .txt and .csv
#dir_path = f'results/{dataset_name}'

to_do_dict = dict() # add datasets to be run into this dict and choose key as dataset name


#data_test = pd.read_csv("datasets/example_datasets/stacked.csv")
#to_do_dict['test'] = data_test

#data_breast_cancer = pd.read_csv("datasets/breast+cancer+wisconsin+diagnostic/wdbc_bin.csv")
#to_do_dict['breast+cancer+wisconsin+diagnostic'] = data_breast_cancer

#data_car_eval = pd.read_csv("datasets/car_evaluation/car_bin.csv")
#to_do_dict['car_evaluation'] = data_car_eval

#data_mushroom = pd.read_csv("datasets/mushroom/agaricus_lepiota_bin.csv")
#to_do_dict['mushroom'] = data_mushroom

#data_nursery = pd.read_csv("datasets/nursery/nursery_bin.csv")
#to_do_dict['nursery'] = data_nursery

#data_seismic = pd.read_csv("datasets/seismic/seismic_bin.csv")
#to_do_dict['seismic'] = data_seismic

#data_spambase = pd.read_csv("datasets/spambase/spambase_bin.csv")
#to_do_dict['spambase'] = data_spambase

#data_wine = pd.read_csv("datasets/wine/wine_bin.csv")
#to_do_dict['wine'] = data_wine

#data_adult = pd.read_csv("datasets/adult/stacked.csv")
#to_do_dict['adult'] = data_adult

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


for dataset_name, data in to_do_dict.items(): #.items() gives key, values

    dir_path = f'results/{dataset_name}'

    features = data.drop(columns=['y'])
    targets = data['y']

    skf = StratifiedKFold(n_splits=folds_cross_val, shuffle=True, random_state=42)

    i=1 # index for fold number

    for train_idx, test_idx in skf.split(features, targets): #gives row indices
        
        with open(f'{dir_path}/cart/fold{i}/fold{i}_times_{dataset_name}.txt', 'w') as f:
            pass  # This just creates/truncates the file

        print(dataset_name)
        print(i)
        
        features_train = features.iloc[train_idx]
        features_test = features.iloc[test_idx]
        targets_train = targets.iloc[train_idx]
        targets_test = targets.iloc[test_idx]

        stacked_train = pd.concat([targets_train, features_train], axis=1, ignore_index=False)
        stacked_test = pd.concat([targets_test, features_test],axis=1, ignore_index=False)

        feature_columns = stacked_train.columns[1:] #assuming labels are in first column, ensured trough move_targets_to_front_and_rename()
        start_time_pulp = time.time()
        result_dict_pulp =rollo_oct_pulp.run(train=stacked_train, test=stacked_test, target_label="y", features=feature_columns, depth=depth_rolling_tree, criterion=criterion_loss)
        end_time_pulp = time.time()

        #for depth in range(2,depth_rolling_tree+1):
            #print(f'10 fold #{i}' + str(result_dict_pulp[depth]))

        #print(f"Pulp execution time; 10 fold #{i}; for depth {depth_rolling_tree} : {end_time_pulp - start_time_pulp} seconds\n")

        # Create the directory if it doesn't exist
        os.makedirs(f'{dir_path}/pulp/fold{i}', exist_ok=True)

        for depth in range(2,depth_rolling_tree+1):
            with open(f'{dir_path}/pulp/fold{i}/fold{i}_acc_times_{dataset_name}.txt', 'a') as f:
                f.write(str(depth) + ': ' + str(result_dict_pulp[depth]) + "\n")

        with open(f'{dir_path}/pulp/fold{i}/fold{i}_acc_times_{dataset_name}.txt', 'a') as f:
                f.write(f"Pulp execution time for depth {depth_rolling_tree} : {end_time_pulp - start_time_pulp} seconds\n")

        for depth in range(2,depth_rolling_tree+1):
            with open(f'{dir_path}/pulp/fold{i}/depth{depth}_classification_{dataset_name}_test.csv', 'w') as f:
                f.write(str(result_dict_pulp['tree'][depth]['test'].to_csv()))
            with open(f'{dir_path}/pulp/fold{i}/depth{depth}_classification_{dataset_name}_train.csv', 'w') as f:
                f.write(str(result_dict_pulp['tree'][depth]['train'].to_csv()))
        
        i+=1


