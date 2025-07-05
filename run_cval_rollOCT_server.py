import pandas as pd
from rolling_lookahead_dt_pulp import rollo_oct_pulp
from sklearn.model_selection import StratifiedKFold
import time
import os


target_label = "y"
depth_rolling_tree = 8
criterion_loss = "gini"
#criterion_loss = "misclassification"

folds_cross_val = 10

dataset_name = 'adult' # folder in 'results' will be named after it, results/dataset_name contains result .txt and .csv
dir_path = f'results/{dataset_name}'



if dataset_name == 'test':
    data = pd.read_csv("datasets/example_datasets/stacked.csv")

if dataset_name == 'adult':
    data = pd.read_csv("datasets/adult/stacked.csv")



features = data.drop(columns=['y'])
targets = data['y']

skf = StratifiedKFold(n_splits=folds_cross_val, shuffle=True, random_state=42)

i=1 # index for fold number

for train_idx, test_idx in skf.split(features, targets): #gives row indices
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




