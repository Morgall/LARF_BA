from forest.forest_refactored_tree.class_forest_refactored import CustomForestClassifier
import pandas as pd
import time
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

    target_label = "y"
    depth_tree_list = [10]
    criterion_loss = "gini"
    #criterion_loss = "misclassification"
    folds_cross_val = 5
    cores_to_use = 80
    n_estimators = 500
    sub_features = 'sqrt'
    if sub_features == None:
        sub_name = 'None'
    else:
        sub_name = sub_features

    to_do_dict = dict() # add datasets to be run into this dict and choose key as dataset name


    data_microbiome_taxa_easy = pd.read_csv("datasets/microbiome_taxa_counts_easy/microbiome_taxa_counts_easy_bin.csv")
    to_do_dict['microbiome_taxa_easy'] = data_microbiome_taxa_easy

    # -------------------------CART-----------------------------------

 
    
    start_time_run_all_rollOCTforest = time.time()
    
    for dataset_name, data in to_do_dict.items(): #.items() gives key, values

        features = data.drop(columns=['y'])
        targets = data['y']

        skf = StratifiedKFold(n_splits=folds_cross_val, shuffle=True, random_state=42)

        for depth in depth_tree_list:

            dir_path = f'microbiome_data/taxa_easy/refactored_random_forest/n{n_estimators}_d{depth}_{sub_name}'

            for run in range(5): #account for randomness boostrapping and subfeature selection

                i=1 # index for fold number

                for train_idx, test_idx in skf.split(features, targets): #gives row indices

                    # Create the directory if it doesn't exist
                    os.makedirs(f'{dir_path}/fold{i}', exist_ok=True)

                    with open(f'{dir_path}/fold{i}/fold{i}_time_run{run+1}.txt', 'w') as f:
                        pass  # This just creates/truncates the file

                    print(dataset_name)
                    print(i)
                    
                    features_train = features.iloc[train_idx]
                    features_test = features.iloc[test_idx]
                    targets_train = targets.iloc[train_idx]
                    targets_test = targets.iloc[test_idx]

                    #stacked_train = pd.concat([targets_train, features_train], axis=1, ignore_index=False)
                    #stacked_test = pd.concat([targets_test, features_test],axis=1, ignore_index=False)


                    start_time_forest = time.time()
                    forest = CustomForestClassifier(n_estimators=n_estimators, random_state=None, cores_to_use=cores_to_use, max_depth=depth, max_features=sub_features)
                    
                    forest.fit(features_train, targets_train)
                    y_pred = forest.predict(features_test)
                    result_test = pd.DataFrame({
                        'y': targets_test,
                        'prediction': y_pred
                    })

                    end_time_forest= time.time()

                    with open(f'{dir_path}/fold{i}/fold{i}_time_run{run+1}.txt', 'a') as f:
                        f.write(f"Forest execution time for {n_estimators} estimators with depth {depth} : {end_time_forest - start_time_forest} seconds\n")

                    with open(f'{dir_path}/fold{i}/result_test_run{run+1}.csv', 'w') as f:
                        f.write(str(result_test.to_csv()))
                    i+=1
        
            end_time_run_all_rollOCTforest = time.time()

            with open(f'{dir_path}/total_runtime_all_folds.txt', 'w') as f:
                pass

            with open(f'{dir_path}/total_runtime_all_folds.txt', 'a') as f:
                f.write(f"Total runtime over all runs and folds: {end_time_run_all_rollOCTforest - start_time_run_all_rollOCTforest} seconds\n")

# -------------------------CART-----------------------------------

    start_time_run_all_sklearn = time.time()
    
    for dataset_name, data in to_do_dict.items(): #.items() gives key, values

        features = data.drop(columns=['y'])
        targets = data['y']

        skf = StratifiedKFold(n_splits=folds_cross_val, shuffle=True, random_state=42)

        for depth in depth_tree_list:

            dir_path = f'microbiome_data/taxa_easy/sklearn_random_forest/n{n_estimators}_d{depth}_{sub_name}'

            for run in range(5): #account for randomness boostrapping and subfeature selection

                i=1 # index for fold number

                for train_idx, test_idx in skf.split(features, targets): #gives row indices

                    # Create the directory if it doesn't exist
                    os.makedirs(f'{dir_path}/fold{i}', exist_ok=True)

                    with open(f'{dir_path}/fold{i}/fold{i}_time_run{run+1}.txt', 'w') as f:
                        pass  # This just creates/truncates the file

                    print(dataset_name)
                    print(i)
                    
                    features_train = features.iloc[train_idx]
                    features_test = features.iloc[test_idx]
                    targets_train = targets.iloc[train_idx]
                    targets_test = targets.iloc[test_idx]

                    #stacked_train = pd.concat([targets_train, features_train], axis=1, ignore_index=False)
                    #stacked_test = pd.concat([targets_test, features_test],axis=1, ignore_index=False)


                    start_time_forest = time.time()
                    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=None, max_depth=depth, max_features=sub_features, bootstrap=True)

                    forest.fit(features_train, targets_train)
                    y_pred = forest.predict(features_test)
                    result_test = pd.DataFrame({
                        'y': targets_test,
                        'prediction': y_pred
                    })

                    end_time_forest= time.time()

                    with open(f'{dir_path}/fold{i}/fold{i}_time_run{run+1}.txt', 'a') as f:
                        f.write(f"Forest execution time for {n_estimators} estimators with depth {depth} : {end_time_forest - start_time_forest} seconds\n")

                    with open(f'{dir_path}/fold{i}/result_test_run{run+1}.csv', 'w') as f:
                        f.write(str(result_test.to_csv()))
                    i+=1
        
            end_time_run_all_sklearn = time.time()
            with open(f'{dir_path}/total_runtime_all_folds.txt', 'w') as f:
                pass

            with open(f'{dir_path}/total_runtime_all_folds.txt', 'a') as f:
                f.write(f"Total runtime over all runs and folds: {end_time_run_all_sklearn - start_time_run_all_sklearn} seconds\n")



