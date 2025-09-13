from forest.forest_refactored_tree.class_forest_refactored import CustomForestClassifier
import pandas as pd
import time
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from memory_profiler import memory_usage

def monitor_memory(func, args, interval=2, dir_path=None, test_cohort = None, run = 1):
    # Record memory usage samples while running func with args
    mem_usage = memory_usage((func, args), interval=interval, include_children=True, multiprocess=True)
    # Append samples to the given file
    with open(f'{dir_path}/test_{test_cohort}/fit_memory_run{run+1}.txt', 'a') as f:
        for usage in mem_usage:
            f.write(f"{usage}\n")


if __name__ == "__main__":

    target_label = "y"
    depth_tree = 5
    criterion_loss = "gini"
    #criterion_loss = "misclassification"
    cores_to_use = 30
    n_estimator = 500
    sub_features = 'sqrt'
    if sub_features == None:
        sub_name = 'None'
    else:
        sub_name = sub_features

    data = pd.read_csv("datasets/microbiome_taxa_counts_all/microbiome_taxa_counts_all_bin.csv")
    dataset_name = 'microbiome_taxa_all'

    print(data)

    cohorts = data['cohort_name'].copy()
    y = data['y'].copy()

    X = data.drop(data.columns[[0, 1]], axis=1) # removes cohort names and targets

    #--------------------rollOCT--------------------------

    dir_path = f'microbiome_data/taxa_all_logo/refactored_random_forest/n{n_estimator}_d{depth_tree}_{sub_name}'


    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X, y, cohorts):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        test_cohort = cohorts.iloc[test_idx].unique()
        print(f'\nTest cohort: {test_cohort}')



        # Create the directory if it doesn't exist
        os.makedirs(f'{dir_path}/test_{test_cohort}', exist_ok=True)

        for run in range(10): #account for randomness boostrapping and subfeature selection

            with open(f'{dir_path}/test_{test_cohort}/time_run{run+1}.txt', 'w') as f:
                pass  # This just creates/truncates the file

            with open(f'{dir_path}/test_{test_cohort}/fit_memory_run{run+1}.txt', 'w') as f:
                pass  # This just creates/truncates the file

            start_time_forest = time.time()
            forest = CustomForestClassifier(n_estimators=n_estimator, random_state=None, cores_to_use=cores_to_use, max_depth=depth_tree, max_features=sub_features)
            
            #forest.fit(X_train, y_train)
            monitor_memory(forest.fit, (X_train, y_train), interval=2, dir_path=dir_path, test_cohort = test_cohort, run = run)
            y_pred = forest.predict(X_test)
            result_test = pd.DataFrame({
                'y': y_test,
                'prediction': y_pred
            })

            end_time_forest= time.time()

            with open(f'{dir_path}/test_{test_cohort}/time_run{run+1}.txt', 'a') as f:
                f.write(f"Forest execution time for {n_estimator} estimators with depth {depth_tree} : {end_time_forest - start_time_forest} seconds\n")

            with open(f'{dir_path}/test_{test_cohort}/result_test_run{run+1}.csv', 'w') as f:
                f.write(str(result_test.to_csv()))
        
#--------------------sklearn--------------------------

    dir_path = f'microbiome_data/taxa_all_logo/sklearn_random_forest/n{n_estimator}_d{depth_tree}_{sub_name}'


    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X, y, cohorts):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        test_cohort = cohorts.iloc[test_idx].unique()
        print(f'\nTest cohort: {test_cohort}')



        # Create the directory if it doesn't exist
        os.makedirs(f'{dir_path}/test_{test_cohort}', exist_ok=True)

        for run in range(10): #account for randomness boostrapping and subfeature selection

            with open(f'{dir_path}/test_{test_cohort}/time_run{run+1}.txt', 'w') as f:
                pass  # This just creates/truncates the file

            start_time_forest = time.time()
            forest = RandomForestClassifier(n_estimators=n_estimator, random_state=None, max_depth=depth_tree, max_features=sub_features, bootstrap=True)
            
            forest.fit(X_train, y_train)
            y_pred = forest.predict(X_test)
            result_test = pd.DataFrame({
                'y': y_test,
                'prediction': y_pred
            })

            end_time_forest= time.time()

            with open(f'{dir_path}/test_{test_cohort}/time_run{run+1}.txt', 'a') as f:
                f.write(f"Forest execution time for {n_estimator} estimators with depth {depth_tree} : {end_time_forest - start_time_forest} seconds\n")

            with open(f'{dir_path}/test_{test_cohort}/result_test_run{run+1}.csv', 'w') as f:
                f.write(str(result_test.to_csv()))




