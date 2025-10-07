from forest.forest_refactored_tree.class_forest_refactored import CustomForestClassifier
import pandas as pd
import time
import os
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    target_label = "y"
    depth_rolling_tree = 8
    criterion_loss = "gini"
    #criterion_loss = "misclassification"
    folds_cross_val = 5
    cores_to_use = 80
    number_of_estimators = 50

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

    start_time_run = time.time()
    
    for dataset_name, data in to_do_dict.items(): #.items() gives key, values

        dir_path = f'results/refactored_random_forest/{dataset_name}/n_est{number_of_estimators}_depth{depth_rolling_tree}_2'

        features = data.drop(columns=['y'])
        targets = data['y']

        skf = StratifiedKFold(n_splits=folds_cross_val, shuffle=True, random_state=42)

        for run in range(5): #account for randomness boostrapping and subfeature selection

            i=1 # index for fold number

            for train_idx, test_idx in skf.split(features, targets): #gives row indices

                # Create the directory if it doesn't exist
                os.makedirs(f'{dir_path}/fold{i}', exist_ok=True)

                with open(f'{dir_path}/fold{i}/fold{i}_time_{dataset_name}_run{run+1}.txt', 'w') as f:
                    pass  # This just creates/truncates the file

                print(dataset_name)
                print(i)
                
                features_train = features.iloc[train_idx]
                features_test = features.iloc[test_idx]
                targets_train = targets.iloc[train_idx]
                targets_test = targets.iloc[test_idx]

                stacked_train = pd.concat([targets_train, features_train], axis=1, ignore_index=False)
                #stacked_test = pd.concat([targets_test, features_test],axis=1, ignore_index=False)


                start_time_forest = time.time()
                forest = CustomForestClassifier(n_estimators=number_of_estimators, random_state=None, cores_to_use=cores_to_use, max_depth=8, max_features='sqrt')
                
                forest.fit(features_train, targets_train)
                y_pred = forest.predict(features_test)
                result_test = pd.DataFrame({
                    'y': targets_test,
                    'prediction': y_pred
                })

                end_time_forest= time.time()

                with open(f'{dir_path}/fold{i}/fold{i}_time_{dataset_name}_run{run+1}.txt', 'a') as f:
                    f.write(f"Forest execution time for {number_of_estimators} estimators with depth {depth_rolling_tree} : {end_time_forest - start_time_forest} seconds\n")

                with open(f'{dir_path}/fold{i}/{dataset_name}_result_test_run{run+1}.csv', 'w') as f:
                    f.write(str(result_test.to_csv()))
                i+=1
    
    end_time_run = time.time()
    with open(f'total_runtime_all_datasets_old_forest.txt', 'w') as f:
        f.write(f"Total runtime: {end_time_run - start_time_run} seconds\n")


