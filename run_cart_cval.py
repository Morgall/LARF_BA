import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

depth_tree = 8
criterion_loss = "gini"
#criterion_loss = "misclassification"

folds_cross_val = 10

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
        
        print(dataset_name)
        print(i)
        
        features_train = features.iloc[train_idx]
        features_test = features.iloc[test_idx]
        targets_train = targets.iloc[train_idx]
        targets_test = targets.iloc[test_idx]

        # Create the directory if it doesn't exist
        os.makedirs(f'{dir_path}/cart/fold{i}', exist_ok=True)

        with open(f'{dir_path}/cart/fold{i}/fold{i}_times_{dataset_name}.txt', 'w') as f:
            pass  # This just creates/truncates the file

        for depth in range(2, depth_tree+1):
            # Initialize the Decision Tree Classifier
            clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth, random_state=1)
            start_time_cart = time.time()
            clf.fit(features_train, targets_train)

            y_pred_test = clf.predict(features_test)
            y_pred_train = clf.predict(features_train)

            df_test = pd.DataFrame([], columns=['y', 'prediction'])
            df_test['y'] = targets_test
            df_test['prediction'] = y_pred_test

            df_train = pd.DataFrame([], columns=['y', 'prediction'])
            df_train['y'] = targets_train
            df_train['prediction'] = y_pred_train

            end_time_cart = time.time()

            # Create the directory if it doesn't exist
            #os.makedirs(f'{dir_path}/cart/fold{i}', exist_ok=True)

            with open(f'{dir_path}/cart/fold{i}/fold{i}_times_{dataset_name}.txt', 'a') as f:
                f.write(f"CART execution time for depth {depth} : {end_time_cart - start_time_cart} seconds\n")

            with open(f'{dir_path}/cart/fold{i}/depth{depth}_classification_{dataset_name}_test.csv', 'w') as f:
                f.write(df_test.to_csv())
                
            with open(f'{dir_path}/cart/fold{i}/depth{depth}_classification_{dataset_name}_train.csv', 'w') as f:
                f.write(df_train.to_csv())
        
        i+=1


