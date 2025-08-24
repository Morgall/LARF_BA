import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
#from sklearn.base import BaseEstimator, ClassifierMixin, clone

from forest.forest_subset_features.edited_rolling_optimize import rolling_optimize
from rolling_lookahead_dt_pulp.oct.tree import *
from rolling_lookahead_dt_pulp.oct.optimal_tree_pulp import predict_model_pulp
from helpers.helpers import preprocess_dataframes

from forest.forest_subset_features.edited_model import generate_model_tree, train_model

# was hiermit eben nicht geht ist, dass man auf Trainingsdaten trainiert (was einem das reine Modell geben sollte). Dabei werden aber leider gleichzeitig
# die Testdaten auf diesen Modell predicted
# Das Resultat ist also, dass man nicht andere Testdaten auf dem fertigen modell testen kann

class CustomTreeWrapper:
    def __init__(self, train_data, test_data, depth=None, criterion='gini', target_label=None, features=None, time_limit = 1800, big_m = 99, random_state=None):
        self.depth = depth
        self.criterion = criterion
        self.test_data = test_data
        self.train_data = train_data
        self.target_label = target_label
        self.features = features
        self.time_limit = time_limit
        self.big_m = big_m
        self.random_state = random_state
        self.construct()

    def construct(self):
        train, test = preprocess_dataframes( #./rollo_oct/utils/helpers.py
        train_df = self.train_data,
        test_df = self.test_data,
        target_label = self.target_label,
        features = self.features)

        df = pd.concat([train, test])
        self.all_features_list = [int(i) for i in list(train.loc[:, train.columns != 'y'].columns)] #used to be P
        self.features_orig_dataset = len(self.all_features_list)
        train.columns = ["y", *self.all_features_list]
        test.columns = ["y", *self.all_features_list]
        self.K = sorted(list(set(df.y)))

        self.result_dict = {} #adding dict to store solutions for every level
        self.result_dict['tree'] = {}
        self.result_dict['tree'][2] = {}

        # --- Choose subset of features for THIS TREE/BLOCK (random forest style) ---
        amount_X_consider = max(1, int(np.sqrt(self.features_orig_dataset)))
        rng = np.random.RandomState(self.random_state)
        subset = rng.choice(self.all_features_list, size=amount_X_consider, replace=False)
        subset = list(subset)
        #Use only the selected features for this tree
        train = train[['y'] + subset]
        test = test[['y'] + subset]

        local_P = list(range(1, len(subset) + 1))  # [1,2,3,...]
        local_to_global = dict(zip(local_P, subset))
        global_to_local = dict(zip(subset, local_P))

        self.P = local_P
        self.feature_idx_map = local_to_global
        
        # generate model
        self.main_model = generate_model_tree(P=self.P, K=self.K, data=train, y_idx=0, big_m=self.big_m, criterion=self.criterion)
    
    def fit(self, X, y):

        self.train_data = pd.concat([y, X], axis=1, ignore_index=False)

        train, test = preprocess_dataframes( #./rollo_oct/utils/helpers.py
                                            train_df = self.train_data,
                                            test_df = self.test_data,
                                            target_label = self.target_label,
                                            features = self.features)
        
        self.processed_train = train
        self.processed_test = test
        
        self.main_model = train_model(model_dict=self.main_model, data=self.processed_train, P=self.P)

        self.result_dict['tree'][2]['trained_dict'] = self.main_model

        # predict model
        result_train = predict_model_pulp(data=self.processed_train, model_dict=self.main_model, P=self.P)

        misclassified_leafs = find_misclassification(df=result_train)

        result_test = predict_model_pulp(data=self.processed_test, model_dict=self.main_model, P=self.P)
        
        
        train_acc = len(result_train.loc[result_train["prediction"] == result_train["y"]]) / \
                    len(result_train["y"])

        test_acc = len(result_test.loc[result_test["prediction"] == result_test["y"]]) / \
                len(result_test["y"])
        
        
        self.result_dict['tree'][2]['train'] = result_train[['y', 'prediction', 'leaf']]
        self.result_dict['tree'][2]['test'] = result_test[['y', 'prediction', 'leaf']]

        self.result_dict[2] = {
        "training_accuracy": train_acc,
        "test_accuracy": test_acc
        }

        train = train.drop(["prediction", "leaf"], axis=1)
        test = test.drop(["prediction", "leaf"], axis=1)

        if self.depth > 2:
            self.result_dict = rolling_optimize(predefined_model=self.main_model,
                                            train_data=train,
                                            test_data=test,
                                            original_training_dataset = self.processed_train,
                                            features_orig_dataset = self.features_orig_dataset,
                                            main_depth=2,
                                            target_depth=self.depth,
                                            features=self.P,
                                            time_limit=self.time_limit,
                                            to_go_deep_nodes=misclassified_leafs,
                                            result_dict=self.result_dict,
                                            criterion=self.criterion)

        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        #print(X)

        model_dict = self.result_dict['tree'][self.depth]['trained_dict']

        dummy = pd.DataFrame({'y': [None]*len(X)}, index=X.index)

        test = pd.concat([dummy, X], axis=1)

        #print(test)

        res = predict_model_pulp(data=test, model_dict=model_dict, P=self.P)

        #print(res)
        
        preds = res['prediction']
        if preds is None:
            raise RuntimeError("No stored predictions found. Run fit first.")
        
        #check = self.result_dict['tree'][self.depth]['test']
        #check = check.drop(columns=['y', 'leaf'])
        #print(check)

        #print(preds.equals(check['prediction']))
        return preds