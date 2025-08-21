import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
#from sklearn.base import BaseEstimator, ClassifierMixin, clone

from rolling_lookahead_dt_pulp.rolling_tree.rolling_optimize_pulp import rolling_optimize_pulp
from rolling_lookahead_dt_pulp.oct.tree import *
from rolling_lookahead_dt_pulp.oct.optimal_tree_pulp import *
from helpers.helpers import preprocess_dataframes

# was hiermit eben nicht geht ist, dass man auf Trainingsdaten trainiert (was einem das reine Modell geben sollte). Dabei werden aber leider gleichzeitig
# die Testdaten auf diesen Modell predicted
# Das Resultat ist also, dass man nicht andere Testdaten auf dem fertigen modell testen kann

class CustomTreeWrapper:
    def __init__(self, train_data, test_data, depth=None, criterion='gini', target_label=None, features=None, time_limit = 1800, big_m = 99):
        self.depth = depth
        self.criterion = criterion
        self.test_data = test_data
        self.train_data = train_data
        self.target_label = target_label
        self.features = features
        self.time_limit = time_limit
        self.big_m = big_m
        self.construct()

    def construct(self):
        train, test = preprocess_dataframes( #./rollo_oct/utils/helpers.py
        train_df = self.train_data,
        test_df = self.test_data,
        target_label = self.target_label,
        features = self.features)

        df = pd.concat([train, test])
        self.P = [int(i) for i in
            list(train.loc[:, train.columns != 'y'].columns)]
        train.columns = ["y", *self.P]
        test.columns = ["y", *self.P]
        self.K = sorted(list(set(df.y)))

        self.result_dict = {} #adding dict to store solutions for every level
        self.result_dict['tree'] = {}
        self.result_dict['tree'][2] = {}
        
        # generate model
        self.main_model = generate_model_pulp(P=self.P, K=self.K, data=train, y_idx=0, big_m=self.big_m, criterion=self.criterion)
    
    def fit(self, X, y):

        self.train_data = pd.concat([y, X], axis=1, ignore_index=False)

        train, test = preprocess_dataframes( #./rollo_oct/utils/helpers.py
                                            train_df = self.train_data,
                                            test_df = self.test_data,
                                            target_label = self.target_label,
                                            features = self.features)
        
        self.P = [int(i) for i in 
            list(train.loc[:, train.columns != 'y'].columns)]
        
        self.main_model = train_model_pulp(model_dict=self.main_model, data=train, P=self.P)

        self.result_dict['tree'][2]['trained_dict'] = self.main_model

        # predict model
        result_train = predict_model_pulp(data=train, model_dict=self.main_model, P=self.P)

        misclassified_leafs = find_misclassification(df=result_train)

        result_test = predict_model_pulp(data=test, model_dict=self.main_model, P=self.P)
        
        
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
            self.result_dict = rolling_optimize_pulp(predefined_model=self.main_model,
                                            train_data=train,
                                            test_data=test,
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