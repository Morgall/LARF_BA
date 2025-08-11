from rolling_lookahead_dt_pulp.rolling_tree.rolling_optimize_pulp import rolling_optimize_pulp
from rolling_lookahead_dt_pulp.oct.tree import *
from rolling_lookahead_dt_pulp.oct.optimal_tree_pulp import *
from helpers.helpers import preprocess_dataframes

import pickle
import os

import pandas as pd
import time


def run(train: pd.DataFrame,
        test: pd.DataFrame,
        target_label: str = "y",
        features: List = None,
        depth: int = 2,
        criterion: str = "gini",
        time_limit: int = 1800,
        big_m: int = 99,
        big_dataset = False):
    """

    :param target_label:
    :param features:
    :param train:
    :param test:
    :param depth:
    :param criterion:
    :param time_limit:
    :param big_m:
    :return:
    """
    current_depth = 2

    train, test = preprocess_dataframes( #./rollo_oct/utils/helpers.py
        train_df=train,
        test_df=test,
        target_label=target_label,
        features=features)

    df = pd.concat([train, test])
    P = [int(i) for i in
         list(train.loc[:, train.columns != 'y'].columns)]
    train.columns = ["y", *P]
    test.columns = ["y", *P]
    K = sorted(list(set(df.y)))

    result_dict = {} #adding dict to store solutions for every level
    result_dict['tree'] = {}
    result_dict['tree'][2] = {}

    main_model_time = time.time()
    # generate model
    main_model = generate_model_pulp(P=P, K=K, data=train, y_idx=0, big_m=big_m, criterion=criterion)
    # train model
    main_model = train_model_pulp(model_dict=main_model, data=train, P=P)

    result_dict['tree'][2]['trained_dict'] = main_model

    # predict model
    result_train = predict_model_pulp(data=train, model_dict=main_model, P=P)

    misclassified_leafs = find_misclassification(df=result_train)

    result_test = predict_model_pulp(data=test, model_dict=main_model, P=P)
    
    
    train_acc = len(result_train.loc[result_train["prediction"] == result_train["y"]]) / \
                len(result_train["y"])

    test_acc = len(result_test.loc[result_test["prediction"] == result_test["y"]]) / \
               len(result_test["y"])
    
    
    result_dict['tree'][2]['train'] = result_train[['y', 'prediction', 'leaf']]
    result_dict['tree'][2]['test'] = result_test[['y', 'prediction', 'leaf']]

    result_dict[2] = {
    "training_accuracy": train_acc,
    "test_accuracy": test_acc,
    "time": time.time() - main_model_time
    }

    train = train.drop(["prediction", "leaf"], axis=1)
    test = test.drop(["prediction", "leaf"], axis=1)

    if depth > 2:
        result_dict = rolling_optimize_pulp(predefined_model=main_model,
                                        train_data=train,
                                        test_data=test,
                                        main_depth=2,
                                        target_depth=depth,
                                        features=P,
                                        time_limit=time_limit,
                                        to_go_deep_nodes=misclassified_leafs,
                                        result_dict=result_dict,
                                        criterion=criterion)

    return result_dict

