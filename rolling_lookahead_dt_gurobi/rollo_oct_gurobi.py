from rolling_lookahead_dt_gurobi.rolling_tree.rolling_optimize_gurobi import rolling_optimize_gurobi
from rolling_lookahead_dt_gurobi.oct.tree import *
from rolling_lookahead_dt_gurobi.oct.optimal_tree_gurobi import *
from helpers.helpers import preprocess_dataframes

import pandas as pd
import time


def run(train: pd.DataFrame,
        test: pd.DataFrame,
        target_label: str = "y",
        features: List = None,
        depth: int = 2,
        criterion: str = "gini",
        time_limit: int = 1800,
        big_m: int = 99):
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
    main_model_time = time.time()
    # generate model
    main_model = generate_model_gurobi(P=P, K=K, data=train, y_idx=0, big_m=big_m, criterion=criterion)
    # train model
    main_model = train_model_gurobi(model_dict=main_model, data=train, P=P)
    # predict model
    result_train = predict_model_gurobi(data=train, model_dict=main_model, P=P)

    misclassified_leafs = find_misclassification(df=result_train)

    result_test = predict_model_gurobi(data=test, model_dict=main_model, P=P)
    
    
    train_acc = len(result_train.loc[result_train["prediction"] == result_train["y"]]) / \
                len(result_train["y"])

    test_acc = len(result_test.loc[result_test["prediction"] == result_test["y"]]) / \
               len(result_test["y"])

    train = train.drop(["prediction", "leaf"], axis=1)
    test = test.drop(["prediction", "leaf"], axis=1)

    if depth > 2:
        result_dict, result_df_test_data, result_df_training_data = rolling_optimize_gurobi(predefined_model=main_model,
                                        train_data=train,
                                        test_data=test,
                                        main_depth=2,
                                        target_depth=depth,
                                        features=P,
                                        time_limit=time_limit,
                                        to_go_deep_nodes=misclassified_leafs,
                                        criterion=criterion)
          # add main model
    result_dict[2] = {
    "training_accuracy": train_acc,
    "test_accuracy": test_acc,
    "time": time.time() - main_model_time
    }

    if depth > 2:
        return result_dict, result_df_test_data, result_df_training_data
    else: 
        return result_dict, result_test[['y', 'prediction', 'leaf']], result_train[['y', 'prediction', 'leaf']]

