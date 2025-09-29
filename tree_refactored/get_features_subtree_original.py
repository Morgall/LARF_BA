import numpy as np
import pandas as pd
from pulp import *
import copy

# adapted from original
def preprocess_dataframes(train_df: pd.DataFrame, target_label: str, features: list):
    """
    Rearranges the DataFrames such that the target label becomes the first column,
    and feature names are converted into ordinal numbers.

    Args:
    - train_df: pandas DataFrame containing the training data
    - test_df: pandas DataFrame containing the test data
    - target_label: string representing the target label
    - features: list of strings representing feature names

    Returns:
    - pd.DataFrame: preprocessed training DataFrame
    - pd.DataFrame: preprocessed test DataFrame
    """

    # Move target label to the first column for both train and test DataFrames
    if target_label in train_df.columns:
        train_target_idx = train_df.columns.get_loc(target_label) # find the integer position (index) of the column named target_label within the DataFrame train_df
        train_df_columns = list(train_df.columns)
        train_df_columns = [train_df_columns[train_target_idx]] + train_df_columns[:train_target_idx] + train_df_columns[train_target_idx + 1:] # new arrangement
        train_df = train_df[train_df_columns] # returns a new DataFrame containing only those columns, arranged in the order specified by the list, panmda reordering

    # Rename features to ordinal numbers for both train and test DataFrames
    train_df.rename(columns={feature: str(i) for i, feature in enumerate(features, start=1)}, inplace=True) #  DataFrame columns renaming ['y', 'age', 'income', 'score', ...] to ['y', '1', '2', '3', ...]

    return train_df

def gini_index(arr: np.array,
               instance_size: int,
               K: list,
               y_idx: int = 0,
               weighted: bool = True):
    """

    :param arr:
    :param instance_size:
    :param K:
    :param y_idx:
    :param weighted:
    :return:
    """
    #if len(arr) == 0:
        #return 99999999  # Return a very large value to penalize empty splits
    
    sum_ = 0
    for k in K:
        sum_ += np.power(len(arr[np.where(arr[:, y_idx] == k)]) / len(arr), 2)
    sum_ = 1 - sum_
    if weighted:
        sum_ = (len(arr) / instance_size) * sum_
    return sum_


def calculate_gini_modified(data: pd.DataFrame,
                   P: list,
                   K: list,
                   nodes: dict) -> dict:
    """

    :param data: pd.DataFrame — The dataset
    :param P: list Indices of features to consider
    :param K: list Unique class labels
    :param nodes: dict — Contains: "leaf_nodes": list of leaf node identifiers. leaf_nodes_path": dict mapping each leaf node to a path (list of values for features).
    :return:
    """

    df_arr = np.array(data)
    n = len(data) # total number of instances.
    gini_dict = dict()
    for leaf_ in nodes["leaf_nodes"]:
        temp = dict() # store Gini values for this leaf
        first_var = nodes["leaf_nodes_path"][leaf_][0] #{'leaf_nodes': [4, 5, 6, 7], 'leaf_nodes_path': {4: [1, 1], 5: [1, 0], 6: [0, 1], 7: [0, 0]}}
        second_var = nodes["leaf_nodes_path"][leaf_][1] #first_var, second_var - the first two values in the path to this leaf (used as feature values for filtering)
        for feature_i in P:
            arr = df_arr[np.where((df_arr[:, feature_i] == first_var))] #Selects rows where feature_i equals first_var

            for feature_j in P:
                arr_2 = arr[np.where(arr[:, feature_j] == second_var)] #Further filters to rows where feature_j equals second_var
                if len(arr_2) > 0: #Calculate Gini index for arr_2 (so for all rows matching (1, 0; having/not having feature i) the decision variables)
                    temp[feature_i, feature_j] = gini_index(arr=arr_2,
                                                            instance_size=n,
                                                            K=K,
                                                            weighted=True)
        gini_dict[leaf_] = copy.deepcopy(temp)
        del temp
        del arr

    return gini_dict

def generate_nodes(depth: int) -> list:
    """

    :param depth:
    :return:
    """
    nodes = list(range(1, int(round(2 ** (depth + 1)))))
    parent_nodes = nodes[0: 2 ** (depth + 1) - 2 ** depth - 1]
    leaf_nodes = nodes[-2 ** depth:]
    return parent_nodes, leaf_nodes

# adapted from original
def solve_features_subtree(P: list, K: list, data: pd.DataFrame, y_idx: int = 0, big_m=99):
    

    """
    variable_features = [
    feature for feature in P
    if len(data[str(lookup_dict[feature])].unique()) > 1
    ]

    # Use only these variable features for the optimization
    P_filtered = variable_features
    """

    leaf_nodes_path = {4: [1, 1], #dict for all leafes in depth 2 tree and the respective split conditions
                       5: [1, 0],
                       6: [0, 1],
                       7: [0, 0]}
    depth = 2
    parent_nodes, leaf_nodes = generate_nodes(depth) # (for depth 2) returns [1,2,3] and [4,5,6,7]

    nodes = dict()
    nodes["leaf_nodes"] = leaf_nodes
    nodes["leaf_nodes_path"] = leaf_nodes_path


    logging.info("Calculating gini..")
    coef_dict = calculate_gini_modified(data=data,
                                P=P,
                                K=K,
                                nodes=nodes)

    # init model
    model = LpProblem("RollOCT", LpMinimize) # Sets the objective to be minimized

    # # x[i,j] and y[i,k] as binary variables
    x = dict()
    for i in P:
        for j in P:
            x[i, j] = LpVariable(f'x[{i},{j}]', cat='Binary') #creates variable
            

    y = dict()
    for i in P:
        for k in P:
            y[i, k] = LpVariable(f'y[{i},{k}]', cat='Binary')


    # Constraint 1: Exactly one (i,j) pair is selected
    model += lpSum(x[i,j] for i in P for j in P) == 1, "C1b" # Ensures that exactly one combination of features (i, j) is selected for the first two splits (criteria (1b))
    # Constraint 2: Exactly one (i,k) pair is selected
    model += lpSum(y[i,k] for i in P for k in P) == 1, "C1c"  # implements criteria (1c) same way as above
    # Constraint 3
    for i in P:
        model += lpSum(x[i,j] for j in P) == lpSum(y[i,k] for k in P), f"C1d_{i}" # Links the x and y variables, ensuring that for each feature i, the sum of x[i, j] across j equals the sum of y[i, k] across k. This ensures consistency between the splits (criteria (1d))

    # Constraint 4
    # add big m; Acts as a penalty for invalid splits (if a combination is not present in coef_dict, it uses big_m as a large penalty)

# implements criteria (1a)
    obj = lpSum(
        (coef_dict[4].get((i, j), big_m) + coef_dict[5].get((i, j), big_m)) *
        x[i, j]
        for i in P
        for j in P) + \
          lpSum((coef_dict[6].get((i, k), big_m) + coef_dict[7].get((i, k), big_m)) *
                   y[i, k]
                   for i in P
                   for k in P)

    model += obj, "Objective"

    model_dict = {
        'model': model,
        'params': {
            'var_x': x,
            'var_y': y,
            'y_idx': 0
        },
        'nodes': {
            'leaf_nodes': leaf_nodes,
            'parent_nodes': parent_nodes,
            "leaf_nodes_path": leaf_nodes_path,
        },
        'depth': depth,
        "P": P,
        "K": K
    }

    nodes = model_dict['nodes']
    params = model_dict['params'] #is reference to the variables in the model; references them directly. They cary solution after m.optimize()
    m = model_dict['model']
    

    m.solve(solver=PULP_CBC_CMD(msg=False)) # solves our optimization model

    #Check Optimization Status

    if LpStatus[m.status] == "Optimal": 
        logging.info(f'Optimal objective: {m.objective.value()}') # OPTIMAL: Solution found.
        status = m.status
    elif LpStatus[m.status] == "Infeasible": # INFEASIBLE: No solution exists
        logging.info("Model is infeasible")
        return
    elif LpStatus[m.status] == "Unbounded": # UNBOUNDED: The objective can be improved infinitely
        logging.info("Model is unbounded")
        return
    #elif m.status == 9 and m.objVal != -math.inf: #Status 9: Time limit reached (if objective is not -∞)
        #logging.info('Time Limit termination')
        #return
    else:
        logging.info(f'Optimization status {LpStatus[m.status]}')
        return

    logging.info(f"Objective Value: {m.objective.value()}")
    
    
    # Arrange/EXTRACT decision variables WITH SOLUTIONS => dict of solutions (optimization for splits)
    #x and y: Dictionaries mapping pairs (i, j) (key) to the value of the decision variables var_x and var_y (data) from the solution.
    # variables are defined as binary, so they are 0 or 1
    x = {
        (i, j): params['var_x'][i, j].value() #Accesses the attribute of the variable params['var_x'][i, j]; In Gurobi, .X gives the solution value of the variable after optimization
        for i in P for j in P
    }
    y = {
        (i, j): params['var_y'][i, j].value()
        for i in P for j in P
    }



    # Identify Selected Features
    # Sets 'first_level', 'left_second_level', and 'right_second_level' variables => wir bekommen also die Identifikation für die features in jedem Split
    # wir wollen hier über alle variablen die Variablen finden, die mit 1 belegt sind, also laut solver gewählt werden sollen. Aus diesen entnehmen wir die features
    for i in P:
        for j in P:
            if x[i, j] > 0: # Finds which features are selected (where x (or y) is positive (1) ).
                first_level = i
                left_second_level = j
                logging.info(
                    f"First Level Feature: {i} & Second Level Left Feature: "
                    f"{j}, {x[i, j]}={x[i, j]}")
            if y[i, j] > 0:
                right_second_level = j
                logging.info(
                    f"First Level Feature: {i} & Second Level Right Feature:"
                    f" {j}, {y[i, j]}={y[i, j]}")
                
    del model_dict

    root_feature = first_level
    node1_feature = left_second_level
    node2_feature =  right_second_level


    """    
    print("\nSelected features subtree:")
    print(f"Root Node Feature: {root_feature}")
    print(f"No (0) instance child feature: {node1_feature}")
    print(f"Yes (1) instance child feature: {node2_feature}")
    """
    return root_feature, node1_feature, node2_feature