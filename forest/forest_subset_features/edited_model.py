import pandas as pd
import numpy as np
from pulp import *
import time

from rolling_lookahead_dt_pulp.oct.tree import generate_nodes, calculate_gini_old, calculate_gini_fast, calculate_misclassification

def generate_model_tree(
        P: list,
        K: list,
        data: pd.DataFrame,
        y_idx: int = 0,
        time_limit: float = 1800, # moved to train_model to give parameter to solver directly
        gap_limit: float = None,
        log_to_console: bool = False,
        big_m: int = 99,
        criterion: str = "gini"
):
    """

    :param criterion:
    :param big_m:
    :param depth:
    :param P:
    :param K:
    :param data:
    :param leaf_nodes_path:
    :param y_idx:
    :param time_limit:
    :param gap_limit:
    :param log_to_console:
    :return:
    """

    # Create parent & leaf nodes

    leaf_nodes_path = {4: [1, 1], #dict for all leafes in depth 2 tree and der respective split conditions
                       5: [1, 0],
                       6: [0, 1],
                       7: [0, 0]}
    depth = 2
    parent_nodes, leaf_nodes = generate_nodes(depth) # (for depth 2) returns [1,2,3] and [4,5,6,7]

    nodes = dict()
    nodes["leaf_nodes"] = leaf_nodes
    nodes["leaf_nodes_path"] = leaf_nodes_path

    print(nodes)

    if criterion == "gini":
        logging.info("Calculating gini..")
        coef_dict = calculate_gini_old(data=data,
                                    P=P,
                                    K=K,
                                    nodes=nodes)

        
    elif criterion == "misclassification":
        logging.info("Calculating misclassification..")
        coef_dict = calculate_misclassification(data=data,
                                                P=P,
                                                nodes=nodes)
    # init model
    model = LpProblem("RollOCT", LpMinimize) # Sets the objective to be minimized

    # # x[i,j] and y[i,k] as binary variables
    x = dict()
    for i in P:
        for j in P:
            x[i, j] = LpVariable(f'x[{i},{j}]', cat='Binary') #creates variable
            # In PuLP, variables are automatically added to the model when you include them in constraints or the objective function—you do not need to explicitly register them with the model
            # cat='Binary': Specifies the variable type as binary
            # name=f'x[{i}, {j}]': Assigns a name to the variable for easier identification and debugging
            # x[i, j]: Binary variable indicating whether feature i is used for the first split and feature j for the second split

    y = dict()
    for i in P:
        for k in P:
            y[i, k] = LpVariable(f'y[{i},{k}]', cat='Binary')

#lpSum is to gulps what quicksum is to gurobi

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
# variable x[i, j], y[i, k] is binary, so 1 or 0. So the multiplication makes sense
# .get(key, default): Dictionary method to safely retrieve values, using big_m as a fallback
# coef_dict[4].get((i, j))
# in coef_dict[4].get((i, j) is value of loss function of leaf 4 with features i,j from P. Value is found in dict if it matches the respective split condition of leaf 4 [1, 1]. If not penalty big m is used
# => damit der Wert fuer loss function bei feature combination i,j im coef_dict steht, müssen i,j (nach definition wie coef_dict erstellt wird) dort zur split condition (vorhanden/nicht vorhanden) gematched haben  
# => also man kann in coef_dict[4].get((i, j, big_m) nur Werte für die features finden, die im Datensatz der Kombination [1,1] entsprochen haben. Falls der Wert nicht vorhanden ist (alle i,j Kombinationen werden iteriert), wird stattdessen Wert big_m genommen

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


    # #set a time limit (if using a compatible solver like CBC or Gurobi):
    # if time_limit:
    #     m.setParam("TimeLimit", time_limit)
    #     logging.info(f'Setting Time Limit as {time_limit}')

    # if gap_limit is not None:
    #     m.setParam("MipGap", gap_limit) #MipGap: Sets the optimality gap tolerance for early termination
    #     logging.info(f'Setting Optimality Gap as {gap_limit}')

    # m.setParam("LogToConsole", int(log_to_console))
    # logging.info(f'Setting LogToConsole as {log_to_console}')
    # m.update() #Updates the model with all changes.

    model_dict = {
        'model': model,
        'params': {
            'var_x': x,
            'var_y': y,
            'y_idx': y_idx
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
    logging.info('Model generation is done.')

    return model_dict