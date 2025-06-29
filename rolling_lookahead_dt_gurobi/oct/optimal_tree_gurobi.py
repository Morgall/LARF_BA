import pandas as pd
import numpy as np
from gurobipy import *
import time

from rolling_lookahead_dt_gurobi.oct.tree import generate_nodes, calculate_gini, \
    calculate_misclassification


def generate_model_gurobi(
        P: list,
        K: list,
        data: pd.DataFrame,
        y_idx: int = 0,
        time_limit: float = 1800,
        gap_limit: float = None,
        log_to_console: bool = False,
        big_m: int = 99,
        criterion: str = "gini",
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
        coef_dict = calculate_gini(data=data,
                                   P=P,
                                   K=K,
                                   nodes=nodes)
        
    elif criterion == "misclassification":
        logging.info("Calculating misclassification..")
        coef_dict = calculate_misclassification(data=data,
                                                P=P,
                                                nodes=nodes)
    # init model
    m = Model("RollOCT") #Creates a new Gurobi optimization model named "RollOCT"

    # Define Variables
    x = dict()
    for i in P:
        for j in P:
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x[{i}, {j}]') 
            # m.addVar adds no decision variable to model
            # vtype=GRB.BINARY: Specifies the variable type as binary
            # name=f'x[{i}, {j}]': Assigns a name to the variable for easier identification and debugging
            # x[i, j]: Binary variable indicating whether feature i is used for the first split and feature j for the second split

    y = dict()
    for i in P:
        for k in P:
            y[i, k] = m.addVar(vtype=GRB.BINARY, name=f'y[{i}, {k}]')

# quicksum is gurobipy function (much like Python’s built-in sum()) but optimized for constructing large Gurobi models

    # Constraint 1
    m.addConstr(quicksum(x[i, j] for j in P for i in P) == 1) # Ensures that exactly one combination of features (i, j) is selected for the first two splits (criteria (1b))
    # Constraint 2
    m.addConstr(quicksum(y[i, k] for k in P for i in P) == 1) # implements criteria (1c) same way as above
    # Constraint 3
    for i in P:
        m.addConstr(
            quicksum(x[i, j] for j in P) == quicksum(y[i, k] for k in P)) # Links the x and y variables, ensuring that for each feature i, the sum of x[i, j] across j equals the sum of y[i, k] across k. This ensures consistency between the splits (criteria (1d))

    # Constraint 4
    # add big m; Acts as a penalty for invalid splits (if a combination is not present in coef_dict, it uses big_m as a large penalty)

# implements criteria (1a)
# variable x[i, j], y[i, k] is binary, so 1 or 0. So the multiplication makes sense
# .get(key, default): Dictionary method to safely retrieve values, using big_m as a fallback
# coef_dict[4].get((i, j))
# in coef_dict[4].get((i, j) is value of loss function of leaf 4 with features i,j from P. Value is found in dict if it matches the respective split condition of leaf 4 [1, 1]. If not penalty big m is used
# => damit der Wert fuer loss function bei feature combination i,j im coef_dict steht, müssen i,j (nach definition wie coef_dict erstellt wird) dort zur split condition (vorhanden/nicht vorhanden) gematched haben  
# => also man kann in coef_dict[4].get((i, j, big_m) nur Werte für die features finden, die im Datensatz der Kombination [1,1] entsprochen haben. Falls der Wert nicht vorhanden ist (alle i,j Kombinationen werden iteriert), wird stattdessen Wert big_m genommen

    obj = quicksum(
        (coef_dict[4].get((i, j), big_m) + coef_dict[5].get((i, j), big_m)) *
        x[i, j]
        for i in P
        for j in P) + \
          quicksum((coef_dict[6].get((i, k), big_m) + coef_dict[7].get((i, k), big_m)) *
                   y[i, k]
                   for i in P
                   for k in P)

    m.setObjective(obj, GRB.MINIMIZE)  # Sets the objective to be minimized.



    if time_limit:
        m.setParam("TimeLimit", time_limit)
        logging.info(f'Setting Time Limit as {time_limit}')

    if gap_limit is not None:
        m.setParam("MipGap", gap_limit) #MipGap: Sets the optimality gap tolerance for early termination
        logging.info(f'Setting Optimality Gap as {gap_limit}')

    m.setParam("LogToConsole", int(log_to_console))
    logging.info(f'Setting LogToConsole as {log_to_console}')
    m.update() #Updates the model with all changes.

    model_dict = {
        'model': m,
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


def train_model_gurobi(data: pd.DataFrame,
                model_dict: dict,
                P: list) -> dict:
    """

    :param data:
    :param model_dict: A dictionary containing model components (nodes, parameters, and the Gurobi model)
    :param P: List of feature indices to consider
    :return:
    """

    #initialization
    logging.info("Training..")
    nodes = model_dict['nodes']
    params = model_dict['params'] #is reference to the variables in the model; references them directly. They cary solution after m.optimize()
    m = model_dict['model']

    """
    The m.optimize() method in Gurobi (gurobipy) is the command that actually solves your optimization model.
    When you call m.optimize(), Gurobi performs the following:

        - Solves the Model: It runs the optimization algorithm (such as simplex, barrier, or branch-and-cut, 
        depending on the model type) to find the best possible solution according to your objective function
        and constraints.

        - Populates Solution Attributes: After optimization, Gurobi updates the model’s internal attributes 
        (like .ObjVal for the objective value, .X for variable values, and .status for the solution status) so you 
        can query the results.

        - Handles Termination: The method will terminate according to any limits you set (such as time limits
        or solution tolerances), and you can check the optimization status to see if it found an optimal solution,
        encountered infeasibility, or hit a limit.

    In summary, m.optimize() is the command that tells Gurobi to solve your optimization problem and
    store the solution and related information in the model object for you to access afterward.

    """

    m.optimize() # solves our optimization model

    #Check Optimization Status

    if m.status == GRB.Status.OPTIMAL or \
            (m.status == 9 and m.objVal != -math.inf): 
        logging.info(f'Optimal objective: {m.objVal}') # OPTIMAL: Solution found.
        status = m.status
    elif m.status == GRB.Status.INFEASIBLE: # INFEASIBLE: No solution exists
        logging.info("Model is infeasible")
        return
    elif m.status == GRB.Status.UNBOUNDED: # UNBOUNDED: The objective can be improved infinitely
        logging.info("Model is unbounded")
        return
    elif m.status == 9 and m.objVal != -math.inf: #Status 9: Time limit reached (if objective is not -∞)
        logging.info('Time Limit termination')
        return
    else:
        logging.info(f'Optimization status {m.status}')
        return

    logging.info(f"Objective Value: {m.objVal}")
    
    
    # Arrange/EXTRACT decision variables WITH SOLUTIONS => dict of solutions (optimization for splits)
    #x and y: Dictionaries mapping pairs (i, j) (key) to the value of the decision variables var_x and var_y (data) from the solution.
    # variables are defined as binary, so they are 0 or 1
    x = {
        (i, j): params['var_x'][i, j].X #Accesses the .X attribute of the variable params['var_x'][i, j]; In Gurobi, .X gives the solution value of the variable after optimization
        for i in P for j in P
    }
    y = {
        (i, j): params['var_y'][i, j].X
        for i in P for j in P
    }


    # Identify Selected Features
    # Sets 'first_level', 'left_second_level', and 'right_second_level' variables => wir bekommen also die Identifikation für die features in jedem Split
    # wir wollen hier über alle variablen die, die Variablen finden, die mit 1 belegt sind, also laut solver gewählt werden sollen. Aus diesen entnehmen wir die features
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
                
# Determine Leaf Node Predictions
# hier wird wieder aus den features ausgedünnt 
    logging.info("Extracting solution..")
    df_arr = np.array(data)
    target_class = dict()
    for leaf_ in nodes["leaf_nodes"]: # iterate through leaf nodes
        first_var = nodes["leaf_nodes_path"][leaf_][0]
        second_var = nodes["leaf_nodes_path"][leaf_][1]
        arr = df_arr[np.where((df_arr[:, first_level] == first_var))] # Selects all rows (observations), but only the column at index first_level. Compares each value in the first_level column to first_var; returns bool: True means the value matches first_var
        # wir wissen nach welchem feature wir root splitten wollen (first_level). Also nehmen wir nur diese Spalte. Dort nehmen wir alle Zeilen, deren Wert first_var entspricht (also Wert aus {0,1})
        if leaf_ in [4, 5]:  # left
            arr_2 = arr[np.where(arr[:, left_second_level] == second_var)] #entsprechend weiter filtern
        elif leaf_ in [6, 7]:  # right
            arr_2 = arr[np.where(arr[:, right_second_level] == second_var)] #entsprechend weiter filtern
        else:
            pass
        values, counts = np.unique(arr_2[:, params["y_idx"]], 
                                   return_counts=True)
        # params["y_idx"] meist= params[0] is coloumn with the target variables (the class label we want to predict))
        # values: An array of the unique class labels present in the filtered data => also alle class labels die nach dem filtern noch übrig sind
        # counts: An array of how many times each unique class label appears in the filtered data.

        if len(counts) > 0:
            target_class[leaf_] = values[np.argmax(counts)] # np.argmax(counts) finds the index of the highest value in counts (i.e., the most frequent class) => values[np.argmax(counts)] gets the class label that corresponds to the most frequent count.
        # target_class speichert also für jedes leaf, welches class label am meisten vorkommt


    # Prepare Feature Selection Output
    # Creates a dictionary var_a: Each key (1, 2, 3) corresponds to a feature selection vector. 1 at the position of the selected feature, 0 otherwise.
    # Each value is a list (length = number of features, len(P)), where: 1 marks the selected feature for that split; 0 marks all other features
    # var_a is a more structured and general way to represent the feature selection
    # var_a represents feature selection over all features in a way some models want it 

    var_a = {1: [0 if i != first_level else 1 for i in P], # Represents the feature selected for the root split
             2: [0 if i != left_second_level else 1 for i in P], # Represents the feature selected for the left branch of the second split
             3: [0 if i != right_second_level else 1 for i in P] # Represents the feature selected for the right branch of the second split
             }
    
    del df_arr


    logging.info(f'Training done. Loss: {m.objVal}\n'
                 f'Optimization status: {status}\n')
    



    # collects model statistics and results
    details = {
        'run_time': m.Runtime,
        'mip_gap': m.MIPGap,
        'objective': m.objVal,
        'status': status,
        'target_class': target_class,
        "var_a": var_a,
        "selected_features": {1: first_level,
                              2: left_second_level,
                              3: right_second_level
                              }
    }
    model_dict["details"] = details
    logging.info("Training is done.")
    return model_dict


    logging.info(f'Training done. Loss: {m.objective.value()}\n'
                 f'Optimization status: {status}\n')
    



    # collects model statistics and results
    details = {
        'run_time': runtime,
        #'mip_gap': m.MIPGap,
        'objective': m.objective.value(),
        'status': status,
        'target_class': target_class,
        "var_a": var_a,
        "selected_features": {1: first_level,
                              2: left_second_level,
                              3: right_second_level
                              }
    }
    model_dict["details"] = details
    logging.info("Training is done.")
    return model_dict


#predicts the class for each instance in data as DataFrame using decision tree model, and marks at which leaf node each instance ends up in.
def predict_model_gurobi(data: pd.DataFrame,
                  P: list,
                  model_dict: dict,
                  pruned_nodes: list = []) -> pd.DataFrame:
    """

    :param model_dict:
    :param pruned_nodes:
    :param data:
    :param P:
    :return:
    """

    prediction = []
    leaf_ = []
    depth = model_dict["depth"]
    var_a = model_dict["details"]["var_a"]
    target_class = model_dict["details"]["target_class"]
    for idx, i in data.iterrows():
        x = np.array(i[P])
        t = 1
        d = 0
        while d < depth:
            at = np.array(var_a[t])
            if at.dot(x) == 1:
                t = t * 2
            else:
                t = t * 2 + 1
            d = d + 1
            if t in pruned_nodes:
                break
        prediction.append(target_class[t])
        leaf_.append(t)
    data["prediction"] = prediction
    data["leaf"] = leaf_
    logging.info("Prediction is done.")
    return data
