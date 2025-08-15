import pandas as pd
import numpy as np
from pulp import *
import time

from rolling_lookahead_dt_pulp.oct.tree import generate_nodes, calculate_gini_old, calculate_gini_fast, calculate_misclassification


def generate_model_pulp(
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


def train_model_pulp(data: pd.DataFrame,
                model_dict: dict,
                P: list,
                time_limit: float = 1800):
                
    
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


    start_time = time.time()
    m.solve(solver=PULP_CBC_CMD(msg=True, timeLimit=time_limit)) # solves our optimization model
    runtime = time.time() - start_time

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
                
# Determine Leaf Node Predictions, get target classes pf the leafs
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
def predict_model_pulp(data: pd.DataFrame,
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
    depth = model_dict["depth"] # defined as depth = 2 in generate_model_pulp and so depth 2 was hardcoded there from the beginning
    var_a = model_dict["details"]["var_a"] #dict with all features for every of the 3 split decisions. Contains just a singular 1 at the index of the feature that was chosen
    target_class = model_dict["details"]["target_class"] #dict with 4 entries 1-4 (leafs of 2depth tree) as keys. The respective values are the labels of datapoints that were put there most (majority vote) 
    
    
    # Verteile rows auf die leafs
    for idx, i in data.iterrows(): #iterates over rows of dataframe
        x = np.array(i[P]) #array of all feature values \in {0,1} of that row
        #print(x)
        t = 1
        d = 0
        while d < depth: #traversing tree until getting the leaves
            if t not in var_a:
                # Stop at this node, treat as leaf
                break
            at = np.array(var_a[t]) #binary vector over all features (P) with 1 where feature was chosen by solver
            #print(at)
            if at.dot(x) == 1: #dot product (sum of the element-wise); überprüft also ob x und at am gleichen Index eine 1 stehen haben. Das bedeutet also, dass wir in eine "ja" Instanz gehen, also in den linken Teilbaum
                t = t * 2
            else: # sonst gehen wir in den rechten Teilbaum
                t = t * 2 + 1
            d = d + 1
            if t in pruned_nodes:
                break

        pred = target_class.get(t, None)
        if pred is None:
            raise ValueError(f"Prediction stopped at node {t}, which does not have a target_class entry. This sample may reach a pruned/unused branch.")
        
        prediction.append(target_class[t]) #prediction (target label determined by model solver) for this row
        leaf_.append(t) #respective leaf to prediction (jedes leaf steht ja für ein target label)
    data["prediction"] = prediction #attach to data dataframe
    data["leaf"] = leaf_ #attach to data dataframe
    logging.info("Prediction is done.")
    return data
