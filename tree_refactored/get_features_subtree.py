import numpy as np
import pandas as pd
from pulp import *



# Function to calculate Gini Index
def gini_index(labels):
    counts = labels.value_counts() #This counts the occurrences of each unique class in the labels series. For a binary classification problem, this might be {'0': 30, '1': 70}
    n_total = len(labels) # the total number of data points in the set
    if n_total == 0:
        return 0.0
    gini = 1.0 - sum((counts[c] / n_total)**2 for c in counts.index)
    return gini


# huge ram needs on adult dataset, can be a lot more intense on ram depending on amount of features and datapoints. Use with caution
def preprocess_gini_ultimate_vectorized(full_data, feature_names, target, n_total, big_m=99999):
    
    print("\nPre-computing loss components with ultimate vectorized method...")

    # Step 1: Create a long-format DataFrame with features only
    full_data['id'] = range(len(full_data))
    df_long = pd.melt(full_data, id_vars=['id'], value_vars=feature_names, var_name='feature', value_name='value')
    
    # Step 2: Create a DataFrame with all feature combinations
    root_features_df = df_long.rename(columns={'feature': 'root_feature', 'value': 'root_value'})
    child_features_df = df_long.rename(columns={'feature': 'child_feature', 'value': 'child_value'})
    
    # Step 3: Perform a cross-join (Cartesian product) to create all combinations
    merged_df = pd.merge(root_features_df, child_features_df, on='id', suffixes=('_root', '_child'))
    
    # Step 4: Add the target column back for Gini calculation
    merged_df[target] = merged_df['id'].map(full_data.set_index('id')[target])
    
    # Step 5: Group by the split conditions and calculate Gini loss
    grouped_results = merged_df.groupby(['root_feature', 'child_feature', 'root_value', 'child_value'])
    
    c_values = {}
    
    for (j, k, r, s), group in grouped_results:
        if j == k:
            c_values[(j, k, r, s)] = big_m
            continue
            
        n_subset = len(group)
        impurity = gini_index(group[target])
        loss_component = (n_subset / n_total) * impurity
        c_values[(j, k, r, s)] = loss_component

    print("Loss components computed.")
    
    return c_values




# great speedup but not trivial due to vectorization magic
# huge ram needs on adult dataset, can be a lot more intense on ram depending on amount of features and datapoints. Use with caution
def solve_features_subtree_vectorized(features: pd.DataFrame, y: pd.DataFrame):
    
    y.columns = ['y']
    full_data = pd.concat([features, y], axis=1)
    feature_names = features.columns
    target = y.columns[0]
    n_total = len(full_data)
    big_m = 99999 #using this penalty for the case that i==k (which isnt handled below) and we explicitly dont want the same feature to be used twice

    c_values = preprocess_gini_ultimate_vectorized(full_data, feature_names, target, n_total, big_m)

    print("Loss components computed.")
    
    # --- Formulate and solve the linear programming problem with PuLP ---
    print("\nFormulating the OCT-2 linear programming problem...")

    # Create the LP problem
    prob = LpProblem("Optimal_2_Depth_Decision_Tree", LpMinimize)

    # The use of LpVariable.dicts creates a dictionary of these following variables, making them easy to reference by their feature pair.
    # Define the binary decision variables
    z1_vars = LpVariable.dicts("z1", [(j, k) for j in features for k in features], 0, 1, 'Binary') #z_jk(1): ​This variable is equal to 1 if feature j is the root split and feature k is the split for the left sub-tree (the 'No' or x_j=0 branch). Otherwise, it's 0.
    z2_vars = LpVariable.dicts("z2", [(j, l) for j in features for l in features], 0, 1, 'Binary') #z_jl(2)​: This variable is equal to 1 if feature j is the root split and feature l is the split for the right sub-tree (the 'Yes' or x_j=1 branch). Otherwise, it's 0.

    # --- Criteria (1a): Objective function ---
    """
    Each term represents the total Gini loss contribution from one of the four leaves
    The precomputed c_values dictionary holds the Gini loss component for each potential leaf
    The objective function multiplies these pre-computed c values by the binary decision variables

    For example, c_values.get((j, k, 0, 0), 0) * z1_vars[(j, k)] represents the Gini loss for the leaf where x_j=0 and x_k=0 (see construction above) 
    If the variable z_jk(1) is selected (i.e., it equals 1), this c value is included in the total sum.
    Since only one z_jk(1) and one z_jl(2) can be 1 (due to the constraints below), the final sum will represent the total Gini loss for the optimal tree structure
    """
    obj = lpSum([c_values.get((j, k, 0, 0), big_m) * z1_vars[(j, k)] for j in features for k in features]) + \
        lpSum([c_values.get((j, k, 0, 1), big_m) * z1_vars[(j, k)] for j in features for k in features]) + \
        lpSum([c_values.get((j, l, 1, 0), big_m) * z2_vars[(j, l)] for j in features for l in features]) + \
        lpSum([c_values.get((j, l, 1, 1), big_m) * z2_vars[(j, l)] for j in features for l in features])
    prob += obj, "Total_Gini_Loss"

    # --- Criteria (1b): Constraint for z1 variables ---
    prob += lpSum(z1_vars[(j, k)] for j in features for k in features) == 1, "Constraint_1b"

    # --- Criteria (1c): Constraint for z2 variables ---
    prob += lpSum(z2_vars[(j, l)] for j in features for l in features) == 1, "Constraint_1c"

    # --- Criteria (1d): Constraint for root feature consistency ---
    for j in features:
        prob += lpSum(z1_vars[(j, k)] for k in features) - lpSum(z2_vars[(j, l)] for l in features) == 0, f"Constraint_1d_{j}"

    # Solve the problem
    print("Solving the LP problem...")
    # After the solver runs, the z1_vars and z2_vars dictionaries will contain the optimal solution (i.e., the variables that are equal to 1)
    prob.solve(solver=PULP_CBC_CMD(msg=False))
    print("-" * 30)



    # --- Extract the results ---
    print(f"Status: {LpStatus[prob.status]}")
    print(f"Optimal Gini Loss: {prob.objective.value():.4f}\n")

    print("Extracting the selected features:")
    root_feature = None
    node1_feature = None
    node2_feature = None

    # Find the selected z1 variable
    for (j, k) in z1_vars:
        if z1_vars[(j, k)].value() == 1.0:
            root_feature = j
            node1_feature = k
            print(f"z_jk^(1) selected: ({j}, {k})")
            
    # Find the selected z2 variable
    for (j, l) in z2_vars:
        if z2_vars[(j, l)].value() == 1.0:
            node2_feature = l
            print(f"z_jl^(2) selected: ({j}, {l})")

    print("\nSelected features subtree:")
    print(f"Root Node Feature: {root_feature}")
    print(f"No (0) instance child feature: {node1_feature}")
    print(f"Yes (1) instance child feature: {node2_feature}")
    return root_feature, node1_feature, node2_feature









# significantly slower but readable; seems to be a lot more easy on ram
def solve_features_subtree(features : pd.DataFrame, y : pd.DataFrame):
    
    # --- Pre-compute the loss function components c_jk^(r,s) ---
    #print("\nPre-computing loss function components...")

    # Combine X and y for easier filtering
    full_data = pd.concat([features, y], axis=1)
    features = features.columns
    target = y.columns[0]
    c_values = {}
    n_total = len(full_data)

    """
    # Iterate over all possible pairs of features
    # feature l handled implicitly by the problem's structure
    # For the 'No' (0) branch of the root, the model selects feature k. The leaves of this branch are defined by the pairs (x_j=0,x_k=0) and (x_j=0,x_k=1)
    # For the 'Yes' (1) branch of the root, the model selects a different feature l. The leaves of this branch are defined by the pairs (x_j=1,x_l=0) and (x_j=1,x_l=1).
    """
    for j in features:
        for k in features: # k is feature of potential child not k as in the definition
            if j != k:
                for r in [0, 1]: #value of the root feature (x_j​). This corresponds to the no (0) or yes (1) instance of the root split
                    for s in [0, 1]: #value of the child feature (x_k​). This corresponds to the no (0) or yes (1) instance of the child split
                        # Filter data based on the split conditions {x_j = r AND x_k = s}
                        subset = full_data[(full_data[j] == r) & (full_data[k] == s)] # for each unique combination of j, k, r, and s, this line filters the full_data to create a subset that represents the data points ending up in a potential leaf node
                        n_subset = len(subset)
                        impurity = gini_index(subset[target])
                        loss_component = (n_subset / n_total) * impurity
                        c_values[(j, k, r, s)] = loss_component
    print("Loss components computed.")


    # --- Formulate and solve the linear programming problem with PuLP ---
    print("\nFormulating the OCT-2 linear programming problem...")

    # Create the LP problem
    prob = LpProblem("Optimal_2_Depth_Decision_Tree", LpMinimize)

    # The use of LpVariable.dicts creates a dictionary of these following variables, making them easy to reference by their feature pair.
    # Define the binary decision variables
    z1_vars = LpVariable.dicts("z1", [(j, k) for j in features for k in features], 0, 1, 'Binary') #z_jk(1): ​This variable is equal to 1 if feature j is the root split and feature k is the split for the left sub-tree (the 'No' or x_j=0 branch). Otherwise, it's 0.
    z2_vars = LpVariable.dicts("z2", [(j, l) for j in features for l in features], 0, 1, 'Binary') #z_jl(2)​: This variable is equal to 1 if feature j is the root split and feature l is the split for the right sub-tree (the 'Yes' or x_j=1 branch). Otherwise, it's 0.

    big_m = 99999 #using this penalty for the case that i==k (which isnt handled above) and we explicitly dont want the same feature to be used twice

    # --- Criteria (1a): Objective function ---
    """
    Each term represents the total Gini loss contribution from one of the four leaves
    The precomputed c_values dictionary holds the Gini loss component for each potential leaf
    The objective function multiplies these pre-computed c values by the binary decision variables

    For example, c_values.get((j, k, 0, 0), 0) * z1_vars[(j, k)] represents the Gini loss for the leaf where x_j=0 and x_k=0 (see construction above) 
    If the variable z_jk(1) is selected (i.e., it equals 1), this c value is included in the total sum.
    Since only one z_jk(1) and one z_jl(2) can be 1 (due to the constraints below), the final sum will represent the total Gini loss for the optimal tree structure
    """
    obj = lpSum([c_values.get((j, k, 0, 0), big_m) * z1_vars[(j, k)] for j in features for k in features]) + \
        lpSum([c_values.get((j, k, 0, 1), big_m) * z1_vars[(j, k)] for j in features for k in features]) + \
        lpSum([c_values.get((j, l, 1, 0), big_m) * z2_vars[(j, l)] for j in features for l in features]) + \
        lpSum([c_values.get((j, l, 1, 1), big_m) * z2_vars[(j, l)] for j in features for l in features])
    prob += obj, "Total_Gini_Loss"

    # --- Criteria (1b): Constraint for z1 variables ---
    prob += lpSum(z1_vars[(j, k)] for j in features for k in features) == 1, "Constraint_1b"

    # --- Criteria (1c): Constraint for z2 variables ---
    prob += lpSum(z2_vars[(j, l)] for j in features for l in features) == 1, "Constraint_1c"

    # --- Criteria (1d): Constraint for root feature consistency ---
    for j in features:
        prob += lpSum(z1_vars[(j, k)] for k in features) - lpSum(z2_vars[(j, l)] for l in features) == 0, f"Constraint_1d_{j}"

    # Solve the problem
    print("Solving the LP problem...")
    # After the solver runs, the z1_vars and z2_vars dictionaries will contain the optimal solution (i.e., the variables that are equal to 1)
    prob.solve(solver=PULP_CBC_CMD(msg=False))
    print("-" * 30)



    # --- Extract the results ---
    print(f"Status: {LpStatus[prob.status]}")
    print(f"Optimal Gini Loss: {prob.objective.value():.4f}\n")

    print("Extracting the selected features:")
    root_feature = None
    node1_feature = None
    node2_feature = None

    # Find the selected z1 variable
    for (j, k) in z1_vars:
        if z1_vars[(j, k)].value() == 1.0:
            root_feature = j
            node1_feature = k
            print(f"z_jk^(1) selected: ({j}, {k})")
            
    # Find the selected z2 variable
    for (j, l) in z2_vars:
        if z2_vars[(j, l)].value() == 1.0:
            node2_feature = l
            print(f"z_jl^(2) selected: ({j}, {l})")

    print("\nSelected features subtree:")
    print(f"Root Node Feature: {root_feature}")
    print(f"No (0) instance child feature: {node1_feature}")
    print(f"Yes (1) instance child feature: {node2_feature}")
    return root_feature, node1_feature, node2_feature