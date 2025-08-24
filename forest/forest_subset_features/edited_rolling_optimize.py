from typing import Optional
import time

from rolling_lookahead_dt_pulp.oct.optimal_tree_pulp import predict_model_pulp

from rolling_lookahead_dt_pulp.oct.tree import *

from forest.forest_subset_features.edited_model import generate_model_tree, train_model


def rolling_optimize(predefined_model: Optional[dict],
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      original_training_dataset: pd.DataFrame,
                      features_orig_dataset : int,
                      main_depth: int,
                      target_depth: int,
                      features: list, # This is the subset from the root tree
                      time_limit: float,
                      to_go_deep_nodes: list,
                      result_dict: dict,
                      criterion: str = "gini",
                      random_state = None
                      ) -> pd.DataFrame:
    """
    Performs rolling optimization to extend a pre-trained decision tree.
    It now includes random feature subsetting for each new subtree.
    """
    model_name = f"rolling_optimizing_gini_{main_depth}"
    logging.info(f"Running {model_name} with main depth {main_depth} by "
                 f"diving in "
                 f"2 level with target depth {target_depth}")

    # Use a copy to avoid modifying the original dataframes
    df = train_data.copy()
    leaf_nodes_path = predefined_model["nodes"]["leaf_nodes_path"]
    y_idx = 0
    predefined_model.pop('model')
    predefined_model.pop('params')
    final_model = copy.deepcopy(predefined_model)

    selected_features_temp = copy.deepcopy(predefined_model["details"]["selected_features"])
    selected_features = {1: selected_features_temp}
    to_go_deep_nodes = {i: i for i in to_go_deep_nodes}
    pruned_nodes = [i for i in leaf_nodes_path if i not in to_go_deep_nodes]
    
    # Get the original feature list from the provided dataset
    original_feature_list = list(original_training_dataset.columns.drop('y'))
    
    new_train_data_dict = {}
    rng = np.random.RandomState(random_state)
    
    for level in range(1, target_depth - main_depth + 1):
        logging.info(f"Extending tree from Depth {main_depth} to "
                     f"{main_depth + level}")
        iter_time = time.time()
        to_go_deep_nodes_ = {}
        sub_K = {}

        parents_to_optimize = parents_of_nodes_to_branch_on(
            go_deep_nodes=to_go_deep_nodes)

        for node_ in parents_to_optimize:
            leafs_ = parents_to_optimize[node_]
            
            # --- Random Feature Selection for the Current Subtree ---
            # Correctly select a random subset of features from the original feature list.
            amount_X_consider = max(1, int(np.sqrt(features_orig_dataset)))
            subset_features_global = list(rng.choice(original_feature_list, 
                                              size=amount_X_consider, 
                                              replace=False))

            # --- Data Filtering for the Subtree based on path conditions ---
            if level == 1:
                # The split feature ID is the key to the selected_features dictionary
                # It is not a list, but a single integer
                main_split_feature_id = selected_features[1][1]
                
                # Filter the DataFrame rows based on the split feature and its value.
                first_var = leaf_nodes_path[leafs_[0]][0]
                
                # Check for the correct feature ID in the data and filter.
                sub_df_rows = df[df[main_split_feature_id] == first_var].copy()
                
                # Filter the columns to create the new sub-problem dataset.
                cols_to_keep = ['y'] + subset_features_global
                sub_train_data = sub_df_rows[cols_to_keep]

                new_train_data_dict[node_] = sub_train_data
                sub_K[node_] = list(np.unique(new_train_data_dict[node_].y))
                
            else:
                # Use the already filtered data from the previous level.
                arr_df = new_train_data_dict[int(node_ / 2)].copy()
                # Get the global feature ID for the previous split.
                prev_split_feature_global = selected_features[int(node_ / 2)][2]
                first_var = leaf_nodes_path[to_go_deep_nodes[leafs_[0]]][0]
                
                # Filter rows based on the previous level's split.
                arr_rows_df = arr_df[arr_df[prev_split_feature_global] == first_var]
                
                # Now, filter columns to the NEW random subset.
                sub_train_data_df = arr_rows_df[['y'] + subset_features_global]
                
                new_train_data_dict[node_] = sub_train_data_df
                sub_K[node_] = list(np.unique(new_train_data_dict[node_].y))

            logging.info(f"Processing for Parent Node: {node_} of Leaf "
                         f"Node(s): {leafs_} at Level: {level}")
                         
            # --- Convert global feature IDs to local indices for the sub-model ---
            # The `generate_model_tree` function expects local indices (1, 2, 3...)
            # so we need to create a new mapping for each sub-problem.
            local_P_sub = list(range(1, len(subset_features_global) + 1))
            
            # Create a local-to-global map to be saved in the model's details
            local_to_global_sub = dict(zip(local_P_sub, subset_features_global))
            
            # --- Model generation & training with the new feature subset ---
            main_model = generate_model_tree(P=local_P_sub,
                                             K=sub_K[node_],
                                             data=new_train_data_dict[node_],
                                             y_idx=y_idx,
                                             time_limit=time_limit,
                                             log_to_console=False,
                                             criterion=criterion)
            
            main_model = train_model(model_dict=main_model,
                                     data=new_train_data_dict[node_],
                                     P=local_P_sub)
            
            # Applies the trained sub-model to the sub-dataset
            result_ = predict_model_pulp(
                data=new_train_data_dict[node_],
                model_dict=main_model,
                P=local_P_sub)
            
            # Determines which leaves remain misclassified
            misclassified_leafs = find_misclassification(df=result_)
            del result_

            # Update model’s splitting rules
            temp_dict = {}
            for t in predefined_model["nodes"]["parent_nodes"]:
                new_idx = parent_pattern(sub_leaf=t,
                                         leaf_node=node_)
                temp_dict[new_idx] = main_model["details"]["var_a"][t]

            main_model["details"]["var_a"] = temp_dict

            # Update target-class assignments
            temp_target = {}
            for t in predefined_model["nodes"]["leaf_nodes"]:
                new_idx = leaf_pattern(sub_leaf=t, depth=main_depth,
                                       leaf=node_)
                if t in main_model["details"]["target_class"]:
                    temp_target[new_idx] = main_model["details"]["target_class"][t]
                    if t in misclassified_leafs:
                        to_go_deep_nodes_[new_idx] = t
                    else:
                        pruned_nodes.append(new_idx)

            temp_class_assign = copy.deepcopy(pruned_nodes)
            for i in temp_class_assign:
                for parent in parents_to_optimize:
                    if not (i != parent * 2 and i != (parent * 2 + 1)):
                        pruned_nodes.remove(i)
                    if not (i != get_child(1, 2, parent) * 2 or i != (
                            get_child(1, 2, parent) * 2 + 1)):
                        pruned_nodes.remove(i)
            
            new_train_data_dict[node_] = new_train_data_dict[node_].drop(
                ["prediction", "leaf"], axis=1)
            
            main_model["details"]["target_class"] = temp_target
            
            # The selected_features dict should store a mapping from local node to the global feature ID
            selected_features[node_] = {
                1: local_to_global_sub[main_model["details"]["selected_features"][1]],
                2: local_to_global_sub[main_model["details"]["selected_features"][2]],
                3: local_to_global_sub[main_model["details"]["selected_features"][3]]
            }
            
            final_model["details"]["var_a"].update(main_model["details"]["var_a"])
            for k in [node_ * 2, node_ * 2 + 1]:
                if k in final_model["details"]["target_class"]:
                    final_model["details"]["target_class"].pop(k)
            final_model["details"]["target_class"].update(main_model["details"]["target_class"])

        result_dict['tree'][current_depth] = {}
        result_dict['tree'][current_depth]['trained_dict'] = final_model

        print('final model reached')
        
        # P for predict_model_pulp needs to be the list of all features used
        # in the tree so far. We must collect them.
        all_features_so_far = []
        for sf_dict in selected_features.values():
            all_features_so_far.extend(list(sf_dict.values()))
        all_features_so_far = list(set(all_features_so_far))
        
        final_model['details']['all_features'] = all_features_so_far
        
        result_training_data = predict_model_pulp(data=train_data,
                                model_dict=final_model,
                                P=final_model['details']['all_features'],
                                pruned_nodes=pruned_nodes)
        
        result_dict['tree'][current_depth]['train'] = result_training_data[['y', 'prediction', 'leaf']]

        train_acc = len(result_training_data.loc[result_training_data["prediction"]
                        == result_training_data["y"]]) / len(result_training_data["y"])

        result_test_data = predict_model_pulp(data=test_data,
                               model_dict=final_model,
                               P=final_model['details']['all_features'],
                               pruned_nodes=pruned_nodes)
        
        result_dict['tree'][current_depth]['test'] = result_test_data[['y', 'prediction', 'leaf']]
        
        prediction_acc = len(result_test_data.loc[result_test_data["prediction"]
                       == result_test_data["y"]]) / len(result_test_data["y"])
                       
        logging.info(
            f"Test Accuracy: {prediction_acc}. Training Accuracy: "
            f"{train_acc}. Iteration is over for level"
            f" {level}. Final "
            f"depth is {main_depth + level}")
        to_go_deep_nodes = to_go_deep_nodes_

        result_dict[current_depth] = {
            "training_accuracy": train_acc,
            "test_accuracy": prediction_acc,
            "time": time.time() - iter_time
        }
    return result_dict

