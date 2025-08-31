from sklearn.utils.validation import check_is_fitted
import time

import numpy as np
import pandas as pd
from queue import Queue

import math

from tree_refactored.class_node import TreeNode
from tree_refactored.subtree_node_handling import *
from tree_refactored.get_features_subtree_original import *

from sklearn.utils.validation import check_is_fitted
import time

# only gini index for now
# must ensure data given to functions has correct binary format; consult makeReady_datasets.ipynb and example
class DecisionTree_rollOCT:
    def __init__(self, max_depth=None, max_features = None, reuse_features = True):
        self.max_depth = max_depth
        self.root = None                    # node representing final tree
        self.max_features = max_features
        self.is_fitted_=False
        self.fit_time = None                # time needed to fit tree with depth max_depth
        self.reuse_features = True

    def fit(self, X, y):
        
        time_start = time.time()
        
        queue = Queue(maxsize=0)
        whole_data = pd.concat([y, X], axis=1, ignore_index=False)
        
        # initialize root
        root_node = TreeNode(number = 0, depth=0, data = whole_data)
        self.root = root_node
        queue.put(root_node)

        # could add check if root is pure if so possibly stop
        get_predict_and_pure(root_node)


        current_depth = 0

        n_features = int(len(X.axes[1]))
        amount_features_train = self._get_feature_subset(n_features=n_features) #amount to train on
        print(f'Training each tree split on {amount_features_train} features of all {n_features} features')

        #features_to_use = X.copy()

        while (current_depth < self.max_depth) and (not(queue.empty())):
            
            
            # Select a node from the queue with the lowest depth; should always be node mit lowest number (BFS)
            print('\n' + "-" * 40)
            print(f"\nqueue empty: {queue.empty()}")

            root_node_subtree = queue.get()
            print(f'\nWorking on node number {root_node_subtree.number} with depth {root_node_subtree.depth}')

            current_depth = root_node_subtree.depth
            if current_depth+2 > self.max_depth:
                print("\n tree getting to deep ... stopping the building")
                break

            #print(list(queue.queue))

            """
            X_list = [int(col) for col in features_to_use.columns] #col names must be int or 'int'
            rng = np.random.RandomState(None) # could maybe use self.random_state
            subset_features_list = rng.choice(X_list, size=amount_features_train, replace=False) #subset features to train on
            subset_features_list = sorted(subset_features_list) #maybe not necessary
            subset_features_list_str = [str(x) for x in subset_features_list]

            subset_features = features_to_use[subset_features_list_str]
            """

            X_list = [int(col) for col in X.columns] #col names must be int or 'int'
            rng = np.random.RandomState(None) # could maybe use self.random_state
            subset_features_list = rng.choice(X_list, size=amount_features_train, replace=False) #subset features to train on
            subset_features_list = sorted(subset_features_list) #maybe not necessary
            subset_features_list_str = [str(x) for x in subset_features_list]

            #subset_features = X[subset_features_list_str]
            subset_features = root_node_subtree.datapoints_in_node[subset_features_list_str] #does not change the features in the node
            print(f"Subset features rows {subset_features.shape[0]}")

            #targets = node.datapoints_in_node['y'] #target col of datapoints still in node; df series
            targets = root_node_subtree.datapoints_in_node[['y']] #gives true dataframe
            #print(targets)
            print(f"targets rows {subset_features.shape[0]}")

            print("\nPreprocessing ...")

            train = pd.concat([targets, subset_features], axis=1)

            features = train.columns[1:]

            #print(f"\n features {features}")

            train = preprocess_dataframes( #./rollo_oct/utils/helpers.py
            train_df=train,
            target_label="y",
            features=features)

            
            P = [int(i) for i in
                list(train.loc[:, train.columns != 'y'].columns)]
            K = sorted(list(set(train.y)))

            lookup_dict = dict()
            for i, feature in enumerate(features):
                lookup_dict[i+1] = feature



            
            # ------ Solve the resulting [OCT-2] problem growing from the selected node --------
            
            """
            node1_feature is No (0) instance (left child of root)
            node2_feature is Yes (1) instance (right child of root)
            it would probably be possible to just delete the cols of selceted features from X if that is something thats needed
            """

            print("\nFinding features for subtree")
            root_feature, node1_feature, node2_feature = solve_features_subtree(P=P, K=K, data=train, y_idx=0, big_m=99)
            #root_feature, node1_feature, node2_feature = solve_features_subtree_vectorized(subset_features, targets)

            root_feature = int(lookup_dict[root_feature])
            node1_feature = int(lookup_dict[node1_feature])
            node2_feature = int(lookup_dict[node2_feature])
            print("\nSelected features subtree:")
            print(f"Root Node Feature: {root_feature}")
            print(f"No (0) instance child feature: {node1_feature}")
            print(f"Yes (1) instance child feature: {node2_feature}")

            """
            create tree with correct numbering of nodes and pass data through tree according to selcted features
            result: tree with data distributed across the leaves according to calculated split features
            identify impure leaves
            """

            subtree, parents_impure_list = self.create_subtree(root_feature, node1_feature, node2_feature, root_node= root_node_subtree) #tree always represented by root


            # Remove the current node from tree
            # and add internal nodes of all resulting leaf nodes with misclassification to the queue
            # => this is handled by parent pointer, current internal node just gets split feature replaced and creates new subtree from there (and there looses its prevoius children)
            for node in parents_impure_list: #lower number node is always first in this list
                #print(node.number)
                queue.put(node)


            # Set current_depth equal to 1 plus the smallest depth of the nodes in the set S
            #current_depth = queue.queue[0].depth +1 #not thread safe, crashes at depth 21-22
            
            print(f"new depth of tree {current_depth+2}")

            """ currenctly not in use
            if not(self.reuse_features):
                #features_to_use = features_to_use.drop(columns=f'{root_feature}')
                del features_to_use[f'{root_feature}']
                if (amount_features_train -1) <= len(features_to_use):
                    amount_features_train -= 1
                else:
                    print('\nsubset of features to small to select from ... exiting')
                    break
                """
            
            #break

        time_end = time.time()
        self.fit_time = time_end - time_start
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        """
        Predict values for a set of inputs
        traverse each datapoint trough tree
        give datapoint the prediction according to leaf where it ends
        """

        X['prediction'] = None
        #print(X['15'])
        for index, row in X.iterrows():
            #print(row)
            node = self.root
            while not(node.is_leaf):
                #print(f'\nnumber {node.number}; split feature {node.feature}')
                #print(f'node left child {node.left.number}; node right child {node.right.number}')
                #print(f'node left child is leaf {node.left.is_leaf}; node right child is leaf {node.right.is_leaf}')
                if (row[str(node.feature)]) == 0:
                    node = node.left
                elif (row[str(node.feature)]) == 1:
                    node = node.right
                #else:
                    #print(f"\nProblem at current node {node.number} with feature {node.feature}")

            #print(f"Leaf number {node.number} found; prediction is {node.prediction}")
            X.loc[index, 'prediction'] = node.prediction

        
        return X
            

        






        # remember to implement check if all datapoints where found again; after collecting datapoints over all leafes!
        pass

    def _get_feature_subset(self, n_features):
        if isinstance(self.max_features, int):
            k = self.max_features
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            k = max(1, int(math.log2(n_features)))
        elif self.max_features is None:
            k = n_features
            if self.reuse_features:
                self.reuse_features = True
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")
        return k
    
    def create_subtree(self, root_feature : int, node1_feature : int, node2_feature : int, root_node : TreeNode):
        subtree = root_node
        root_node.feature = root_feature
        
        # nodes 1 and 2
        left_child = TreeNode(feature=node1_feature, number=2*root_node.number+1, parent = root_node, depth = root_node.depth+1) #No (0) instance
        right_child = TreeNode(feature=node2_feature, number=2*root_node.number+2, parent = root_node, depth = root_node.depth+1) #Yes (1) instance
        root_node.left = left_child
        root_node.right = right_child

        left_child_df, right_child_df= distribute_data_to_children(root_node)
        left_child.datapoints_in_node = left_child_df
        right_child.datapoints_in_node = right_child_df
        #print(left_child.datapoints_in_node)
        get_predict_and_pure(left_child)
        get_predict_and_pure(right_child)

        # leaves; nodes 3-6

        node3 = TreeNode(number=2*left_child.number+1, parent = left_child, depth = left_child.depth+1, is_leaf=True)
        left_child.left = node3
        node4 = TreeNode(number=2*left_child.number+2, parent = left_child, depth = left_child.depth+1, is_leaf=True)
        left_child.right = node4

        node3_df, node4_df = distribute_data_to_children(left_child)
        node3.datapoints_in_node = node3_df
        node4.datapoints_in_node = node4_df
        get_predict_and_pure(node3)
        get_predict_and_pure(node4)

        node5 = TreeNode(number=2*right_child.number+1, parent = right_child, depth = right_child.depth+1, is_leaf=True)
        right_child.left = node5
        node6 = TreeNode(number=2*right_child.number+2, parent = right_child, depth = right_child.depth+1, is_leaf=True)
        right_child.right = node6

        """
        distributing data to leafes
        get_predict_and_pure(node) must be called before distribute_data_to_children(node)
        because distribute_data_to_children cleans up after itself
        """

        node5_df, node6_df = distribute_data_to_children(right_child)
        node5.datapoints_in_node = node5_df
        node6.datapoints_in_node = node6_df
        get_predict_and_pure(node5)
        get_predict_and_pure(node6)



        # impurity check of leaves


        parents_impure_list = []
        if (node3.is_pure == False and node3.is_empty == False) or (node4.is_pure == False and node4.is_empty == False):
            parents_impure_list.append(left_child)
        if (node5.is_pure == False and node5.is_empty == False)  or (node6.is_pure == False and node6.is_empty == False):
            parents_impure_list.append(right_child)

        
        

        return subtree, parents_impure_list
    