from sklearn.utils.validation import check_is_fitted
import time

import numpy as np
import pandas as pd
from queue import Queue

import math

from tree_refactored.class_node import TreeNode
from tree_refactored.subtree_node_handling import *
from tree_refactored.get_features_subtree import solve_features_subtree, solve_features_subtree_vectorized

# only gini index for now
# must ensure data given to functions has correct binary format; consult makeReady_datasets.ipynb and example
class DecisionTree_rollOCT:
    def __init__(self, max_depth=None, max_features = None):
        self.max_depth = max_depth
        self.root = None                    # node representing final tree
        self.max_features = max_features
        self.is_fitted_=False
        self.fit_time = None                # time needed to fit tree with depth max_depth

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

            X_list = [int(col) for col in X.columns] #col names must be int or 'int'
            rng = np.random.RandomState(None) # could maybe use self.random_state
            subset_features_list = rng.choice(X_list, size=amount_features_train, replace=False) #subset features to train on
            subset_features_list = sorted(subset_features_list) #maybe not necessary
            subset_features_list_str = [str(x) for x in subset_features_list]

            subset_features = X[subset_features_list_str]
            #print(subset_features)

            #targets = node.datapoints_in_node['y'] #target col of datapoints still in node; df series
            targets = root_node_subtree.datapoints_in_node[['y']] #gives true dataframe
            #print(targets)

            
            # ------ Solve the resulting [OCT-2] problem growing from the selected node --------
            
            """
            node1_feature is No (0) instance (left child of root)
            node2_feature is Yes (1) instance (right child of root)
            it would probably be possible to just delete the cols of selceted features from X if that is something thats needed
            """
            #root_feature, node1_feature, node2_feature = solve_features_subtree(subset_features, targets)
            root_feature, node1_feature, node2_feature = solve_features_subtree_vectorized(subset_features, targets)

            """
            create tree with correct numbering of nodes and pass data through tree according to selcted features
            result: tree with data distributed across the leaves according to calculated split features
            identify impure leaves
            """

            subtree, parents_impure_list = self.create_subtree(root_feature, node1_feature, node2_feature, root_node_subtree) #tree always represented by root


            # Remove the current node from tree
            # and add internal nodes of all resulting leaf nodes with misclassification to the queue
            # => this is handled by parent pointer, current internal node just gets split feature replaced and creates new subtree from there (and there looses its prevoius children)
            for node in parents_impure_list: #lower number node is always first in this list
                #print(node.number)
                queue.put(node)


            # Set current_depth equal to 1 plus the smallest depth of the nodes in the set S
            #current_depth = queue.queue[0].depth +1 #not thread safe, crashes at depth 21-22
            
            print(f"new depth of tree {current_depth+2}")
            
            
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

        for index, row in X.iterrows():
            node = self.root
            while not(node.is_leaf):
                if row[node.feature] == 0:
                    node = node.left
                else:
                    node = node.right
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

        # leaves; nodes 3-6

        node3 = TreeNode(number=2*left_child.number+1, parent = left_child, depth = left_child.depth+1, is_leaf=True)
        left_child.left = node3
        node4 = TreeNode(number=2*left_child.number+2, parent = left_child, depth = left_child.depth+1, is_leaf=True)
        left_child.right = node4

        node5 = TreeNode(number=2*right_child.number+1, parent = right_child, depth = right_child.depth+1, is_leaf=True)
        right_child.left = node5
        node6 = TreeNode(number=2*right_child.number+2, parent = right_child, depth = right_child.depth+1, is_leaf=True)
        right_child.right = node6

        """
        distributing data to leafes
        get_predict_and_pure(node) must be called before distribute_data_to_children(node)
        because distribute_data_to_children cleans up after itself
        """
        distribute_data_to_children(root_node)
        get_predict_and_pure(left_child)
        get_predict_and_pure(right_child)

        distribute_data_to_children(left_child)
        get_predict_and_pure(node3)
        get_predict_and_pure(node4)

        distribute_data_to_children(right_child)
        get_predict_and_pure(node5)
        get_predict_and_pure(node6)

        """
        print(f"\nroot predict: {root_node.prediction}")
        print(f"\nleft predict: {left_child.prediction}")
        print(f"\nright predict: {right_child.prediction}")
        print(f"\nnode3predict: {node3.prediction}")
        print(f"\nnode4 predict: {node4.prediction}")
        print(f"\nnode5 predict: {node5.prediction}")
        print(f"\nnode6 predict: {node6.prediction}")
        """



        # impurity check of leaves

        parents_impure_list = []
        if (node3.is_pure == False and node3.is_empty == False) or (node4.is_pure == False and node4.is_empty == False):
            parents_impure_list.append(left_child)
        if (node5.is_pure == False and node5.is_empty == False)  or (node6.is_pure == False and node6.is_empty == False):
            parents_impure_list.append(right_child)
        
        

        return subtree, parents_impure_list
    