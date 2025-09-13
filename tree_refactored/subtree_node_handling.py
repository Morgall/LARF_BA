from tree_refactored.class_node import TreeNode
import pandas as pd
import gc

def distribute_data_to_children(root_node : TreeNode):
    
    if root_node.parent != None:
        del root_node.parent.datapoints_in_node
        root_node.parent.datapoints_in_node = None
        #gc.collect()
    
    """
    edgecase; split feature was perfect. 
    remainder of code should run regardless and "distribute" the nonexistend data
    get_predict_and_pure() handles that empty nodes inherit the prediction from parent
    """

    #print("Node", root_node.number, "Incoming data:")
    #print(root_node.datapoints_in_node)

    if root_node.is_pure == True:
        print(f"\nDistributing to children from pure node {root_node.number}...")

    data_left = pd.DataFrame()
    data_right = pd.DataFrame()

    data_right = root_node.datapoints_in_node[root_node.datapoints_in_node.iloc[:, root_node.feature] == 0].copy()
    data_left = root_node.datapoints_in_node[root_node.datapoints_in_node.iloc[:, root_node.feature] == 1].copy()
            
    if len(root_node.datapoints_in_node) != ((len(data_left) + len(data_right))):
        raise Exception(f"for some reason node {root_node.number} wasnt emptied correctly")
    
    
    else: # data is needed for tree building
        return data_left , data_right

def get_predict_and_pure(node : TreeNode):
    if node.datapoints_in_node.empty:
        node.is_empty = True
        #del node.datapoints_in_node
        #node.datapoints_in_node = None
        node.prediction = node.parent.prediction
        node.is_pure = False #needed for case where parent is pure and split is one sided => one child is empty => check which parent to add to todo list would add parent
        return
    vals = node.datapoints_in_node['y'].value_counts().index #gets the value not the frequencies
    l = len(vals)
    node.prediction = vals[0]
    if l == 1:
        node.is_pure = True       
        return
    else: 
        node.is_pure = False
        return