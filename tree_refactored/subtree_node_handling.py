from tree_refactored.class_node import TreeNode
import pandas as pd

def distribute_data_to_children(root_node : TreeNode):
    
    """
    edgecase; split feature was perfect. 
    remainder of code should run regardless and "distribute" the nonexistend data
    get_predict_and_pure() handles that empty nodes inherit the prediction from parent
    """
    if root_node.is_pure == True:
        print(f"\nDistributing to children from pure node {root_node.number}...")

    left_child = root_node.left
    left_child.datapoints_in_node = pd.DataFrame()
    right_child = root_node.right
    right_child.datapoints_in_node = pd.DataFrame()

    for index, row in root_node.datapoints_in_node.iterrows(): #row is a dataframe series, not a dataframe, if root_node.datapoints_in_node is an empty DataFrame (no rows), the loop simply does not execute any iterations
        if row[root_node.feature] == 0:
            left_child.datapoints_in_node = pd.concat([left_child.datapoints_in_node, row.to_frame().T], ignore_index=False)
        else:
            right_child.datapoints_in_node = pd.concat([right_child.datapoints_in_node, row.to_frame().T], ignore_index=False)
            
    if len(root_node.datapoints_in_node) != ((len(left_child.datapoints_in_node) + len(right_child.datapoints_in_node))):
        raise Exception(f"for some reason node {root_node.number} wasnt emptied correctly")
    
    else: # data is needed for tree building
        #del root_node.datapoints_in_node
        #root_node.datapoints_in_node = None
        return

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