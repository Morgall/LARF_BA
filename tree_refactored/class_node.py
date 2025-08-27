class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, number=None, prediction=None, parent = None, depth = None, data = None, is_pure = False, is_leaf = False, is_empty = False):
        self.feature = feature          # Feature to split on, if None Node is leaf
        self.threshold = threshold      # Split threshold
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.prediction = prediction    # Predicted value at a leaf
        self.number = number            # Number of leaf (root starting at 0)
        self.parent = parent            # save parent node object here
        self.depth = depth              # depth of the node (in final tree)
        self.datapoints_in_node = data  # dataframe of datapoints having made it to this node
        self.is_pure = is_pure
        self.is_leaf = is_leaf 
        self.is_empty = is_empty         