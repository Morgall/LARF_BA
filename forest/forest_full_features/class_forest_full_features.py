import numpy as np
import pandas as pd

#from sklearn.base import BaseEstimator, ClassifierMixin, clone

from scipy.stats import mode
import multiprocessing

from forest.forest_full_features.class_tree_full_features import CustomTreeWrapper

def fit_single_tree(args):
    X, y, tree_kwargs, random_state = args
    rng = np.random.RandomState(random_state)
    indices = np.array(X.index)
    sample_indices = rng.choice(indices, size=len(indices), replace=True)
    #sample_indices = rng.choice(indices, size=int(np.sqrt(len(indices))), replace=False) # Picks len(indices) samples with replacement => means some rows will repeat, others will be omitted → those omitted are "out-of-bag" (OOB)
    oob_mask = ~np.isin(indices, sample_indices)
    oob_indices = indices[oob_mask]

    X_boot = X.loc[sample_indices]
    y_boot = y.loc[sample_indices]
    train_data = pd.concat([y_boot, X_boot], axis=1)
    train_data.columns = ['y'] + list(range(X.shape[1]))

    if len(oob_indices) > 0:
        X_oob = X.loc[oob_indices]
        y_oob = y.loc[oob_indices]
        test_data = pd.concat([y_oob, X_oob], axis=1)
        test_data.columns = ['y'] + list(range(X.shape[1]))
    else:
        test_data = train_data.copy()

    tree = CustomTreeWrapper(train_data=train_data, test_data=test_data, **tree_kwargs)
    tree.fit(X_boot, y_boot)
    return tree, (sample_indices, oob_indices)


class CustomEnsembleClassifier:
    def __init__(self, n_estimators=10, tree_kwargs=None, random_state=None, cores_to_use = 1):
        """
        n_estimators: number of trees in the ensemble
        tree_kwargs: dictionary of keyword args for CustomTreeWrapper (except train_data and test_data)
        random_state: seed for reproducible bootstrap sampling
        """
        self.n_estimators = n_estimators
        self.tree_kwargs = tree_kwargs if tree_kwargs is not None else {}
        self.random_state = random_state
        self.trees_ = []
        self.bootstrap_indices_ = []
        self.cores_to_use = cores_to_use



    def fit(self, X, y):
        """
        # this should gibe option to just give full dataset split into X and y or to do extra preprocessing first (eg crossvalidation) if needed
        X: pd.DataFrame of features
        y: pd.Series of target labels
        """
        
        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        self.bootstrap_indices_ = []

        indices = np.array(X.index)
        for i in range(self.n_estimators):
            # Bootstrap sample indices
            # returns an array of row indices (can repeat).
            sample_indices = rng.choice(indices, size=len(indices), replace=True) # Picks len(indices) samples with replacement => means some rows will repeat, others will be omitted → those omitted are "out-of-bag" (OOB)
            #sample_indices = rng.choice(indices, size=int(np.sqrt(len(indices))), replace=False) # Picks len(indices) samples with replacement => means some rows will repeat, others will be omitted → those omitted are "out-of-bag" (OOB)
            #oob_mask finds rows not included in sample_indices
            #oob_mask = ~np.in1d(indices, sample_indices) #True for each index not present in the bootstrap sample
            oob_mask = ~np.isin(indices, sample_indices) #True for each index not present in the bootstrap sample
            
            oob_indices = indices[oob_mask] # out-of-bag (OOB) indices for the bootstrap sampling process
            self.bootstrap_indices_.append((sample_indices, oob_indices)) #for this tree: indices chosen as the tree's bootstrap (training) sample; indices not chosen, used for out-of-bag validation

            # Create train_data DataFrame: target as first column, features with integer columns
            #Selects bootstrap feature rows for training
            X_boot = X.loc[sample_indices]
            y_boot = y.loc[sample_indices]
            train_data = pd.concat([y_boot, X_boot], axis=1)
            train_data.columns = ['y'] + list(range(X.shape[1])) # Feature columns are renamed to integers 0, 1, 2, ... so that the underlying tree implementation gets standardized input

            # Out-of-bag for test_data
            if len(oob_indices) > 0:
                X_oob = X.loc[oob_indices]
                y_oob = y.loc[oob_indices]
                test_data = pd.concat([y_oob, X_oob], axis=1)
                test_data.columns = ['y'] + list(range(X.shape[1]))
            else:
                # If somehow no oob sample, just use train_data (edge case)
                test_data = train_data.copy()

            # Initialize and fit the tree
            tree = CustomTreeWrapper(train_data=train_data,
                                     test_data=test_data,
                                     **self.tree_kwargs) # ** operator "unpacks" a dictionary so that each key-value pair becomes a separate keyword 
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self
    


    def parallel_fit(self, X, y):
        #num_cores_to_use = 6
        pool = multiprocessing.Pool(processes=self.cores_to_use)
        random_states = [self.random_state + i for i in range(self.n_estimators)] if self.random_state is not None else [None]*self.n_estimators
        args = [(X, y, self.tree_kwargs, rs) for rs in random_states]

        results = pool.map(fit_single_tree, args)
        pool.close()
        pool.join()

        self.trees_ = [res[0] for res in results] #rebuilds the classifier’s internal list of trained trees from the results returned by pool.map()
        self.bootstrap_indices_ = [res[1] for res in results]
        return self
  

    def predict(self, X):
        """
        Majority-vote ensemble prediction.
        Returns: pd.Series with predictions, aligned to X.index
        """
        # Aggregate predictions (each as Series aligned to X.index)
        all_preds = pd.DataFrame()
        for tree in self.trees_:
            pred = tree.predict(X)
            all_preds = pd.concat([all_preds, pred], axis=1)
        # row-wise majority vote (handle multiple modes by picking first)
        maj_vote = all_preds.mode(axis=1)[0]
        maj_vote.index = X.index  # ensure correct alignment
        maj_vote = maj_vote.astype(int)
        return maj_vote

    def predict_proba(self, X):
        """
        For binary classification:
        Returns an array of shape (n_samples, 2)
        """
        all_preds = []
        for tree in self.trees_:
            pred = tree.predict(X)
            if not isinstance(pred, pd.Series):
                pred = pd.Series(pred, index=X.index)
            all_preds.append(pred)
        preds_matrix = pd.concat(all_preds, axis=1)

        # Works for binary or multiclass
        classes_ = np.unique(preds_matrix.values)
        proba = np.zeros((X.shape[0], len(classes_)))
        for i, c in enumerate(classes_):
            proba[:, i] = (preds_matrix == c).sum(axis=1) / self.n_estimators
        return proba

    def oob_score(self, X, y):
        """
        Returns out-of-bag score.
        """
        # Prepare OOB predictions
        oob_votes = {idx: [] for idx in X.index}
        for (sample_ind, oob_ind), tree in zip(self.bootstrap_indices_, self.trees_):
            if len(oob_ind) == 0:
                continue
            X_oob = X.loc[oob_ind]
            preds = tree.predict(X_oob)
            for idx, pred in preds.items():
                oob_votes[idx].append(pred)
        # Only score samples with at least one OOB prediction
        final_oob_preds = []
        final_oob_true = []
        for idx, votes in oob_votes.items():
            if votes:
                final_oob_preds.append(mode(votes)[0][0])
                final_oob_true.append(y.loc[idx])
        if not final_oob_preds:
            raise ValueError("No OOB predictions collected.")
        accuracy = np.mean(np.array(final_oob_preds) == np.array(final_oob_true))
        return accuracy