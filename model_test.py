import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cProfile
from rolling_lookahead_dt_pulp import rollo_oct_pulp
import os
import time
import pickle

depth_rolling_tree = 8
time_limit = 1800
criterion_loss = "gini"

#data = pd.read_csv("datasets/nursery/nursery_bin.csv")
data = pd.read_csv("datasets/microbiome_taxa_counts_easy/microbiome_taxa_counts_easy_bin.csv")

# maybe also try m.solve(PULP_CBC_CMD(msg=True, timeLimit=time_limit, options=['presolve off']))
train_df, test_df = train_test_split(data, test_size=0.05, train_size = 0.1, stratify=data['y'], random_state=42, big_m = 1)

feature_columns = train_df.columns[1:]

# Create the directory if it doesn't exist
os.makedirs(f'results/microbiome_taxa_easy/test', exist_ok=True)
os.makedirs(f'predict_model_dicts/test/microbiome_taxa_easy', exist_ok=True)

result_dict_pulp =rollo_oct_pulp.run(train=train_df, test=test_df, target_label="y", features=feature_columns, depth=depth_rolling_tree, criterion=criterion_loss)

for depth in range(2,depth_rolling_tree+1):
    with open(f'results/microbiome_taxa_easy/test/depth{depth}_classification_microbiome_taxa_easy_test.csv', 'w') as f:
        f.write(str(result_dict_pulp['tree'][depth]['test'].to_csv()))
    with open(f'results/microbiome_taxa_easy/test/depth{depth}_classification_microbiome_taxa_easy_train.csv', 'w') as f:
        f.write(str(result_dict_pulp['tree'][depth]['train'].to_csv()))
    with open(f'predict_model_dicts//test/microbiome_taxa_easy/predict_dict_{depth}.pkl', 'wb') as f: # 'wb' for write binary, rb for read binary
        pickle.dump(result_dict_pulp['tree'][depth]['trained_dict'], f)