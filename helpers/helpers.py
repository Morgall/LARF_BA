import numpy as np
import pandas as pd
import logging


def train_validation_test_split(data: pd.DataFrame,
                                train_percent: float = .5,
                                validation_percent: float = .25,
                                seed: float = None) -> pd.DataFrame:
    np.random.seed(seed)
    perm = np.random.permutation(data.index)
    m = len(data.index)
    train_end = int(train_percent * m)
    validate_end = int(validation_percent * m) + train_end
    train_df = data.iloc[perm[:train_end]]
    validation_df = data.iloc[perm[train_end:validate_end]]
    test_df = data.iloc[perm[validate_end:]]
    return train_df, validation_df, test_df

def preprocess_numerical(data: pd.DataFrame, target_label='y') -> pd.DataFrame:
    excluded_cols = [cname for cname in data.columns if data[cname].nunique() == 2] # list; excluded cols are those with target vars (not to be categorized) and cols with binary data
    excluded_cols = list(set(excluded_cols + [target_label])) #adds target col to excluded cols and handles case that target col is binary
    int_cols_non_binary = [col for col in data.select_dtypes(include=['int']).columns 
                      if col not in excluded_cols]
    float_cols_non_binary = [col for col in data.select_dtypes(include=['float']).columns 
                      if col not in excluded_cols]
    numeric_cols_non_binary = int_cols_non_binary + float_cols_non_binary
    for col in numeric_cols_non_binary:
        unique_vals = data[col].nunique()
        if unique_vals < 7:
            data[col] = data[col].astype('category')
        else:
            #data[col] = pd.cut(data[col], bins=4, labels=False, duplicates='drop') #pd.cut does work for quantiles for us
            data[col] = pd.qcut(data[col], q=5, labels=False, duplicates='drop')
    return data


# moves cloumn with target labels to first column and renames it to 'y' and return reordered dataframe
def move_targets_to_front_and_rename(data: pd.DataFrame, target_label='y') -> pd.DataFrame:
    data.rename(columns={target_label: 'y'}, inplace=True)
    if data.columns[0] != 'y': # Checks if 'y' is not the first column; hier wurde vorher aus irgendeinem Grund data.columns[-1] != "y" geprüft, also ob 'y' in letzter Spalte war
        logging.info("Reordering y column at the beginning of data.")
        cols_ = list(data.columns)
        cols_.remove('y')
        cols_.insert(0, 'y')
        data = data[cols_]
    return data



# The function transforms input data (a pandas DataFrame) into a binary/one-hot encoded format suitable for certain machine learning tasks.
# It handles missing values, binary columns, categorical columns, and the target column separately.
# needs target label column to be named 'y'

def make_data_binary(data: pd.DataFrame) -> pd.DataFrame:
    """

    :param data: input data
    :return: data with binary columns
    """
    cols_with_missing = [col for col in data.columns #Identify Columns with Missing Values
                         if data[col].isnull().any()]
    if cols_with_missing:
        for col in cols_with_missing:
            data[col].fillna(data[col].mode()[0], inplace=True) # Replace Missing Values with Mode: For each column with missing values, fill them with the most frequent value (mode).
        logging.info("""There are columns with missing
            values.\nColumns are: {0}\n Replacing with mode. 
            """.format(cols_with_missing))
    
    

    binary_cols = [cname for cname in data.columns if # Find Columns with Exactly 2 Unique Values (excluding 'y');  list of column names that have exactly two unique values (excluding 'y')
                   data[cname].nunique() == 2 and cname != 'y']

    for col in binary_cols: # Convert Non-Integer Binary Columns to Integer: If a binary column is not already an integer type, convert it using category codes.
        if data[col].dtype not in ['int8', 'int16']: # check data type
            logging.info(f"Column {col} is not int type. Transforming it into "
                         f"integer.")
            data[col] = data[col].astype('category').cat.codes
            # astype('category'): Converts the column to a pandas "category" type for categorical variables (integer codes; cat.codes) (limited, fixed set of possible values). This is useful for columns with a small number of unique values.

    # Ensure Binary Columns are 0 and 1: If the unique values do not sum to 1 (i.e., not already 0 and 1), remap them to 0 and 1.
    if binary_cols: 
        for col in binary_cols:
            # if sum of unique entries is not equal to 1
            if sum(data[col].unique()) != 1:
                replace = {data[col].unique()[0]: 0, # Create a mapping dictionary where the first unique value maps to 0 and the second to 1. Apply this mapping to the entire column, converting all values to 0 or 1.
                           data[col].unique()[1]: 1} # Example: If the column has values ['A', 'B'], it will be mapped to {'A': 0, 'B': 1}
                data[col] = [replace[item] for item in data[col]]
        logging.info("There are {0} binary columns. \nColumns are: {1}".format(
            len(binary_cols), binary_cols))
    else:
        logging.info("No binary columns.")
        # so now it is ensured that binary columns contain strict binary (0/1) encoding


    total_col = 0 #expected total number of columns after all transformations; total number of columns after one-hot encoding and other transformations


    # Log the Number of Unique Values for Each Non-Target Column
    # non-binary columns (excluding 'y'), sum the number of unique values
    # This is used to check if one-hot encoding later produces the expected number of columns.
    for col in data.columns:
        if col != "y":
            logging.info("Column: {0} - Unique Values: {1}".format( #For each non-target column, log its name and the number of unique values it contains.
                col, data[col].nunique()))
            if data[col].nunique() != 2:
                total_col += data[col].nunique() #if a column does NOT have exactly 2 unique values (i.e., it’s not a binary column), add its number of unique values to total_col.
    total_col += len(binary_cols) + 1
    # This is preparation for one-hot encoding:
    # For categorical columns, one-hot encoding will create as many new columns as there are unique values.
    # For binary columns, no extra columns are needed beyond the original (since they are already handled separately).


    #the one hot encoding
    for col in data.columns:
        if col not in binary_cols and col != "y": #for Each Non-Binary, Non-Target Column:
            dummy_col = pd.get_dummies(data[col], prefix=col,dtype=int) #Creates a new DataFrame (dummy_col) where the original column is split into multiple binary (0/1) columns, one for each unique value in the original column
            data = pd.concat([data, dummy_col], axis=1) # adds new dummy columns to the original DataFrame
            data = data.drop(col, axis=1) # Removes the original column from the DataFrame, since it got replaced by one-hot encoding

    if total_col != data.shape[1]: # If the expected number of columns does not match the actual, log an error and return None
        logging.error("# of expected column is not equal to actual.")
        return None

    # Convert Target Column 'y' to Categorical Codes
    if data.y.dtype == "O": # checks if the column 'y' is of type object (usually strings or mixed types)
        logging.info("Converting y values into numerical.")
        data['y'] = data['y'].astype('category') # converts the column to a categorical type
        data['y'] = data['y'].cat.codes # replaces each unique value in 'y' with a numerical code (e.g., "cat" → 0, "dog" → 1, etc.)

    # ensure Target Values Are At Least 1, if not shift all values up by 1
    if data.y.min() < 1:
        data['y'] = data['y'] + 1
    # Reason: Some algorithms or libraries expect labels to start at 1 rather than 0

    # move Target Column 'y' to the Front
    if data.columns[0] != "y": # Checks if 'y' is not the first column; hier wurde vorher aus irgendeinem Grund data.columns[-1] != "y" geprüft, also ob 'y' in letzter Spalte war
        logging.info("Reordering y column at the beginning of data.")
        cols_ = list(data.columns)
        cols_.remove('y')
        cols_.insert(0, "y")
        data = data[cols_]

    # Rename All Columns Except 'y' to 1, 2, 3, ..., based on their position
    logging.info("Renaming columns..")
    column_indices = [i for i in range(1, len(data.columns))]
    new_names = column_indices
    old_names = data.columns[column_indices]
    data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    logging.info("Data is binarized.")
    return data


def preprocess_dataframes(train_df: pd.DataFrame, test_df: pd.DataFrame, target_label: str, features: list):
    """
    Rearranges the DataFrames such that the target label becomes the first column,
    and feature names are converted into ordinal numbers.

    Args:
    - train_df: pandas DataFrame containing the training data
    - test_df: pandas DataFrame containing the test data
    - target_label: string representing the target label
    - features: list of strings representing feature names

    Returns:
    - pd.DataFrame: preprocessed training DataFrame
    - pd.DataFrame: preprocessed test DataFrame
    """

    # Move target label to the first column for both train and test DataFrames
    if target_label in train_df.columns:
        train_target_idx = train_df.columns.get_loc(target_label) # find the integer position (index) of the column named target_label within the DataFrame train_df
        train_df_columns = list(train_df.columns)
        train_df_columns = [train_df_columns[train_target_idx]] + train_df_columns[:train_target_idx] + train_df_columns[train_target_idx + 1:] # new arrangement
        train_df = train_df[train_df_columns] # returns a new DataFrame containing only those columns, arranged in the order specified by the list, panmda reordering

    if target_label in test_df.columns:
        test_target_idx = test_df.columns.get_loc(target_label)
        test_df_columns = list(test_df.columns)
        test_df_columns = [test_df_columns[test_target_idx]] + test_df_columns[:test_target_idx] + test_df_columns[test_target_idx + 1:]
        test_df = test_df[test_df_columns]

    # Rename features to ordinal numbers for both train and test DataFrames
    train_df.rename(columns={feature: str(i) for i, feature in enumerate(features, start=1)}, inplace=True) #  DataFrame columns renaming ['y', 'age', 'income', 'score', ...] to ['y', '1', '2', '3', ...]
    test_df.rename(columns={feature: str(i) for i, feature in enumerate(features, start=1)}, inplace=True)

    return train_df, test_df
