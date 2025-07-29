import pandas as pd
import time
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, matthews_corrcoef

# Go up one directory to get to master/
project_root = str(Path.cwd().parent)
sys.path.append(project_root)


def get_solutions_all_folds_rollOCT(name_dataset: str, depth=3, folds_available = 10) -> dict: #for 10 fold cross valdidation, carefull that trees have min depth 2
    project_root = str(Path.cwd().parent)
    sys.path.append(project_root)
    sol_dict = {} # one entry for every fold, [fold][depth]['test'] for train_data classification for fold of depth; same for training
    for i in range(1,folds_available+1):
        sol_dict[i] = {}
        for j in range(2,depth+1):
            sol_dict[i][j] = {}
            sol_dict[i][j]['test'] = pd.read_csv(f"./results/{name_dataset}/pulp/fold{i}/depth{j}_classification_{name_dataset}_test.csv")
            sol_dict[i][j]['train'] = pd.read_csv(f"./results/{name_dataset}/pulp/fold{i}/depth{j}_classification_{name_dataset}_train.csv")
    return sol_dict


def get_solutions_all_folds_cart(name_dataset: str, depth=3, folds_available = 10) -> dict: #for 10 fold cross valdidation, carefull that trees have min depth 2
    project_root = str(Path.cwd().parent)
    sys.path.append(project_root)
    sol_dict = {} # one entry for every fold, [fold][depth]['test'] for train_data classification for fold of depth; same for training
    for i in range(1,folds_available+1):
        sol_dict[i] = {}
        for j in range(2,depth+1):
            sol_dict[i][j] = {}
            sol_dict[i][j]['test'] = pd.read_csv(f"./results/{name_dataset}/cart/fold{i}/depth{j}_classification_{name_dataset}_test.csv")
            sol_dict[i][j]['train'] = pd.read_csv(f"./results/{name_dataset}/cart/fold{i}/depth{j}_classification_{name_dataset}_train.csv")
    return sol_dict


def get_class_confusion_values(cm, class_idx):
    tp = cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - tp
    fp = cm[:, class_idx].sum() - tp
    tn = cm.sum() - (tp + fp + fn)
    return tp, fp, fn, tn


def calc_acc_sens_spec_prec(tp, fp, fn, tn):
    #number_datapoints = tp+tn+fp+fn
    sensitivity = tp/(tp+fn) #also called recall, True Positive Rate
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    #accuracy = (tp+tn)/number_datapoints
    #return accuracy, sensitivity, specificity,precision
    return sensitivity, specificity,precision

def calc_f1(result_dict, list_target_vars, max_tree_depth):
    for depth in range(2,max_tree_depth+1):
        for i,class_idx in enumerate(list_target_vars):
            precision_test = result_dict[depth]['test'][class_idx]['precision']
            precision_train = result_dict[depth]['train'][class_idx]['precision']

            sensitivity_test = result_dict[depth]['test'][class_idx]['sensitivity'] #also called recall
            sensitivity_train = result_dict[depth]['train'][class_idx]['sensitivity']
            
            result_dict[depth]['test'][class_idx]['f1'] = 2*((precision_test*sensitivity_test)/(precision_test+sensitivity_test))
            result_dict[depth]['train'][class_idx]['f1'] = 2*((precision_train*sensitivity_train)/(precision_train+sensitivity_train))
    return result_dict

# does not contain time stuff
# we want result_dict[depth]['test'/'train'][target_var] and then sens,spec,prec,acc for all folds. Then we are able to combine it with original target var afterwards
def solutions_all_depths_all_folds_multiclass(dataset_name, list_target_vars, max_tree_depth, folds_available, cart = False):
    if cart == False:
        sol_dict = get_solutions_all_folds_rollOCT(name_dataset = dataset_name, depth=max_tree_depth, folds_available = folds_available)
    else:
        sol_dict = get_solutions_all_folds_cart(name_dataset = dataset_name, depth=max_tree_depth, folds_available = folds_available)
    result_dict = dict()
    for depth in range(2,max_tree_depth+1):
        result_dict[depth] = dict()
        result_dict[depth]['test'] = dict()
        result_dict[depth]['train'] = dict()

        result_dict[depth]['test']['accuracy'] = 0
        result_dict[depth]['train']['accuracy'] = 0

        mcc_scores_test = []
        mcc_scores_train = []

        for i,class_idx in enumerate(list_target_vars): #initialize per depth
                result_dict[depth]['test'][class_idx] = dict()
                result_dict[depth]['train'][class_idx] = dict()

                #result_dict[depth]['test'][class_idx]['accuracy'] = 0
                result_dict[depth]['test'][class_idx]['sensitivity'] = 0
                result_dict[depth]['test'][class_idx]['specificity'] = 0
                result_dict[depth]['test'][class_idx]['precision'] = 0
                result_dict[depth]['test'][class_idx]['f1'] = 0

                #result_dict[depth]['train'][class_idx]['accuracy'] = 0
                result_dict[depth]['train'][class_idx]['sensitivity'] = 0
                result_dict[depth]['train'][class_idx]['specificity'] = 0
                result_dict[depth]['train'][class_idx]['precision'] = 0
                result_dict[depth]['train'][class_idx]['f1'] = 0

        for fold in range(1, folds_available+1):
            
            y_true_test = sol_dict[fold][depth]['test']['y']
            y_predict_test = sol_dict[fold][depth]['test']['prediction']

            y_true_train = sol_dict[fold][depth]['train']['y']
            y_predict_train = sol_dict[fold][depth]['train']['prediction']

            cm_test = confusion_matrix(y_true_test, y_predict_test) #(true labels, predicted labels)
            cm_train = confusion_matrix(y_true_train, y_predict_train)

            result_dict[depth]['test']['accuracy'] += accuracy_score(y_true_test, y_predict_test) / folds_available
            result_dict[depth]['train']['accuracy'] += accuracy_score(y_true_train, y_predict_train) / folds_available
            
            mcc_test = matthews_corrcoef(y_true_test, y_predict_test) #For a multiclass problem, sklearn.metrics.matthews_corrcoef returns a single float value representing the overall Matthews correlation coefficient (MCC) across all classes
            mcc_train = matthews_corrcoef(y_true_train, y_predict_train)

            mcc_scores_test.append(mcc_test)
            mcc_scores_train.append(mcc_train)
            
            for j,class_idx in enumerate(list_target_vars): #every target class possible


                tp_test, fp_test, fn_test, tn_test = get_class_confusion_values(cm_test, class_idx-1) #-1 because conusion matrix cm starts indices with 0
                tp_train, fp_train, fn_train, tn_train = get_class_confusion_values(cm_train, class_idx-1)

                sens_test, spec_test, prec_test = calc_acc_sens_spec_prec(tp_test, fp_test, fn_test, tn_test)
                sens_train, spec_train, prec_train = calc_acc_sens_spec_prec(tp_train, fp_train, fn_train, tn_train)

                #result_dict[depth]['test'][class_idx]['accuracy'] += acc_test/folds_available #division here could maybe lead to precision problems
                result_dict[depth]['test'][class_idx]['sensitivity'] += sens_test/folds_available
                result_dict[depth]['test'][class_idx]['specificity'] += spec_test/folds_available
                result_dict[depth]['test'][class_idx]['precision'] += prec_test/folds_available

                #result_dict[depth]['train'][class_idx]['accuracy'] += acc_train/folds_available
                result_dict[depth]['train'][class_idx]['sensitivity'] += sens_train/folds_available
                result_dict[depth]['train'][class_idx]['specificity'] += spec_train/folds_available
                result_dict[depth]['train'][class_idx]['precision'] += prec_train/folds_available
        
        mean_mcc_test = np.mean(mcc_scores_test)
        mean_mcc_train = np.mean(mcc_scores_train)

        result_dict[depth]['test']['mcc'] = mean_mcc_test
        result_dict[depth]['train']['mcc'] = mean_mcc_train

        #std_mcc_test = np.std(mcc_scores_test)
        #std_mcc_train = np.std(mcc_scores_train)
    
    result_dict = calc_f1(result_dict= result_dict, list_target_vars= list_target_vars, max_tree_depth=max_tree_depth)
    
    return result_dict

def get_evaluation_for_class(result_dict, target_var, depth, training_data = False):
    rows_depths = []
    cols = ['sensitivity','specificity', 'precision', 'f1']
    for i in range(2,depth+1):
        rows_depths.append(f'depth{i}')
    df = pd.DataFrame([], index=rows_depths, columns=cols)
    for j in range(2,depth+1):
        if training_data == False:
            df.at[f'depth{j}', 'sensitivity'] = result_dict[j]['test'][target_var]['sensitivity']
            df.at[f'depth{j}', 'specificity'] = result_dict[j]['test'][target_var]['specificity']
            df.at[f'depth{j}', 'precision'] = result_dict[j]['test'][target_var]['precision']
            df.at[f'depth{j}', 'f1'] = result_dict[j]['test'][target_var]['f1']
        else:
            df.at[f'depth{j}', 'sensitivity'] = result_dict[j]['train'][target_var]['sensitivity']
            df.at[f'depth{j}', 'specificity'] = result_dict[j]['train'][target_var]['specificity']
            df.at[f'depth{j}', 'precision'] = result_dict[j]['train'][target_var]['precision']
            df.at[f'depth{j}', 'f1'] = result_dict[j]['train'][target_var]['f1']

    return df

def get_accuracy_every_depth(result_dict, depth):
    rows_depths = []
    cols = ['test','train']
    for i in range(2,depth+1):
        rows_depths.append(f'depth{i}')
    df = pd.DataFrame([], index=rows_depths, columns=cols)
    for j in range(2,depth+1):
        df.at[f'depth{j}', 'test'] = result_dict[j]['test']['accuracy']
        df.at[f'depth{j}', 'train'] = result_dict[j]['train']['accuracy']
    return df

def get_mcc_every_depth(result_dict, depth): #Matthews Correlation Coefficient
    rows_depths = []
    cols = ['test','train']
    for i in range(2,depth+1):
        rows_depths.append(f'depth{i}')
    df = pd.DataFrame([], index=rows_depths, columns=cols)
    for j in range(2,depth+1):
        df.at[f'depth{j}', 'test'] = result_dict[j]['test']['mcc']
        df.at[f'depth{j}', 'train'] = result_dict[j]['train']['mcc']
    return df

#for all metrics which consider all target vars in a single value
#for acc and mcc
def plot_on_ax(ax, x, categories, bar_width, dataset_var_rollOCT, dataset_var_cart, dataset_name, metric_name, metric_short, train=False):
    ax.bar(x - bar_width, dataset_var_rollOCT, width=bar_width, label='rollOCT')
    ax.bar(x, dataset_var_cart, width=bar_width, label='CART')
    ax.set_xticks(ticks=x, labels=categories)
    ax.set_xlabel('depth')
    ax.set_ylabel(metric_short)
    if train == False:
        ax.set_title(f'out-of-sample {metric_name}; dataset {dataset_name}')
    else:
        ax.set_title(f'in-sample {metric_name}; dataset {dataset_name}')
    ax.legend(loc='lower left')

#for all metrics which consider one target var in a single value
#for variable specific metric 
def plot_var_on_ax(ax, x, categories, bar_width, dataset_var_rollOCT, dataset_var_cart, dataset_name, metric_name, metric_short, target_var, train=False):
    ax.bar(x - bar_width, dataset_var_rollOCT, width=bar_width, label='rollOCT')
    ax.bar(x, dataset_var_cart, width=bar_width, label='CART')
    ax.set_xticks(ticks=x, labels=categories)
    ax.set_xlabel('depth')
    ax.set_ylabel(metric_short)
    if train == False:
        ax.set_title(f'{target_var}; out-of-sample {metric_name}; dataset {dataset_name}')
    else:
        ax.set_title(f'{target_var}; in-sample {metric_name}; dataset {dataset_name}')
    ax.legend(loc='lower left')

