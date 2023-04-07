import pandas as pd
import numpy as np
import seaborn as sn
import math
import random
import torch
import sklearn
import argparse
import tabsyndex as ts

parser = argparse.ArgumentParser(description='Evaluate the performance for each model per dataset using TabSynDex')
parser.add_argument("dataset", help='Name of dataset to convert filetype',type=str)
parser.add_argument("path_orig_data", help="Directory containing folder where original data file is saved", type=str)
parser.add_argument("path_syn_data", help="Directory containing folder where synthetic data file is saved", type=str)
parser.add_argument("metric_type", help="Evaluation metric to compute", type=str)

args = parser.parse_args()

if args.dataset == 'bank':
    path_orig_data = args.path_orig_data
    path_syn_data = args.path_syn_data
    if args.metric_type == 'TabSynDex':
        real_data = pd.read_csv(path_orig_data)
        synthetic_data = pd.read_csv(path_syn_data)
        cat_columns = real_data.select_dtypes(exclude=['int','float']).columns.tolist()
        scores = ts.tabsyndex(real_data, synthetic_data, cat_cols=cat_columns,target_type='class')

