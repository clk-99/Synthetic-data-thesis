#import packages for pipeline
import sdv
import pandas as pd
import numpy as np
import seaborn as sn
import math
import random
import torch
import sklearn
import argparse
import performance_models as pm

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

#input from user to select datasets and model
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Evaluation procedure to select optimal set of hyperparameters and model')
parser.add_argument("dataset", help='Name of dataset',type=str)
parser.add_argument("model", help='Model to generate synthetic data',type=str)
#parser.add_argument("path_data", help="Directory containing folder where data file is saved", type=str)

args = parser.parse_args()


if args.dataset == 'bank':
    data_path = './LISA/Data/BANK CUSTOMER CHURN/Bank Customer Churn Prediction.csv'
    df_bank = pd.read_csv(data_path, delimiter=',')
    if args.model == 'ctgan':
        print('a')
    if args.model == 'tvae':
        print('a')
    if args.model == 'arf':
        print('b')
    if args.model == 'cart':
        print('c')
