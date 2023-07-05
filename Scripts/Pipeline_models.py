#import packages for pipeline
import pandas as pd
import numpy as np
import math
import random
import torch
import sklearn
import argparse
from distutils.util import strtobool

#import self-written functions
import performance_models as pm
import prep_data
import synthcity

import os
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

#input from user to select datasets and model
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Evaluation procedure to select optimal set of hyperparameters and model')
parser.add_argument("dataset", help='Name of dataset',type=str,default='iris')
parser.add_argument("model", help='Model to generate synthetic data',type=str,default='ctgan')
parser.add_argument("nr_combinations",help='# of trials for various hyperparameter settings',type=int,default=1)
parser.add_argument("outliers",help='Investigate whether dataset contains outliers',type=lambda x: bool(strtobool(x)),default=True)

args = parser.parse_args()
trials = args.nr_combinations
dataset = args.dataset
model = args.model
outliers_bool = args.outliers
data_folder = '../Data'
cat_columns_dict = {
    "iris": ['class'],
    "bank" : ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y'], 
    "metro" : ['holiday','weather_main','weather_description'], #regression predictor: 'traffic_volume'
    "adult": ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income'],
    "covertype": ['Soil_type','Warea','Cover_Type']
}

if dataset:
    if dataset == 'covertype': #due to the size limit
        data_path = data_folder + '/' + dataset + '/' + dataset + '.zip'
    else:
        data_path = data_folder + '/' + dataset + '/' + dataset + '.csv' 
    cat_columns = cat_columns_dict[dataset]
    if dataset == 'metro': #deze regel kan verwijderd als een json file wordt ingeladen voor elke data met zn properties
        target = 'traffic_volume'
    else:
        target = cat_columns[-1]
    real_train_data = prep_data.main(data_path,dataset,cat_columns,target,outliers_bool)
    output_path = data_folder + '/' + dataset + '/' + model
    if model == 'arf':
        synthetic_arf = pm.tune_performance_arf(dataset,real_train_data,str(output_path),cat_columns,trials)
    elif model == 'cart':
        synthetic_cart = pm.tune_performance_cart(dataset,real_train_data,str(output_path),cat_columns,trials)
    elif model == 'ctgan':
        synthetic_ctgan = pm.tune_performance_ctgan(dataset,real_train_data,target,output_path,trials)
    elif model == 'tvae':
        synthetic_tvae = pm.tune_performance_tvae(dataset,real_train_data,target,output_path,trials)
    elif model == 'tabddpm':
        synthetic_ddpm = pm.tune_performance_ddpm(dataset,real_train_data,target,output_path,trials)

