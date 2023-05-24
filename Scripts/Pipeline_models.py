#import packages for pipeline
import sdv
import pandas as pd
import numpy as np
import math
import random
import torch
import sklearn
import argparse

#import self-written functions
import performance_models as pm
import prep_data 

import os
from pathlib import Path
from sdv.metadata import SingleTableMetadata

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

#input from user to select datasets and model
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Evaluation procedure to select optimal set of hyperparameters and model')
parser.add_argument("dataset", help='Name of dataset',type=str,default='bank')
parser.add_argument("model", help='Model to generate synthetic data',type=str,default='ctgan')
parser.add_argument("nr_combinations",help='# of trials for various hyperparameter settings',type=int,default=1)

args = parser.parse_args()
trials = args.nr_combinations
dataset = args.dataset
model = args.model
data_folder = '../Data'
cat_columns_dict = {
    "bank" : ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y'],
    "metro" : ['holiday','weather_main','weather_description'],
    "adult": ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income'],
    "covertype": ['Soil_type','Warea','Cover_Type']
}

if dataset:
    if dataset == 'covertype': #due to the size limit
        data_path = data_folder + '/' + dataset + '/' + dataset + '.zip'
    else:
        data_path = data_folder + '/' + dataset + '/' + dataset + '.csv' 

    cat_columns = cat_columns_dict[dataset]
    real_train_data, table_dict = prep_data.main(data_path,dataset,cat_columns)
    table_metadata = SingleTableMetadata.load_from_dict(table_dict)
    if args.model == 'ctgan':
        output_path = data_folder + '/' + dataset + '/CTGAN'
        synthetic_ctgan = pm.tune_performance_ctgan(dataset,real_train_data,table_metadata,None,output_path,trials)
    if args.model == 'tvae':
        output_path = data_folder + '/' + dataset + '/TVAE'
        synthetic_tvae = pm.tune_performance_tvae(dataset,real_train_data,table_metadata,None,output_path,trials)
    if args.model == 'arf':
        output_path = data_folder + '/' + dataset + '/ARF/'
        synthetic_arf = pm.tune_performance_arf(dataset,real_train_data,str(output_path),cat_columns,trials)
    if args.model == 'cart':
        output_path = data_folder + '/' + dataset + '/CART/'
        synthetic_cart = pm.tune_performance_cart(dataset,real_train_data,str(output_path),cat_columns,trials)
    if args.model == 'tabddpm':
        output_path = data_folder + '/' + dataset + '/TABDDPM'
        print('d')


#toevoegen van use case WW naar Bijstand
#wel opslaan onder andere naam (mag niet op de Github)
#probeer een kleine mini set aan hyperparameters
#input variabelen meenemen in de code: output path en dictionary voor categorische kolommen
