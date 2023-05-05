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
parser.add_argument("dataset", help='Name of dataset',type=str,default='heart')
parser.add_argument("model", help='Model to generate synthetic data',type=str,default='ctgan')

args = parser.parse_args()
data_folder = '../Data'
#toevoegen van kleine data analyse

if args.dataset == 'bank':
    data_path = data_folder + '/BANK_MARKETING/bank-additional-full.csv'
    cat_columns = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
    real_train_data, target = prep_data.main(data_path,args.dataset,cat_columns)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_train_data)
    table_metadata = metadata.to_dict()
    if args.model == 'ctgan':
        output_path = data_folder + '/BANK_MARKETING/CTGAN'
        synthetic_ctgan = pm.tune_performance_ctgan('bank',real_train_data,table_metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/BANK_MARKETING/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('bank',real_train_data,table_metadata,None,output_path)
    if args.model == 'arf':
        output_path = data_folder + '/BANK_MARKETING/ARF/'
        synthetic_arf = pm.tune_performance_arf('bank',real_train_data,str(output_path),cat_columns)
    if args.model == 'cart':
        output_path = data_folder + '/BANK_MARKETING/CART/'
        synthetic_cart = pm.tune_performance_cart('bank',real_train_data,str(output_path),cat_columns)
    if args.model == 'tabddpm':
        output_path = data_folder + '/BANK_MARKETING/TABDDPM'
        print('d')

if args.dataset == 'heart':
    data_path = data_folder + '/HEART_ATTACK_PREDICTION/heart.csv'
    cat_columns =  ['sex','cp','fbs','restecg','exang','slope','thal','target']
    real_train_data, target = prep_data.main(data_path,args.dataset,cat_columns) #new adjustment of code!
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_train_data)
    table_metadata = metadata.to_dict()
    if args.model == 'ctgan':
       output_path = data_folder + '/HEART_ATTACK_PREDICTION/CTGAN'
       synthetic_ctgan = pm.tune_performance_ctgan('heart',real_train_data,metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/HEART_ATTACK_PREDICTION/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('heart',real_train_data,metadata,None,output_path)
    if args.model == 'arf':
        output_path = data_folder + '/HEART_ATTACK_PREDICTION/ARF/'
        synthetic_arf = pm.tune_performance_arf('heart',real_train_data,str(output_path),cat_columns)
    if args.model == 'cart':
        output_path = data_folder + '/HEART_ATTACK_PREDICTION/CART/'
        synthetic_cart = pm.tune_performance_cart('heart',real_train_data,str(output_path),cat_columns)
    if args.model == 'tabddpm':
        output_path = data_folder + '/HEART_ATTACK_PREDICTION/TABDDPM'
        print('d')

if args.dataset == 'adult':
    data_path = data_folder + '/ADULT_CENSUS_INCOME/adult.csv'
    cat_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','class'] 
    real_train_data, target = prep_data.main(data_path,args.dataset,cat_columns)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_train_data)
    table_metadata = metadata.to_dict()
    if args.model == 'ctgan':
       output_path = data_folder + '/ADULT_CENSUS_INCOME/CTGAN'
       synthetic_ctgan = pm.tune_performance_ctgan('adult',real_train_data,metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/ADULT_CENSUS_INCOME/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('adult',real_train_data,metadata,None,output_path)
    if args.model == 'arf':
        output_path = data_folder + '/ADULT_CENSUS_INCOME/ARF/'
        synthetic_arf = pm.tune_performance_arf('adult',real_train_data,str(output_path),cat_columns)
    if args.model == 'cart':
        output_path = data_folder + '/ADULT_CENSUS_INCOME/CART/'
        synthetic_cart = pm.tune_performance_cart('adult',real_train_data,str(output_path),cat_columns)
    if args.model == 'tabddpm':
        output_path = data_folder + '/ADULT_CENSUS_INCOME/TABDDPM'
        print('d')

if args.dataset == 'wine':
    data_path = data_folder + '/WINE_QUALITY/wine.csv' 
    cat_columns = ['quality'] 
    real_train_data, target = prep_data.main(data_path,args.dataset,cat_columns)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_train_data)
    table_metadata = metadata.to_dict()
    if args.model == 'ctgan':
       output_path = data_folder + '/WINE_QUALITY/CTGAN'
       synthetic_ctgan = pm.tune_performance_ctgan('wine',real_train_data,metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/WINE_QUALITY/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('wine',real_train_data,metadata,None,output_path)
    if args.model == 'arf':
        output_path = data_folder + '/WINE_QUALITY/ARF/'
        synthetic_arf = pm.tune_performance_arf('wine',real_train_data,str(output_path),cat_columns)
    if args.model == 'cart':
        output_path = data_folder + '/WINE_QUALITY/CART/'
        synthetic_cart = pm.tune_performance_cart('wine',real_train_data,str(output_path),cat_columns)
    if args.model == 'tabddpm':
        output_path = data_folder + '/WINE_QUALITY/TABDDPM'
        print('d')
