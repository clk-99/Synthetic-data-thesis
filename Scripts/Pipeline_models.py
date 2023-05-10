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
    real_train_data, table_dict = prep_data.main(data_path,args.dataset,cat_columns,'/BANK_MARKETING/BANK_TRAIN_SET.csv')
    table_metadata = SingleTableMetadata.load_from_dict(table_dict)
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

if args.dataset == 'metro':
    data_path = data_folder + '/TRAFFIC_VOLUME/Metro_Interstate_Traffic_Volume.csv'
    cat_columns =  ['holiday','weather_main','weather_description'] 
    real_train_data, table_dict = prep_data.main(data_path,args.dataset,cat_columns,'/TRAFFIC_VOLUME/TRAFFIC_VOLUME_TRAIN_SET.csv') #including target variable
    table_metadata = SingleTableMetadata.load_from_dict(table_dict)
    print(table_metadata)
    if args.model == 'ctgan':
       output_path = data_folder + '/TRAFFIC_VOLUME/CTGAN'
       synthetic_ctgan = pm.tune_performance_ctgan('metro',real_train_data,table_metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/TRAFFIC_VOLUME/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('metro',real_train_data,table_metadata,None,output_path)
    if args.model == 'arf':
        output_path = data_folder + '/TRAFFIC_VOLUME/ARF/'
        synthetic_arf = pm.tune_performance_arf('metro',real_train_data,str(output_path),cat_columns)
    if args.model == 'cart':
        output_path = data_folder + '/TRAFFIC_VOLUME/CART/'
        synthetic_cart = pm.tune_performance_cart('metro',real_train_data,str(output_path),cat_columns)
    if args.model == 'tabddpm':
        output_path = data_folder + '/TRAFFIC_VOLUME/TABDDPM'
        print('d')

if args.dataset == 'adult':
    data_path = data_folder + '/ADULT_CENSUS_INCOME/adult.csv'
    cat_columns = ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income'] 
    real_train_data, table_dict = prep_data.main(data_path,args.dataset,cat_columns,'/ADULT_CENSUS_INCOME/ADULT_TRAIN_SET.csv')
    table_metadata = SingleTableMetadata.load_from_dict(table_dict)
    if args.model == 'ctgan':
       output_path = data_folder + '/ADULT_CENSUS_INCOME/CTGAN'
       synthetic_ctgan = pm.tune_performance_ctgan('adult',real_train_data,table_metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/ADULT_CENSUS_INCOME/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('adult',real_train_data,table_metadata,None,output_path)
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
    cat_columns = ['free sulfur dioxide','quality']
    real_train_data, table_dict = prep_data.main(data_path,args.dataset,cat_columns,'/WINE_QUALITY/WINE_TRAIN_SET.csv')
    table_metadata = SingleTableMetadata.load_from_dict(table_dict)
    if args.model == 'ctgan':
       output_path = data_folder + '/WINE_QUALITY/CTGAN'
       synthetic_ctgan = pm.tune_performance_ctgan('wine',real_train_data,table_metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/WINE_QUALITY/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('wine',real_train_data,table_metadata,None,output_path)
    if args.model == 'arf':
        output_path = data_folder + '/WINE_QUALITY/ARF/'
        synthetic_arf = pm.tune_performance_arf('wine',real_train_data,str(output_path),cat_columns)
    if args.model == 'cart':
        output_path = data_folder + '/WINE_QUALITY/CART/'
        synthetic_cart = pm.tune_performance_cart('wine',real_train_data,str(output_path),cat_columns)
    if args.model == 'tabddpm':
        output_path = data_folder + '/WINE_QUALITY/TABDDPM'
        print('d')
