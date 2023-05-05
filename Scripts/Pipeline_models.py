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
data_folder_r = '~/Data'
#toevoegen van kleine data analyse

if args.dataset == 'bank':
    data_path = data_folder + '/BANK_CUSTOMER_CHURN/Bank_Customer_Churn_Prediction.csv'
    # file = open(data_path, encoding='utf8')
    # df_bank = pd.read_csv(file, sep=',')
    # df_bank.drop('customer_id',axis=1,inplace=True)
    # features = df_bank.iloc[:,:-1]
    # target = df_bank.iloc[:,-1]
    # X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.20,random_state=12) #voeg nog stratify parameter toe
    # real_train_data = pd.concat([X_train.reset_index(drop=True),
    #                                 y_train.reset_index(drop=True)],axis=1)
    # real_train_data.to_csv(data_folder + '/BANK_CUSTOMER_CHURN/BANK_TRAIN_SET.csv',index_label='Index')
    real_train_data, target, numerical_var, categorical_vars = prep_data.main(data_path,args.dataset)
    #real_train_data.columns = df_bank.columns.to_list()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_train_data)
    table_metadata = metadata.to_dict()
    if args.model == 'ctgan':
        output_path = data_folder + '/BANK_CUSTOMER_CHURN/CTGAN'
        synthetic_ctgan = pm.tune_performance_ctgan('bank',real_train_data,table_metadata,None,output_path)
    if args.model == 'tvae':
        output_path = data_folder + '/BANK_CUSTOMER_CHURN/TVAE'
        synthetic_tvae = pm.tune_performance_tvae('bank',real_train_data,table_metadata,None,output_path)
    if args.model == 'arf':
        output_path = data_folder + '/BANK_CUSTOMER_CHURN/ARF/'
        synthetic_arf = pm.tune_performance_arf('bank',real_train_data,str(output_path))
    if args.model == 'cart':
        output_path = data_folder + '/BANK_CUSTOMER_CHURN/CART/'
        synthetic_cart = pm.tune_performance_cart('bank',real_train_data,str(output_path))
    if args.model == 'tabddpm':
        output_path = data_folder + '/BANK_CUSTOMER_CHURN/TABDDPM'
        print('d')

if args.dataset == 'heart':
    data_path = data_folder + '/HEART_ATTACK_PREDICTION/heart.csv'
    cat_columns =  ['sex','thal','target']
    real_train_data, target, numerical_var, categorical_vars = prep_data.main(data_path,args.dataset,cat_columns) #new adjustment of code!
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
    real_train_data, target, numerical_var, categorical_vars = prep_data.main(data_path,args.dataset)
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
        synthetic_arf = pm.tune_performance_arf('adult',real_train_data,str(output_path))
    if args.model == 'cart':
        output_path = data_folder + '/ADULT_CENSUS_INCOME/CART/'
        synthetic_cart = pm.tune_performance_cart('adult',real_train_data,str(output_path))
    if args.model == 'tabddpm':
        output_path = data_folder + '/ADULT_CENSUS_INCOME/TABDDPM'
        print('d')

if args.dataset == 'wine':
    data_path = data_folder + '/WINE_QUALITY/wine.csv' 
    real_train_data, target, numerical_var, categorical_vars = prep_data.main(data_path,args.dataset)
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
        synthetic_arf = pm.tune_performance_arf('wine',real_train_data,str(output_path))
    if args.model == 'cart':
        output_path = data_folder + '/WINE_QUALITY/CART/'
        synthetic_cart = pm.tune_performance_cart('wine',real_train_data,str(output_path))
    if args.model == 'tabddpm':
        output_path = data_folder + '/WINE_QUALITY/TABDDPM'
        print('d')