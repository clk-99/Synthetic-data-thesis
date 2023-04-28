import pandas as pd
import numpy as np
import math
import random
import torch
import sklearn
import argparse
import tabsyndex as ts
from pathlib import Path
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

import os

os.environ['R_HOME'] = 'V:\KS\Software\R\R-4.2.2'

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from rpy2.robjects.packages import importr

base = importr('base')
r = ro.r
r['source']('load_ARF.R')
r['source']('load_CART.R')

load_arf_r = ro.globalenv['load_arf']
load_cart_r = ro.globalenv['load_cart']

#start pipeline
parser = argparse.ArgumentParser(description='Evaluate the performance for each model per dataset using TabSynDex')
parser.add_argument("dataset", help='Name of dataset to convert filetype',type=str)
parser.add_argument("metric_type", help="Evaluation metric to compute", type=str, default='tabsyndex')

args = parser.parse_args()
data_folder = Path("Q:/CI_Analisten/TFS_Workspaces/cke035/CIA/Synthetic_data/LISA/Data") #this should be changed to Github path

def evaluate_models(real_data,data_type,data_path,target_type='class'):
    cat_columns = real_data.select_dtypes(exclude=['int','float']).columns.tolist()
    models = ['ARF','CART','CTGAN','TVAE']
    results = pd.DataFrame(columns=['Model_type','Dataset','TabSynDex_score','Basic_score','Correlation_score','Machine_learning_efficiency_score','Support_coverage_score','PMSE_score'])
    for m in models:
        model_results = data_path / m 
        i = 0
        if m == 'ARF':
            with os.scandir(model_results) as it:
                for result in it:
                    print(result.name)
                    output_path = model_results 

                    #converting data into r object for passing into r function
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        df_r = ro.conversion.py2rpy(real_data)

                    synthetic_df_r = load_arf_r(df_r,str(output_path),result.name)

                    with localconverter(ro.default_converter + pandas2ri.converter):
                        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)
                    
                    print(real_data.dtypes)
                    print(synthetic_data.dtypes)
                    scores = ts.tabsyndex(real_data, synthetic_data, cat_cols=cat_columns,target_type=target_type)
                    results.loc[i,'Model_type'] = result.name
                    results.loc[i,'Dataset'] = data_type
                    results.loc[i,'TabSynDex_score'] = scores['score']
                    results.loc[i,'Basic_score'] = scores['basic_score']
                    results.loc[i,'Correlation_score'] = scores['corr_score']
                    results.loc[i,'Machine_learning_efficiency_score'] = scores['ml_score']
                    results.loc[i,'Support_coverage_score'] = scores['sup_score']
                    results.loc[i,'PMSE_score'] = scores['pmse_score']

                    i+=1

        elif m == 'CART':
            with os.scandir(model_results) as it:
                for result in it:
                    print(result.name)

                    with localconverter(ro.default_converter + pandas2ri.converter):
                        df_r = ro.conversion.py2rpy(real_data)

                    synthetic_df_r = load_cart_r(df_r,str(output_path),str(result))
                    
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)
                    
                    scores = ts.tabsyndex(real_data, synthetic_data, cat_cols=cat_columns,target_type=target_type)
                    results.loc[i,'Model_type'] = result
                    results.loc[i,'Dataset'] = data_type
                    results.loc[i,'TabSynDex_score'] = scores['score']
                    results.loc[i,'Basic_score'] = scores['basic_score']
                    results.loc[i,'Correlation_score'] = scores['corr_score']
                    results.loc[i,'Machine_learning_efficiency_score'] = scores['ml_score']
                    results.loc[i,'Support_coverage_score'] = scores['sup_score']
                    results.loc[i,'PMSE_score'] = scores['pmse_score']

                    i+=1

        elif m == 'CTGAN': 
            with os.scandir(model_results) as it:
                for result in it:
                    print(result)
                    #reload ctgan model
                    ctgan_model = CTGANSynthesizer.load(filepath=result)
                    synthetic_data = ctgan_model.sample(num_rows=len(real_data))
                    scores = ts.tabsyndex(real_data, synthetic_data, cat_cols=cat_columns,target_type=target_type)
                    results.loc[i,'Model_type'] = result
                    results.loc[i,'Dataset'] = data_type
                    results.loc[i,'TabSynDex_score'] = scores['score']
                    results.loc[i,'Basic_score'] = scores['basic_score']
                    results.loc[i,'Correlation_score'] = scores['corr_score']
                    results.loc[i,'Machine_learning_efficiency_score'] = scores['ml_score']
                    results.loc[i,'Support_coverage_score'] = scores['sup_score']
                    results.loc[i,'PMSE_score'] = scores['pmse_score']
                    i+=1
        
        elif m == 'TVAE': 
            with os.scandir(model_results) as it:
                for result in it:
                    print(result)
                    #reload tvae model
                    tvae_model = TVAESynthesizer.load(filepath=result)
                    synthetic_data = tvae_model.sample(num_rows=len(real_data))
                    scores = ts.tabsyndex(real_data, synthetic_data, cat_cols=cat_columns,target_type=target_type)
                    results.loc[i,'Model_type'] = result
                    results.loc[i,'Dataset'] = data_type
                    results.loc[i,'TabSynDex_score'] = scores['score']
                    results.loc[i,'Basic_score'] = scores['basic_score']
                    results.loc[i,'Correlation_score'] = scores['corr_score']
                    results.loc[i,'Machine_learning_efficiency_score'] = scores['ml_score']
                    results.loc[i,'Support_coverage_score'] = scores['sup_score']
                    results.loc[i,'PMSE_score'] = scores['pmse_score']
                    i+=1
    return results

def select_best_model(data_type,results_df):
    print('The best performing model for dataset '+data_type+'equals... \n')
    models_df = results_df.set_index('Model_type')
    best_model = models_df['TabSynDex_score'].idxmax()
    print('Model: '+str(best_model))


if args.dataset == 'bank':
    path_orig_data = data_folder / 'BANK_CUSTOMER_CHURN/BANK_TRAIN_SET.csv'
    if args.metric_type == 'tabsyndex':
        real_data = pd.read_csv(path_orig_data,index_col=0)
        performance_df = evaluate_models(real_data,'bank',data_folder)
        select_best_model('bank', performance_df)
    if args.metric_type == 'statistical':
        print('c')

if args.dataset == 'heart':
    path_orig_data = data_folder / 'HEART_ATTACK_PREDICTION/HEART_TRAIN_SET.csv'
    data_path = data_folder / 'HEART_ATTACK_PREDICTION'
    if args.metric_type == 'tabsyndex':
        real_data = pd.read_csv(path_orig_data,index_col=0)
        performance_df = evaluate_models(real_data,'heart',data_path)
        select_best_model('heart', performance_df)
    if args.metric_type == 'statistical':
        print('c')
        #use statistical tests to compute scores
        #from here other metrics can be computed


if args.dataset == 'adult':
    path_orig_data = data_folder / 'ADULT_CENSUS_INCOME/ADULT_TRAIN_SET.csv'
    if args.metric_type == 'tabsyndex':
        real_data = pd.read_csv(path_orig_data,index_col=0)
        performance_df = evaluate_models(real_data,'heart',data_folder,'regr')
        select_best_model('heart', performance_df)
    if args.metric_type == 'statistical':
        print('c')

if args.dataset == 'wine':
    path_orig_data = data_folder / 'WINE_QUALITY/WINE_TRAIN_SET.csv'
    if args.metric_type == 'tabsyndex':
        real_data = pd.read_csv(path_orig_data,index_col=0)
        performance_df = evaluate_models('wine',real_data,data_folder)
        select_best_model('wine', performance_df)
    if args.metric_type == 'statistical':
        print('c')

