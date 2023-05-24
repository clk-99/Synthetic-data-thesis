import pandas as pd
import numpy as np
import math
import random
import torch
import sklearn
import argparse
import tabsyndex as ts
import metrics  
import visuals as vs
from pathlib import Path
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

import os

os.environ['R_HOME'] = 'V:\KS\Software\R\R-4.2.2' #adjust to the version on LISA!!

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
data_folder = '../Data' 
dataset = args.dataset
cat_columns_dict = {
    "bank" : ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y'],
    "metro" : ['holiday','weather_main','weather_description'],
    "adult": ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income'],
    "covertype": ['Soil_type','WArea','Cover_Type']
}

def evaluate_models(real_data,data_type,data_path,cat_columns,target_type='class'):
    models = ['CART','ARF','CTGAN','TVAE']
    results = pd.DataFrame(columns=['Model_type','Dataset','TabSynDex_score','Basic_score','Correlation_score','Machine_learning_efficiency_score','Support_coverage_score','PMSE_score'])
    for m in models:
        model_results = data_path + m 
        i = 0
        with os.scandir(model_results) as it:
            for result in it:
                print(result.name)
                output_path = model_results
                os.chdir(output_path)
                print(output_path)
                if m == 'ARF':
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        df_r = ro.conversion.py2rpy(real_data)

                    synthetic_df_r = load_arf_r(df_r,str(output_path),cat_columns,str(result.name))

                    with localconverter(ro.default_converter + pandas2ri.converter):
                        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)

                elif m == 'CART':

                    with localconverter(ro.default_converter + pandas2ri.converter):
                        df_r = ro.conversion.py2rpy(real_data)

                    synthetic_df_r = load_cart_r(df_r,str(output_path),cat_columns,str(result.name))
                    
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)
                    
                elif m == 'CTGAN': 
                    #reload ctgan model
                    ctgan_model = CTGANSynthesizer.load(filepath=result.name)
                    synthetic_data = ctgan_model.sample(num_rows=len(real_data))
    
                elif m == 'TVAE': 
                    tvae_model = TVAESynthesizer.load(filepath=result.name)
                    synthetic_data = tvae_model.sample(num_rows=len(real_data))
                
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
                
    return results

def select_best_model(data_type,results_df):
    print('The best performing model for dataset '+data_type+' equals... \n')
    models_df = results_df.set_index('Model_type')
    print(models_df)
    best_model = models_df['TabSynDex_score'].idxmax()
    print('Model: '+str(best_model))


if dataset:
    path_orig_data = data_folder + '/' + dataset + '/' + dataset +'_TRAIN_SET.csv'
    data_path = data_folder + '/' + dataset
    cat_vars = cat_columns_dict[dataset]
    if args.metric_type == 'tabsyndex':
        real_data = pd.read_csv(path_orig_data,index_col=0)
        if dataset == 'metro':
            performance_df = evaluate_models(real_data,dataset,data_path,cat_vars,'regr')
        else:
            performance_df = evaluate_models(real_data,dataset,data_path,cat_vars,'class')
        select_best_model(dataset, performance_df)
    if args.metric_type == 'statistical':
        print('c')
    if args.metric_type == 'visuals':
        print('e')
    if args.metric_type == 'ml':
        print('e')

# if args.dataset == 'metro':
#     path_orig_data = data_folder + '/metro/metro_TRAIN_SET.csv'
#     data_path = data_folder + '/metro/'
#     cat_vars =  ['holiday','weather_main','weather_description'] 
#     if args.metric_type == 'tabsyndex':
#         real_data = pd.read_csv(path_orig_data,index_col=0)
#         performance_df = evaluate_models(real_data,'metro',data_path,cat_vars,'regr')
#         select_best_model('metro', performance_df)
#     if args.metric_type == 'statistical':
#         print('c')
#     if args.metric_type == 'outlier':
#         print('d')
#         #use statistical tests to compute scores
#         #from here other metrics can be computed


# if args.dataset == 'adult':
#     path_orig_data = data_folder + '/adult/adult_TRAIN_SET.csv'
#     data_path = data_folder + '/adult'
#     cat_vars = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','class'] 
#     if args.metric_type == 'tabsyndex':
#         real_data = pd.read_csv(path_orig_data,index_col=0)
#         performance_df = evaluate_models(real_data,'adult',data_path,cat_vars,'regr')
#         select_best_model('heart', performance_df)
#     if args.metric_type == 'statistical':
#         print('c')
#     if args.metric_type == 'outlier':
#         print('d')

