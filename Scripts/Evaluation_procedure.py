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

#os.environ['R_HOME'] = 'V:\KS\Software\R\R-4.2.2' #adjust to the version on LISA!!

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
metric_type = args.metric_type
target_type = {
    "bank": "class",
    "metro": "regr",
    "adult": "class",
    "covertype": "class"
}
multi_target_bool = {
    "bank": False,
    "metro": True,
    "adult": False,
    "covertype": True
}
cat_columns_dict = {
    "bank" : ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y'],
    "metro" : ['holiday','weather_main','weather_description'],
    "adult": ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income'],
    "covertype": ['Soil_type','WArea','Cover_Type']
}

def evaluate_models(real_data,test_data,data_type,data_path,cat_columns,target_type='class',multi_target=False):
    models = ['arf','cart','ctgan','tvae']
    results = pd.DataFrame()
    i = 0
    for m in models:
        model_results = data_path + m  
        with os.scandir(model_results) as it:
            output_path = model_results
            os.chdir(output_path)
            print(output_path)
            for result in it:
                print(result.name)  
                if result.name == 'empty.keep':
                    #to skip empty.keep files and break current for loop.
                    print('Incorrect file to proceed in current directory')
                    os.chdir('..')
                    print(os.getcwd())
                    os.chdir('..')
                    print(os.getcwd())
                    break
                else:
                    if m == 'arf':
                        synthetic_data = reload_ARF(real_data,cat_columns,result.name)
                        
                    elif m == 'cart':
                        synthetic_data = reload_CART(real_data,cat_columns,result.name)
                        
                    elif m == 'ctgan': 
                        #reload ctgan model
                        synthetic_data = reload_CTGAN(real_data,result.name)

                    else: #elif m == 'tvae': #tvae model
                        synthetic_data = reload_TVAE(real_data,result.name)
                    
                    for c in cat_columns:
                        synthetic_data[c] = synthetic_data[c].astype(str).str.split('.').str[0]
                    print(synthetic_data.shape)
                    scores, missing_unique = ts.tabsyndex(real_data, synthetic_data, cat_cols=cat_columns,target_col=real_data.columns.to_list()[-1],target_type=target_type)
                    print(scores)
                    ml_metrics = metrics.MLefficiency(synthetic_data, test_data, cat_columns, target_type=target_type, multi=multi_target)
                    print(ml_metrics)
                    mean_HD = metrics.hellinger_distance(real_data,synthetic_data)
                    print(mean_HD)
                    mean_KS = metrics.KStest(real_data, synthetic_data, cat_columns)
                    print(mean_KS)
                    mean_ES = metrics.EStest(real_data, synthetic_data, cat_columns)
                    print(mean_ES)

                    results.loc[i,'Model_type'] = result.name
                    results.loc[i,'Dataset'] = data_type
                    results.loc[i,'TabSynDex_score'] = scores['score']
                    results.loc[i,'Basic_score'] = scores['basic_score']
                    results.loc[i,'Correlation_score'] = scores['corr_score']
                    results.loc[i,'Machine_learning_efficiency_score'] = scores['ml_score']
                    results.loc[i,'Support_coverage_score'] = scores['sup_score']
                    results.loc[i,'PMSE_score'] = scores['pmse_score']
                    results.loc[i,'Missing_unique_values'] = missing_unique
                    results.loc[i,'mean_Hellinger_Distance'] = mean_HD
                    results.loc[i,'mean_Kolmogorov_Smirnov_test'] = mean_KS
                    results.loc[i,'mean_Epps_Singleton_test'] = mean_ES
                    for metric in ml_metrics:
                        results.loc[i,metric] = ml_metrics[metric]
                    i+=1

    results.to_csv('../Data/'+data_type+'/metrics_SDG_'+data_type+'.csv',index_label='Index')          
    return results

def select_best_model(data_type,results_df):
    print('The best performing model for dataset '+data_type+' equals... \n')
    models_df = results_df.set_index('Model_type')
    print(models_df)
    best_model = models_df['TabSynDex_score'].idxmax()
    print('Model: '+str(best_model))

    return best_model

def reload_ARF(df,cat_vars,result):
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df)

    synthetic_df_r = load_arf_r(df_r,cat_vars,str(result))

    with localconverter(ro.default_converter + pandas2ri.converter):
        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)

    return synthetic_data

def reload_CART(df,cat_vars,result):
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df)

    synthetic_df_r = load_cart_r(df_r,cat_vars,str(result))
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)

    return synthetic_data

def reload_CTGAN(df,result):
    ctgan_model = CTGANSynthesizer.load(filepath=result)
    synthetic_data = ctgan_model.sample(num_rows=len(df))

    return synthetic_data

def reload_TVAE(df,result):
    tvae_model = TVAESynthesizer.load(filepath=result)
    synthetic_data = tvae_model.sample(num_rows=len(df))

    return synthetic_data

def generate_visualize_best_SDG(real_df,cat_columns,sdg,dataset,data_path,output_path):
    model = sdg.split('.')[0].split('_')[0]
    print(model)
    if model == 'CART':
        model_path = data_path + 'cart/'
        os.chdir(model_path)
        syn_df = reload_CART(real_df,cat_columns,sdg)
    elif model == 'ARF':
        model_path = data_path + 'arf/'
        os.chdir(model_path)
        syn_df = reload_ARF(real_df,cat_columns,sdg)
    elif model == 'CTGAN':
        model_path = data_path + 'ctgan/'
        os.chdir(model_path)
        syn_df = reload_CTGAN(real_df,sdg)
    else: #tvae model
        model_path = data_path + 'tvae/'
        os.chdir(model_path)
        syn_df = reload_TVAE(real_df,sdg)
    
    os.chdir('..')
    os.chdir('..')
    os.chdir(output_path) #to save all the visuals  
    print(real_df.shape)
    print(syn_df.shape) 
    vs.create_heatmaps(real_df,dataset,True)
    vs.create_heatmaps(syn_df,dataset,False)
    vs.create_boxplots(real_df,syn_df,dataset)
    vs.create_violinplots(real_df,syn_df,dataset)
    vs.create_kdeplots(real_df,syn_df,dataset)
    vs.create_ecdfplots(real_df,syn_df,dataset)  
    

if dataset:
    path_orig_data = data_folder + '/' + dataset + '/' + dataset +'_TRAIN_SET.csv'
    path_test_data = data_folder + '/' + dataset + '/' + dataset +'_TEST_SET.csv'
    data_path = data_folder + '/' + dataset + '/'
    visual_path = data_folder + '/' + dataset + '/visuals/'
    cat_vars = cat_columns_dict[dataset]
    real_data = pd.read_csv(path_orig_data,index_col=0,dtype={col:'object' for col in cat_vars})
    test_data = pd.read_csv(path_test_data,index_col=0,dtype={col:'object' for col in cat_vars})
    if metric_type == 'tabsyndex':
        performance_df = evaluate_models(real_data,test_data,dataset,data_path,cat_vars,target_type[dataset],multi_target_bool[dataset])
    if metric_type == 'visuals':
        result_path = data_folder + '/' + dataset + '/metrics_SDG_' + dataset + '.csv'
        performance_df = pd.read_csv(result_path,index_col=0)
        best_SDG = select_best_model(dataset, performance_df)
        generate_visualize_best_SDG(real_data,cat_vars,str(best_SDG),dataset,data_path,visual_path)
