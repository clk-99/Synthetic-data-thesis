import pandas as pd
import numpy as np
import math
import random
import re
import torch
import sklearn
import argparse
import tabsyndex as ts
import metrics  
import visuals as vs
from pathlib import Path

import synthcity
from synthcity.plugins import Plugins
from synthcity.utils.serialization import load_from_file

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

target_vars = {
    "iris": "class",
    "bank": "y",
    "metro": "traffic_volume",
    "adult": "income",
    "covertype": "Cover_Type"
}
target_type = {
    "iris": "class",
    "bank": "class",
    "metro": "regr",
    "adult": "class",
    "covertype": "class"
}
multi_target_bool = {
    "iris": True,
    "bank": False,
    "metro": True,
    "adult": False,
    "covertype": True
}
cat_columns_dict = {
    "iris": ['class'],
    "bank" : ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y'],
    "metro" : ['holiday','weather_main','weather_description'],
    "adult": ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income'],
    "covertype": ['Soil_type','WArea','Cover_Type']
}

def evaluate_models(real_data,test_data,data_type,data_path,cat_columns,target_var,target_type='class',multi_target=False):
    models = ['arf','cart','ctgan','tvae','tabddpm']
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
                if re.search('.keep|.csv',result.name):
                    #to skip empty.keep and .csv files and break current for loop.
                    print('Incorrect file to proceed in current directory')                    
                    continue
                else:
                    if m == 'arf':
                        synthetic_data = reload_ARF(real_data,cat_columns,result.name)
                        
                    elif m == 'cart':
                        synthetic_data = reload_CART(real_data,cat_columns,result.name)
                        
                    else: 
                        #reload deep generative model
                        os.chdir('..')
                        os.chdir('..')
                        result_path = output_path + '/' + result.name

                        synthetic_data = reload_generative_models(real_data, result_path)
                        os.chdir(output_path)               
                    
                    for c in cat_columns:
                        synthetic_data[c] = synthetic_data[c].astype(str).str.split('.').str[0]
                    
                    real_target_var_uniq = real_data[target_var].unique().tolist()
                    syn_target_var_uniq = synthetic_data[target_var].unique().tolist()
                    if set(real_target_var_uniq) == set(syn_target_var_uniq):
                        print(synthetic_data.shape)
                        scores, missing_unique = ts.tabsyndex(real_data, synthetic_data, cat_cols=cat_columns,target_col=target_var,target_type=target_type) #,multi=multi_target)
                        print(scores)
                        ml_metrics = metrics.MLefficiency(synthetic_data, test_data, cat_columns, target_var, target_type=target_type, multi=multi_target)
                        print(ml_metrics)
                        mean_HD = metrics.hellinger_distance(real_data,synthetic_data)
                        print(mean_HD)
                        mean_KS = metrics.KStest(real_data, synthetic_data, cat_columns, target_var)
                        print(mean_KS)                    

                        results.loc[i,'Saved_model'] = result.name
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
                        for metric in ml_metrics:
                            results.loc[i,metric] = ml_metrics[metric]
                        i+=1
                    else:
                        print('Target variable does not contain all categories.')
                        i+=1
                        continue

        os.chdir('..')
        print(os.getcwd())
        os.chdir('..')
        print(os.getcwd())

    return results

def select_best_model(data_type,results_df):
    print('The best performing model for dataset '+data_type+' equals... \n')
    models_df = results_df.set_index('Saved_model')
    print(models_df)
    best_model = models_df['TabSynDex_score'].idxmax()
    print('Model: '+str(best_model))

    return best_model

def merge_performance_dfs(data_type,data_path,model):
    model_df = pd.DataFrame()
    model_results = data_path + model 
    with os.scandir(model_results) as it:
        output_path = model_results   
        os.chdir(output_path)
     
        for result in it:
            if re.search('.csv',result.name):
                performance_model = pd.read_csv(result.name,delimiter=',')
                temp_df = pd.concat([model_df,performance_model],ignore_index=True,sort=False)
                model_df = temp_df
            else:
                continue #go to next result in the loop
    os.chdir('..')
    os.chdir('..')

    return model_df

def merge_models(data_type,metrics_df,data_path):
    models = ['arf','cart','ctgan','tvae','tabddpm']
    final_df = pd.DataFrame()
    for m in models:
        m_df = merge_performance_dfs(data_type,data_path,m)
        temp_df = pd.concat([final_df,m_df],ignore_index=True)
        final_df = temp_df
    
    if 'Est_train_time' in final_df.columns.to_list(): 
        final_df['Train_time(in seconds)'] = final_df['Train_time(in seconds)'].combine_first(final_df['Est_train_time'])
        print(final_df.head())
        #final_df.drop('Est_train_time',inplace=True)

    result_df = final_df.merge(metrics_df,on='Saved_model',how='left')
    result_df.to_csv('../Data/'+data_type+'/final_results_SDG_'+data_type+'.csv',index_label='Index') 

    return result_df

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

def reload_generative_models(df,result):
    reloaded_model = load_from_file(result)
    synthetic_data = reloaded_model.generate(count=len(df)).dataframe()

    return synthetic_data

def generate_visualize_best_SDG(real_df,cat_columns,sdg,dataset,data_path,output_path):
    model = sdg.split('.')[0].split('_')[0]
    lowercase_model = model.lower()
    print(model)
    model_path = data_path + lowercase_model + '/'
    os.chdir(model_path)

    if lowercase_model == 'cart':
        syn_df = reload_CART(real_df,cat_columns,sdg)

    elif lowercase_model == 'arf':
        syn_df = reload_ARF(real_df,cat_columns,sdg)
    
    else: #DGN models (tvae, ctgan & ddpm)
        syn_df = reload_generative_models(real_df,sdg)
    
    os.chdir('..')
    os.chdir('..')
    os.chdir(output_path) #to save all the visuals  
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
    target_col = target_vars[dataset]
    real_data = pd.read_csv(path_orig_data,index_col=0,dtype={col:'object' for col in cat_vars})
    test_data = pd.read_csv(path_test_data,index_col=0,dtype={col:'object' for col in cat_vars})
    if metric_type == 'metrics':
        performance_df = evaluate_models(real_data,test_data,dataset,data_path,cat_vars,target_col,target_type[dataset],multi_target_bool[dataset])
        print(performance_df.head())
        results_df = merge_models(dataset,performance_df,data_path)
    if metric_type == 'visuals':
        result_path = data_folder + '/' + dataset + '/final_results_SDG_' + dataset + '.csv'
        performance_df = pd.read_csv(result_path,index_col=0)
        print(performance_df.columns)
        best_SDG = select_best_model(dataset, performance_df)
        generate_visualize_best_SDG(real_data,cat_vars,str(best_SDG),dataset,data_path,visual_path)
        vs.create_model_performance_plot(dataset, performance_df, 'Train_time(in seconds)', 'TabSynDex_score') #, visual_path)
        #os.chdir(visual_path)
        vs.create_dgn_performance_plot(dataset, 'CTGAN', performance_df, 'Train_time(in seconds)', 'Nr_epochs')
        vs.create_dgn_performance_plot(dataset, 'CTGAN', performance_df, 'TabSynDex_score', 'Nr_epochs')
        vs.create_dgn_performance_plot(dataset, 'TVAE', performance_df, 'Train_time(in seconds)', 'Nr_epochs')
        vs.create_dgn_performance_plot(dataset, 'TVAE', performance_df, 'TabSynDex_score', 'Nr_epochs')
        vs.create_dgn_performance_plot(dataset, 'TABDDPM', performance_df, 'Train_time(in seconds)', 'Nr_epochs')
        vs.create_dgn_performance_plot(dataset, 'TABDDPM', performance_df, 'TabSynDex_score', 'Nr_epochs')
        vs.create_arf_performance_plot(dataset, performance_df, 'Train_time(in seconds)', 'Nr_trees')
