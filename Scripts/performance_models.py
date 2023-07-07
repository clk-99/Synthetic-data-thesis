import torch
import pandas as pd
import numpy as np

import time
import random

import os
import string

import synthcity
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import save_to_file

#os.environ['R_HOME'] = 'V:\KS\Software\R\R-4.2.2' #adjust to version of LISA

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

r = ro.r
r['source']('CART_function.R')
r['source']('ARF_function.R')

cart_function_r = ro.globalenv['run_cart']
arf_function_r = ro.globalenv['run_arf']

def choose_random(max_limit=100,modulo=10,value=10,size=1):
    y = np.arange(value, max_limit, value)
    return np.random.choice(y, size=size, replace=True)[0]

def generate_unique_id(size=6):
    unique_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(size)])

    return unique_id

def tune_performance_tvae(data_type,df,target_var,output_path,nr_combinations): 
    i=0

    while i<nr_combinations:
        performance_df = pd.DataFrame()
        unique_id = generate_unique_id()
        X = GenericDataLoader(data=df,target_column=target_var)
     
        plugin_params = dict(
            n_iter = random.randrange(5,101,15),
            batch_size = random.sample([2**i for i in range(4,17) if 2**i<df.shape[0]],1)[0],
            lr = np.round(np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0],3),
            n_units_embedding = random.sample([4,8,16,32,64,128,256],1)[0],
            decoder_n_units_hidden = random.sample([4,8,16,32,64,128,256],1)[0],
            decoder_n_layers_hidden = random.randrange(1,5,1),
            encoder_n_units_hidden = random.sample([4,8,16,32,64,128,256],1)[0],
            encoder_n_layers_hidden = random.randrange(1,5,1)
        )

        st = time.process_time()
        print('The next combination of parameters is tried now: ', plugin_params)
        
        #model fit
        tvae_plugin = Plugins().get('tvae',**plugin_params)
        tvae_plugin.fit(X)

        filepath = str(output_path) + '/' + 'TVAE_' + str(unique_id) + '.pkl'

        #model opslaan
        save_to_file(filepath,tvae_plugin)
        print('Synthetic model has been saved.')
        et = time.process_time()
        performance_df.loc[i,'Dataset'] = data_type
        performance_df.loc[i,'Nr_rows'] = len(df)
        performance_df.loc[i,'Nr_epochs'] = plugin_params['n_iter']
        performance_df.loc[i,'Batch_size'] = plugin_params['batch_size']
        performance_df.loc[i,'Hidden_units_size_enc'] = plugin_params['encoder_n_units_hidden']
        performance_df.loc[i,'Hidden_units_size_dec'] = plugin_params['decoder_n_units_hidden']
        performance_df.loc[i,'Hidden_layer_size_enc'] = plugin_params['encoder_n_layers_hidden']
        performance_df.loc[i,'Hidden_layer_size_dec'] = plugin_params['decoder_n_layers_hidden']
        performance_df.loc[i,'Embedding_dimension'] = plugin_params['n_units_embedding']
        performance_df.loc[i,'Learning_rate'] = plugin_params['lr']   
        performance_df.loc[i,'Saved_model'] = 'TVAE_' + str(unique_id) + '.pkl'     
        performance_df.loc[i,'Train_time(in seconds)'] = et - st
        print("Performance dataframe has been filled and saved with current combination.")
        performance_df.to_csv(str(output_path) + '/performance_tvae_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
        i += 1

def tune_performance_ctgan(data_type,df,target_var,output_path,nr_combinations): 
    i=0

    while i<nr_combinations:
        performance_df = pd.DataFrame()
        unique_id = generate_unique_id()
        X = GenericDataLoader(data=df,target_column=target_var)
        plugin_params = dict(
            n_iter = random.randrange(5,101,15), #random.randrange(100,501,100)
            batch_size = random.sample([2**i for i in range(4,17) if 2**i<df.shape[0]],1)[0],
            generator_n_units_hidden = random.sample([4,8,16,32,64,128,256],1)[0],
            generator_n_layers_hidden = random.randrange(1,5,1),
            discriminator_n_units_hidden =  random.sample([4,8,16,32,64,128,256],1)[0],
            discriminator_n_layers_hidden = random.randrange(1,5,1),
            lr = np.round(np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0],3)
        )
   
        st = time.process_time()
        print('The next combination of parameters is tried now: ', plugin_params)
        #model fit
        ctgan_plugin = Plugins().get('ctgan',**plugin_params)
        ctgan_plugin.fit(X)
        #save model
        filepath = str(output_path) + '/' + 'CTGAN_'+ str(unique_id) + '.pkl'
        save_to_file(filepath,ctgan_plugin)

        print('Synthetic model has been saved.')
        et = time.process_time()
        performance_df.loc[i,'Dataset'] = data_type
        performance_df.loc[i,'Nr_rows'] = len(df)
        performance_df.loc[i,'Nr_epochs'] = plugin_params['n_iter']
        performance_df.loc[i,'Batch_size'] = plugin_params['batch_size']
        performance_df.loc[i,'Hidden_unit_size_gen'] = plugin_params['generator_n_units_hidden']
        performance_df.loc[i,'Hidden_unit_size_dis'] = plugin_params['discriminator_n_units_hidden']
        performance_df.loc[i,'Hidden_layer_size_gen'] = plugin_params['generator_n_layers_hidden']
        performance_df.loc[i,'Hidden_layer_size_dis'] = plugin_params['discriminator_n_layers_hidden']
        performance_df.loc[i,'Learning_rate'] = plugin_params['lr']
        performance_df.loc[i,'Saved_model'] = 'CTGAN_'+ str(unique_id) + '.pkl' 
        performance_df.loc[i,'Train_time(in seconds)'] = et - st
        print("Performance dataframe has been filled and saved with current combination.")
        performance_df.to_csv(str(output_path) + '/performance_ctgan_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
        i += 1

def tune_performance_ddpm(data_type,df,target_var,output_path,nr_combinations): 
    i=0

    while i<nr_combinations:
        performance_df = pd.DataFrame()
        unique_id = generate_unique_id()
        X = GenericDataLoader(data=df,target_column=target_var)

        plugin_params = dict(
            n_iter = random.randrange(5,101,15), #random.randrange(100,501,100)
            batch_size = random.sample([2**i for i in range(4,17) if 2**i<df.shape[0]],1)[0],
            num_timesteps = random.randrange(100,1001,100),
            dim_embed = random.sample([4,8,16,32,64,128,256],1)[0],
            model_params = dict(n_layers_hidden=random.randrange(1,5,1), n_units_hidden=random.sample([4,8,16,32,64,128,256],1)[0], dropout=np.round(np.random.choice(np.linspace(start=0.0,stop=0.5,num=10),size=1)[0],1)),
            lr = np.round(np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0],3)
        )    
        st = time.process_time()
        print('The next combination of parameters is tried now: ', plugin_params)
        #model fit
        ddpm_plugin = Plugins().get('ddpm',**plugin_params)
        ddpm_plugin.fit(X)
        #save model
        filepath = str(output_path) + '/' + 'TABDDPM_'+ str(unique_id) + '.pkl'
        save_to_file(filepath,ddpm_plugin)
        
        print('Synthetic model has been saved.')
        et = time.process_time()
        performance_df.loc[i,'Dataset'] = data_type
        performance_df.loc[i,'Nr_rows'] = len(df)
        performance_df.loc[i,'Nr_epochs'] = plugin_params['n_iter']
        performance_df.loc[i,'Batch_size'] = plugin_params['batch_size']
        performance_df.loc[i,'Num_timesteps'] = plugin_params['num_timesteps']
        performance_df.loc[i,'Learning_rate'] = plugin_params['lr']
        performance_df.loc[i,'Embedding_dimension'] = plugin_params['dim_embed']
        performance_df.loc[i,'Nr_hidden_layers'] = plugin_params['model_params']['n_layers_hidden']
        performance_df.loc[i,'Nr_hidden_units'] = plugin_params['model_params']['n_units_hidden']
        performance_df.loc[i,'Dropout_rate'] = plugin_params['model_params']['dropout']
        performance_df.loc[i,'Saved_model'] = 'TABDDPM_'+ str(unique_id) + '.pkl' 
        performance_df.loc[i,'Train_time(in seconds)'] = et - st
        print("Performance dataframe has been filled and saved with current combination.")
        performance_df.to_csv(str(output_path) + '/performance_ddpm_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
        i += 1


def tune_performance_arf(data_type,data,output_path,cat_columns,nr_combinations):    
    i=0
    n=nr_combinations
    os.chdir(output_path)

    nr_trees_set = list()
    while i<n:
        unique_id = generate_unique_id()
        performance_arf = pd.DataFrame()
        #converting data into r object for passing into r function
        with (ro.default_converter + pandas2ri.converter).context():
            df_r = ro.conversion.get_conversion().py2rpy(data)
        
        #Choose random nr of trees
        nr_trees = random.randrange(5,101,5)        

        if nr_trees in nr_trees_set:
            continue
        else:
            nr_trees_set.append(nr_trees)
        
        #Model type to save model results
        model_name = 'ARF_' + str(unique_id) + '.rds'

        st = time.process_time()
        print('The next nr of trees tried now: ', nr_trees)
        #Invoking the R function and getting result
        synthetic_df_r = arf_function_r(df_r,nr_trees,cat_columns,model_name)

        et = time.process_time()
        #Convert back to pandas dataframe
        with (ro.default_converter + pandas2ri.converter).context():
            synthetic_data = ro.conversion.get_conversion().rpy2py(synthetic_df_r)
        
        performance_arf.loc[i,'Dataset'] = data_type
        performance_arf.loc[i,'Nr_rows'] = len(data)
        performance_arf.loc[i,'Nr_trees'] = nr_trees
        performance_arf.loc[i,'Saved_model'] = model_name
        performance_arf.loc[i,'Train_time(in seconds)'] = et - st
        print("Performance dataframe has been filled and saved with current hyperparameter.")
        performance_arf.to_csv('performance_arf_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
        i += 1

 

def tune_performance_cart(data_type,data,output_path,cat_columns,trials):
    performance_cart = pd.DataFrame()
    os.chdir(output_path)
    i = 0
    #converting data into r object for passing into r function
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(data)
    
    unique_id = generate_unique_id()
    #Model type to save model results
    model_name = 'CART_'+str(unique_id)+'.rds'

    st = time.process_time()
    print('The CART model starts with training')
    #Invoking the R function and getting result
    synthetic_df_r = cart_function_r(df_r,model_name,cat_columns)

    et = time.process_time()
    #Convert back to pandas dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)
    

    performance_cart.loc[i,'Dataset'] = data_type
    performance_cart.loc[i,'Nr_rows'] = len(data)
    performance_cart.loc[i,'Saved_model'] = model_name
    performance_cart.loc[i,'Train_time(in seconds)'] = et - st
    print("Performance dataframe has been filled and saved with current hyperparameter.")
    performance_cart.to_csv('performance_cart_'+str(unique_id)+'.csv',encoding='utf-8',index=False)

    return synthetic_data 
