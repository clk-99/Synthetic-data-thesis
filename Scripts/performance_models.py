import torch
import pandas as pd
import numpy as np

import time
import random

import os
import string

from synthcity import *
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import save_to_file

#from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
#from sdmetrics.single_table import NewRowSynthesis
#os.environ['R_HOME'] = 'V:\KS\Software\R\R-4.2.2' #adjust to version of LISA
#os.environ['R_HOME'] = 'C:\Program Files\R\R-3.5.2' #private laptop R version

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# from rpy2.robjects.packages import importr

# #base = importr('base')
# from rpy2.robjects.vectors import StrVector

# utils = importr('utils')
# package_names = ('arf','synthpop')
# utils.install_packages(StrVector(package_names))

# r = ro.r
# r['source']('CART_function.R')
# r['source']('ARF_function.R')

# cart_function_r = ro.globalenv['run_cart']
# arf_function_r = ro.globalenv['run_arf']

def choose_random(max_limit=100,modulo=10,value=10,size=1):
    y = np.arange(value, max_limit, value)
    return np.random.choice(y, size=size, replace=True)[0]

def generate_unique_id(size=6):
    unique_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(size)])

    return unique_id

# def tune_performance_tvae(data_type,data,metadata,output_path,nr_combinations): 
#     i=0
#     n=nr_combinations

#     while i<n:
#         performance_df = pd.DataFrame()
#         unique_id = generate_unique_id()
#         epochs = random.randrange(5,101,15) #random.randrange(100,501,100)
#         batch_size = random.randrange(100,501,100)
#         l2scale = np.round(np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0],3)
#         nr_layers = np.random.randint(low=1,high=3,size=1)[0]

#         dimensions = random.sample([4,8,16,32,64,128,256],3)
#         compress_dims = dimensions[0]
#         decompress_dims = dimensions[1]
#         embedding_dim = dimensions[2]

#         combination = str(epochs)+'_'+str(batch_size)+'_'+str(compress_dims)+'_'+str(decompress_dims)+'_'+str(embedding_dim)+'_'+str(l2scale)               
#         st = time.process_time()
#         print('The next combination of parameters is tried now: ', combination)
#         if nr_layers == 2:
#             synthesizer = TVAESynthesizer(metadata,epochs=epochs,batch_size=batch_size,compress_dims=(compress_dims,compress_dims),decompress_dims=(decompress_dims,decompress_dims),embedding_dim=embedding_dim,l2scale=l2scale)
#         else:
#             synthesizer = TVAESynthesizer(metadata,epochs=epochs,batch_size=batch_size,compress_dims=(compress_dims,),decompress_dims=(decompress_dims,),embedding_dim=embedding_dim,l2scale=l2scale)
#         #model opslaan
#         synthesizer.fit(data) 
#         filepath = str(output_path) + '/' + 'TVAE_' + str(unique_id) + '.pkl'
#         synthesizer.save(filepath=filepath)
#         print('Synthetic model has been saved.')
#         et = time.process_time()
#         performance_df.loc[i,'Dataset'] = data_type
#         performance_df.loc[i,'Nr_rows'] = len(data)
#         performance_df.loc[i,'Nr_epochs'] = epochs
#         performance_df.loc[i,'Nr_layers'] = nr_layers
#         performance_df.loc[i,'Batch_size'] = batch_size
#         performance_df.loc[i,'Hidden_layer_size_enc'] = compress_dims
#         performance_df.loc[i,'Hidden_layer_size_dec'] = decompress_dims
#         performance_df.loc[i,'Embedding_dimension'] = embedding_dim
#         performance_df.loc[i,'Regularization_term'] = l2scale   
#         performance_df.loc[i,'Saved_model'] = 'TVAE_' + str(unique_id) + '.pkl'     
#         performance_df.loc[i,'Train_time(in seconds)'] = et - st
#         print("Performance dataframe has been filled and saved with current combination.")
#         performance_df.to_csv(str(output_path) + '/performance_tvae_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
#         i += 1

# def tune_performance_ctgan(data_type,data,metadata,output_path,nr_combinations): 
#     i=0
#     n=nr_combinations

#     while i<n:
#         performance_df = pd.DataFrame()
#         unique_id = generate_unique_id()
#         epochs = random.randrange(5,101,15) #random.randrange(100,501,100)
#         batch_size = random.randrange(100,501,100)
#         nr_layers = np.random.randint(low=1,high=3,size=1)[0]
#         dimensions = random.sample([4,8,16,32,64,128,256],3)
#         gen_dim = dimensions[0]
#         dis_dim = dimensions[1]
#         em_dim = dimensions[2]

#         combination = str(epochs)+'_'+str(batch_size)+'_'+str(gen_dim)+'_'+str(dis_dim)+'_'+str(em_dim)                          

#         st = time.process_time()
#         print('The next combination of parameters is tried now: ', combination)
#         if nr_layers == 2:
#             synthesizer = CTGANSynthesizer(metadata,verbose=True,epochs=epochs,batch_size=batch_size,generator_dim=(gen_dim,gen_dim),discriminator_dim=(dis_dim,dis_dim),embedding_dim=em_dim)
#         else:
#             synthesizer = CTGANSynthesizer(metadata,verbose=True,epochs=epochs,batch_size=batch_size,generator_dim=(gen_dim,),discriminator_dim=(dis_dim,),embedding_dim=em_dim)
#         synthesizer.fit(data)         
#         #save model
#         filepath = str(output_path) + '/' + 'CTGAN_'+ str(unique_id) + '.pkl'
#         synthesizer.save(filepath=filepath)
#         print('Synthetic model has been saved.')
#         et = time.process_time()
#         performance_df.loc[i,'Dataset'] = data_type
#         performance_df.loc[i,'Nr_rows'] = len(data)
#         performance_df.loc[i,'Nr_epochs'] = epochs
#         performance_df.loc[i,'Nr_layers'] = nr_layers
#         performance_df.loc[i,'Batch_size'] = batch_size
#         performance_df.loc[i,'Generator_dimension'] = gen_dim
#         performance_df.loc[i,'Discriminator_dimension'] = dis_dim
#         performance_df.loc[i,'Embedding_dimension'] = em_dim
#         performance_df.loc[i,'Saved_model'] = 'CTGAN_'+ str(unique_id) + '.pkl' 
#         performance_df.loc[i,'Train_time(in seconds)'] = et - st
#         print("Performance dataframe has been filled and saved with current combination.")
#         performance_df.to_csv(str(output_path) + '/performance_ctgan_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
#         i += 1

def tune_performance_tvae(data_type,df,target_var,output_path,nr_combinations): 
    i=0

    while i<nr_combinations:
        performance_df = pd.DataFrame()
        unique_id = generate_unique_id()
        X = GenericDataLoader(data=df,target_column=target_var)

        plugin_params = dict(
            n_iter = random.randrange(5,101,15),
            batch_size = random.randrange(100,501,100),
            lr = np.round(np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0],3),
            n_units_embedding = random.sample([4,8,16,32,64,128,256],1)[0],
            decoder_n_units_hidden = random.sample([4,8,16,32,64,128,256],1)[0],
            encoder_n_units_hidden = random.sample([4,8,16,32,64,128,256],1)[0]
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
        performance_df.loc[i,'Hidden_layer_size_enc'] = plugin_params['encoder_n_units_hidden']
        performance_df.loc[i,'Hidden_layer_size_dec'] = plugin_params['decoder_n_units_hidden']
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
            batch_size = random.randrange(100,501,100),
            generator_n_units_hidden = random.sample([4,8,16,32,64,128,256],1)[0],
            discriminator_n_units_hidden =  random.sample([4,8,16,32,64,128,256],1)[0],
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
        performance_df.loc[i,'Generator_dimension'] = plugin_params['generator_n_units_hidden']
        performance_df.loc[i,'Discriminator_dimension'] = plugin_params['discriminator_n_units_hidden']
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
            batch_size = random.randrange(100,501,100),
            num_timesteps = random.randrange(100,1001,100),
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
        performance_df.loc[i,'Saved_model'] = 'TABDDPM_'+ str(unique_id) + '.pkl' 
        performance_df.loc[i,'Train_time(in seconds)'] = et - st
        print("Performance dataframe has been filled and saved with current combination.")
        performance_df.to_csv(str(output_path) + '/performance_ddpm_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
        i += 1


# def tune_performance_arf(data_type,data,output_path,cat_columns,nr_combinations):    
#     i=0
#     n=nr_combinations
#     os.chdir(output_path)

#     nr_trees_set = list()
#     while i<n:
#         unique_id = generate_unique_id()
#         performance_arf = pd.DataFrame()
#         #converting data into r object for passing into r function
#         with (ro.default_converter + pandas2ri.converter).context():
#             df_r = ro.conversion.get_conversion().py2rpy(data)
        
#         #Choose random nr of trees
#         nr_trees = random.randrange(5,101,5)        

#         if nr_trees in nr_trees_set:
#             continue
#         else:
#             nr_trees_set.append(nr_trees)
        
#         #Model type to save model results
#         model_name = 'ARF_' + str(unique_id) + '.rds'

#         st = time.process_time()
#         print('The next nr of trees tried now: ', nr_trees)
#         #Invoking the R function and getting result
#         synthetic_df_r = arf_function_r(df_r,nr_trees,cat_columns,model_name)

#         et = time.process_time()
#         #Convert back to pandas dataframe
#         with (ro.default_converter + pandas2ri.converter).context():
#             synthetic_data = ro.conversion.get_conversion().rpy2py(synthetic_df_r)
        
#         performance_arf.loc[i,'Dataset'] = data_type
#         performance_arf.loc[i,'Nr_rows'] = len(data)
#         performance_arf.loc[i,'Nr_trees'] = nr_trees
#         performance_arf.loc[i,'Saved_model'] = model_name
#         performance_arf.loc[i,'Train_time(in seconds)'] = et - st
#         print("Performance dataframe has been filled and saved with current hyperparameter.")
#         performance_arf.to_csv('performance_arf_'+str(unique_id)+'.csv',encoding='utf-8',index=False)
#         i += 1

 

# def tune_performance_cart(data_type,data,output_path,cat_columns,trials):
#     performance_cart = pd.DataFrame()
#     os.chdir(output_path)
#     i = 0
#     #converting data into r object for passing into r function
#     with localconverter(ro.default_converter + pandas2ri.converter):
#         df_r = ro.conversion.py2rpy(data)
    
#     unique_id = generate_unique_id()
#     #Model type to save model results
#     model_name = 'CART_'+str(unique_id)+'.rds'

#     st = time.process_time()
#     print('The CART model starts with training')
#     #Invoking the R function and getting result
#     synthetic_df_r = cart_function_r(df_r,model_name,cat_columns)

#     et = time.process_time()
#     #Convert back to pandas dataframe
#     with localconverter(ro.default_converter + pandas2ri.converter):
#         synthetic_data = ro.conversion.rpy2py(synthetic_df_r)
    

#     performance_cart.loc[i,'Dataset'] = data_type
#     performance_cart.loc[i,'Nr_rows'] = len(data)
#     performance_cart.loc[i,'Saved_model'] = model_name
#     performance_cart.loc[i,'Train_time(in seconds)'] = et - st
#     print("Performance dataframe has been filled and saved with current hyperparameter.")
#     performance_cart.to_csv('performance_cart_'+str(unique_id)+'.csv',encoding='utf-8',index=False)

#     return synthetic_data 