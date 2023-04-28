import torch
import pandas as pd
import numpy as np
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
import time
import random
from sdmetrics.single_table import NewRowSynthesis
import os

os.environ['R_HOME'] = 'V:\KS\Software\R\R-4.2.2'

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from rpy2.robjects.packages import importr

base = importr('base')

r = ro.r
r['source']('CART_function.R')
r['source']('ARF_function.R')

cart_function_r = ro.globalenv['run_cart']
arf_function_r = ro.globalenv['run_arf']

def choose_random(max_limit=100,modulo=10,value=10,size=1):
    y = np.arange(value, max_limit, value)
    return np.random.choice(y, size=size, replace=True)[0]

def grid_search(param_grid, max_evals = 5):
    """ Grid search algorithm (with limit on max evals) """

    # Dataframe to store results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                            index = list(range(max_evals)))
    
    keys, values = zip(*param_grid.items())

    i = 0

    for v in itertools.product(*values):

        # Create a hyperparameter dictionary
        hyperparameters = dict(zip(keys,v))

        eval_results = objective(hyperparameters, i)

        results.loc[i,:] = eval_results

        i += 1

        if i > max_evals:
            break
    
    results.sort_values('score', ascending=False, inplace=True)
    results.reset_index(inplace=True)

    return results

def tune_performance_tvae(data_type,data,metadata,performance_tvae,output_path): 
    if performance_tvae is None:
        performance_df = pd.DataFrame()
        i=0
        n=1
    else:
        performance_df = performance_tvae 
        n=len(performance_df)+8
        i=len(performance_df)
        #input is a csv file that is saved under 'performance_tvae' which contains dataframe with performance measures    
    while i<n:
        epochs = random.randrange(100,501,100)
        batch_size = random.randrange(100,501,100)
        l2scale = np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0]
        nr_layers = np.random.randint(low=1,high=3,size=1)[0]

        dimensions = random.sample([4,8,16,32,64,128,256],3)
        compress_dims = dimensions[0]
        decompress_dims = dimensions[1]
        embedding_dim = dimensions[2]

        combination = str(epochs)+'_'+str(batch_size)+'_'+str(compress_dims)+'_'+str(decompress_dims)+'_'+str(embedding_dim)+'_'+str(l2scale)               
        if len(performance_df.columns) != 0:
            if combination in performance_df.values:
                #select new set of hyperparameters
                epochs = random.randrange(100,501,100)
                batch_size = random.randrange(100,501,100)
                dimensions = random.sample([4,8,16,64,128,256],3)
                compress_dims = dimensions[0]
                decompress_dims = dimensions[1]
                embedding_dim = dimensions[2]
                l2scale = np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0]
                combination = str(epochs)+'_'+str(batch_size)+'_'+str(compress_dims)+'_'+str(decompress_dims)+'_'+str(embedding_dim)+'_'+str(l2scale) 
                performance_df.loc[i,'Combination_parameters'] = combination   
            else:
                performance_df.loc[i,'Combination_parameters'] = combination

        else:
            performance_df.loc[i,'Combination_parameters'] = combination
        st = time.process_time()
        print('The next combination of parameters is tried now: ', combination)
        if nr_layers == 2:
            synthesizer = TVAESynthesizer(metadata,epochs=epochs,batch_size=batch_size,compress_dims=(compress_dims,compress_dims),decompress_dims=(decompress_dims,decompress_dims),embedding_dim=embedding_dim,l2scale=l2scale)
        else:
            synthesizer = TVAESynthesizer(metadata,epochs=epochs,batch_size=batch_size,compress_dims=(compress_dims,),decompress_dims=(decompress_dims,),embedding_dim=embedding_dim,l2scale=l2scale)
        #model opslaan
        synthesizer.fit(data) 
        filepath = str(output_path) + '/' + 'TVAE_' + combination + '.pkl'
        synthesizer.save(filepath=filepath)
        print('Synthetic model has been saved.')
        et = time.process_time()
        performance_df.loc[i,'Nr_rows'] = len(data)
        performance_df.loc[i,'Nr_epochs'] = epochs
        performance_df.loc[i,'Nr_layers'] = nr_layers
        performance_df.loc[i,'Batch_size'] = batch_size
        performance_df.loc[i,'Hidden_layer_size_enc'] = compress_dims
        performance_df.loc[i,'Hidden_layer_size_dec'] = decompress_dims
        performance_df.loc[i,'Embedding_dimension'] = embedding_dim
        performance_df.loc[i,'Regularization_term'] = l2scale   
        performance_df.loc[i,'Saved_model'] = combination + '.pkl'     
        performance_df.loc[i,'CPU Execution time'] = et - st
        print("Performance dataframe has been filled with current combination.")
        i += 1
    
    if data_type:
        performance_tvae = performance_df.to_csv(str(output_path)+'/'+'performance_tvae_file_'+data_type+'.csv',encoding='utf-8',index=False)
    
    else:
        print('No performance file saved.')

    return performance_tvae

def tune_performance_ctgan(data_type,data,metadata,performance_ctgan,output_path): 
    if performance_ctgan is None:
        performance_df = pd.DataFrame()
        i=0
        n=1
    else:
        performance_df = performance_ctgan 
        n=len(performance_df)+8
        i=len(performance_df)
        #input is a csv file that is saved under 'performance_ctgan' which contains dataframe with performance measures
  
    while i<n:
        epochs = random.randrange(100,501,100)
        batch_size = random.randrange(100,501,100)
        nr_layers = np.random.randint(low=1,high=3,size=1)[0]
        dimensions = random.sample([4,8,16,32,64,128,256],3)
        gen_dim = dimensions[0]
        dis_dim = dimensions[1]
        em_dim = dimensions[2]

        combination = str(epochs)+'_'+str(batch_size)+'_'+str(gen_dim)+'_'+str(dis_dim)+'_'+str(em_dim)                          
        if len(performance_df.columns) != 0:
            if combination in performance_df.values:
                #select new set of hyperparameters
                epochs = random.randrange(100,501,100)
                batch_size = random.randrange(100,501,100)
                nr_layers = np.random.randint(low=1,high=3,size=1)[0]
                dimensions = random.sample([4,8,16,64],3)
                gen_dim = dimensions[0]
                dis_dim = dimensions[1]
                em_dim = dimensions[2]
                combination = str(epochs)+'_'+str(batch_size)+'_'+str(gen_dim)+'_'+str(dis_dim)+'_'+str(em_dim)       
                performance_df.loc[i,'Combination_parameters'] = combination  
            else:
                performance_df.loc[i,'Combination_parameters'] = combination
        else:
            performance_df.loc[i,'Combination_parameters'] = combination
        st = time.process_time()
        print('The next combination of parameters is tried now: ', combination)
        if nr_layers == 2:
            synthesizer = CTGANSynthesizer(metadata,verbose=True,epochs=epochs,batch_size=batch_size,generator_dim=(gen_dim,gen_dim),discriminator_dim=(dis_dim,dis_dim),embedding_dim=em_dim)
        else:
            synthesizer = CTGANSynthesizer(metadata,verbose=True,epochs=epochs,batch_size=batch_size,generator_dim=(gen_dim,),discriminator_dim=(dis_dim,),embedding_dim=em_dim)
        synthesizer.fit(data)         
        #save model
        filepath = str(output_path) + '/' + 'CTGAN_'+ combination + '.pkl'
        synthesizer.save(filepath=filepath)
        print('Synthetic model has been saved.')
        et = time.process_time()
        performance_df.loc[i,'Nr_rows'] = len(data)
        performance_df.loc[i,'Nr_epochs'] = epochs
        performance_df.loc[i,'Nr_layers'] = nr_layers
        performance_df.loc[i,'Batch_size'] = batch_size
        performance_df.loc[i,'Generator_dimension'] = gen_dim
        performance_df.loc[i,'Discriminator_dimension'] = dis_dim
        performance_df.loc[i,'Embedding_dimension'] = em_dim
        performance_df.loc[i,'Saved_model'] = combination + '.pkl' 
        performance_df.loc[i,'CPU Execution time'] = et - st
        print("Performance dataframe has been filled with current combination.")
        i += 1

    if data_type:
        performance_ctgan = performance_df.to_csv(str(output_path)+'/'+'performance_ctgan_file_'+data_type+'.csv',encoding='utf-8',index=False)
    else:
        print('No performance file saved.')
    
    return performance_ctgan

def tune_performance_arf(data_type,data,output_path):
    performance_arf = dict()
    i=0
    n=10

    while i<n:
        #converting data into r object for passing into r function
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(data)
        
        #Choose random nr of trees
        nr_trees = random.randrange(5,101,5)

        if performance_arf:
            if nr_trees in performance_arf:
                nr_trees = random.randrange(5,101,5)
        
        #Model type to save model results
        model_name = 'ARF_' + str(nr_trees) + '.rds'

        #Invoking the R function and getting result
        synthetic_df_r = arf_function_r(df_r,nr_trees,output_path,model_name)

        #Convert back to pandas dataframe
        with localconverter(ro.default_converter + pandas2ri.converter):
            synthetic_data = ro.conversion.rpy2py(synthetic_df_r)

        performance_arf[nr_trees] = synthetic_data
        i += 1

    return  performance_arf

def tune_performance_cart(data_type,data,output_path):

    #converting data into r object for passing into r function
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(data)
    
    #Model type to save model results
    model_name = 'CART.rds'

    #Invoking the R function and getting result
    synthetic_df_r = cart_function_r(df_r,output_path,model_name)

    #Convert back to pandas dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        synthetic_data = ro.conversion.rpy2py(synthetic_df_r)
    

    performance_cart = synthetic_data

    return synthetic_data 