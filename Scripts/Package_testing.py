import torch
import pandas as pd
import numpy as np
import SQLconnect as sql
from sdv.tabular import TVAE, CTGAN, CopulaGAN
import datgan
from sdmetrics.reports.single_table import QualityReport
import time
import random
from sdmetrics.single_table import NewRowSynthesis


def choose_random(max_limit=100,modulo=10,value=10,size=1):
    y = np.arange(value, max_limit, value)
    return np.random.choice(y, size=size, replace=True)[0]

def tune_performance_DPGAN(performance_dpgan,data):
    if performance_dpgan is None:
        performance_df = pd.DataFrame()
        i=0
        n=10
    else:
        performance_df = performance_dpgan 
        n=len(performance_df)+10
        i=len(performance_df)

    return performance_df

def tune_performance_CTABGAN(performance_ctabgan,data):
    if performance_ctabgan is None:
        performance_df = pd.DataFrame()
        i=0
        n=10
    else:
        performance_df = performance_ctabgan 
        n=len(performance_df)+10
        i=len(performance_df)

    return performance_df

#voeg nog loss toe door model aan te passen -- gaat niet lukken want is een package
#checken of combinatie van parameters is geweest
#doe dit door te zoeken of dataframe al bestaat, anders maak je nieuwe aan.
#voor elke nieuwe combinatie, voeg deze toe aan dataframe.
def tune_performance_tvae(data_type,performance_tvae,metadata,data): 
    if performance_tvae is None:
        performance_df = pd.DataFrame()
        i=0
        n=8
    else:
        performance_df = performance_tvae 
        n=len(performance_df)+8
        i=len(performance_df)
        #input is a csv file that is saved under 'performance_tvae' which contains dataframe with performance measures
    
    while i<n:
        epochs = np.random.randint(low=1,high=6,size=1)[0]
        batch_size = random.randrange(100,501,100)
        input_var = choose_random()
        compress_dims = input_var
        decompress_dims = input_var
        embedding_dim = input_var
        l2scale = np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0]

        combination = str(epochs)+'_'+str(batch_size)+'_'+str(compress_dims)+'_'+str(decompress_dims)+'_'+str(embedding_dim)+'_'+str(l2scale)               
        if len(performance_df.columns) != 0:
            if combination in performance_df.values:
                #select new set of hyperparameters
                epochs = np.random.randint(low=1,high=6,size=1)[0]
                batch_size = random.randrange(100,501,100)
                input_var = choose_random()
                compress_dims = input_var
                decompress_dims = input_var
                embedding_dim = input_var
                l2scale = np.random.choice(np.linspace(start=0.01,stop=0.05,num=10000),size=1)[0]
                combination = str(epochs)+'_'+str(batch_size)+'_'+str(compress_dims)+'_'+str(decompress_dims)+'_'+str(embedding_dim)+'_'+str(l2scale) 
                performance_df.loc[i,'Combination_parameters'] = combination   
            else:
                performance_df.loc[i,'Combination_parameters'] = combination

        else:
            performance_df.loc[i,'Combination_parameters'] = combination
        tvae_report = QualityReport()
        st = time.process_time()
        print('The next combination of parameters is tried now: ', combination)
        model = TVAE(table_metadata=metadata,epochs=epochs,batch_size=batch_size,compress_dims=(compress_dims,compress_dims),decompress_dims=(decompress_dims,decompress_dims),embedding_dim=embedding_dim,l2scale=l2scale)
        #model opslaan
        model.fit(data) 
        synthetic_data = model.sample(num_rows=len(data))
        print('Synthetic data has been generated.')
        et = time.process_time()
        tvae_report.generate(data,synthetic_data,metadata)
        properties = tvae_report.get_properties()
        performance_df.loc[i,'Nr_rows'] = len(data)
        performance_df.loc[i,'Nr_epochs'] = epochs
        performance_df.loc[i,'Batch_size'] = batch_size
        performance_df.loc[i,'Hidden_layer_size_enc'] = compress_dims
        performance_df.loc[i,'Hidden_layer_size_dec'] = decompress_dims
        performance_df.loc[i,'Embedding_dimension'] = embedding_dim
        performance_df.loc[i,'Regularization_term'] = l2scale
        performance_df.loc[i,'Overall_score'] = tvae_report.get_score()
        performance_df.loc[i,'Column_shapes'] = properties.loc[properties['Property']=='Column Shapes','Score'].values[0]
        performance_df.loc[i,'Column_pair_trends'] = properties.loc[properties['Property']=='Column Pair Trends','Score'].values[0]
        performance_df.loc[i,'Row_synthesis'] = NewRowSynthesis.compute(
                                                    real_data = data,
                                                    synthetic_data = synthetic_data,
                                                    metadata = metadata,
                                                    numerical_match_tolerance=0.01,
                                                    synthetic_sample_size = 1000)
        performance_df.loc[i,'CPU Execution time'] = et - st
        print("Performance dataframe has been filled with current combination.")
        i += 1
    
    if data_type=='wwb':
        performance_tvae = performance_df.to_csv('performance_tvae_file_wwb.csv',encoding='utf-8',index=False)
    else:
        performance_tvae = performance_df.to_csv('performance_tvae_file_polis.csv',encoding='utf-8',index=False)

    return performance_tvae


def tune_performance_ctgan(data_type,performance_ctgan,metadata,data): 
    #werkt niet op de package SDV, maar ctgan package
    if performance_ctgan is None:
        performance_df = pd.DataFrame()
        i=0
        n=8
    else:
        performance_df = performance_ctgan 
        n=len(performance_df)+8
        i=len(performance_df)
        #input is a csv file that is saved under 'performance_ctgan' which contains dataframe with performance measures
  
    while i<n:
        epochs = np.random.randint(low=1,high=6,size=1)[0]
        batch_size = random.randrange(100,501,100)
        input_var = choose_random()
        gen_dim = input_var
        dis_dim = input_var
        em_dim = input_var

        combination = str(epochs)+'_'+str(batch_size)+'_'+str(gen_dim)+'_'+str(dis_dim)+'_'+str(em_dim)                          
        if len(performance_df.columns) != 0:
            if combination in performance_df.values:
                #select new set of hyperparameters
                epochs = np.random.randint(low=1,high=6,size=1)[0]
                batch_size = random.randrange(100,501,100)
                input_var = choose_random()
                gen_dim = input_var
                dis_dim = input_var
                em_dim = input_var
                combination = str(epochs)+'_'+str(batch_size)+'_'+str(gen_dim)+'_'+str(dis_dim)+'_'+str(em_dim)       
                performance_df.loc[i,'Combination_parameters'] = combination  
            else:
                performance_df.loc[i,'Combination_parameters'] = combination
        else:
            performance_df.loc[i,'Combination_parameters'] = combination
        ctgan_report = QualityReport()
        st = time.process_time()
        print('The next combination of parameters is tried now: ', combination)
        model = CTGAN(table_metadata=metadata,verbose=True,epochs=epochs,batch_size=batch_size,generator_dim=(gen_dim,gen_dim),discriminator_dim=(dis_dim,dis_dim),embedding_dim=em_dim)
        model.fit(data) #,discrete_columns=discrete_columns)
        synthetic_data = model.sample(num_rows=len(data))
        print('Synthetic data has been generated.')
        et = time.process_time()
        ctgan_report.generate(data,synthetic_data,metadata)
        properties = ctgan_report.get_properties()
       # print(properties)
        performance_df.loc[i,'Nr_rows'] = len(data)
        performance_df.loc[i,'Nr_epochs'] = epochs
        performance_df.loc[i,'Batch_size'] = batch_size
        performance_df.loc[i,'Generator_dimension'] = gen_dim
        performance_df.loc[i,'Discriminator_dimension'] = dis_dim
        performance_df.loc[i,'Embedding_dimension'] = em_dim
        performance_df.loc[i,'Overall_score'] = ctgan_report.get_score()
        performance_df.loc[i,'Column_shapes'] = properties.loc[properties['Property']=='Column Shapes','Score'].values[0]
        performance_df.loc[i,'Column_pair_trends'] = properties.loc[properties['Property']=='Column Pair Trends','Score'].values[0]
        performance_df.loc[i,'Row_synthesis'] = NewRowSynthesis.compute(
                                                    real_data = data,
                                                    synthetic_data = synthetic_data,
                                                    metadata = metadata,
                                                    numerical_match_tolerance=0.01,
                                                    synthetic_sample_size = 1000)
        performance_df.loc[i,'CPU Execution time'] = et - st
        print("Performance dataframe has been filled with current combination.")
        i += 1

    if data_type=='wwb':
        performance_ctgan = performance_df.to_csv('performance_ctgan_file_wwb.csv',encoding='utf-8',index=False)
    else:
        performance_ctgan = performance_df.to_csv('performance_ctgan_file_polis.csv',encoding='utf-8',index=False)

    return performance_ctgan
