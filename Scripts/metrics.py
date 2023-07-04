import pandas as pd
import numpy as np
import math
import scipy.stats as ss
import dython
import sklearn
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import  roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, explained_variance_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def hellinger_distance(real_df,syn_df): #for numeric features only?
    """
    A function that returns the mean Hellinger Distance of real and synthetic datasets over all features.
    """
    features = real_df.columns.to_list()
    hellinger = {}

    for f in features:
   
        p, bins0, ignored0 = plt.hist(real_df[f],10,density=True,color='b',alpha=0.3,label='Real')
        q, bins2, ignored2 = plt.hist(syn_df[f],10,density=True,color='r',alpha=0.3,label='Synthetic')
        hd = math.sqrt(
            sum([(math.sqrt(p_i) - math.sqrt(q_i)) ** 2 for p_i, q_i in zip(p,q)]) / 2
        )
        print("Hellinger distance for feature "+str(f)+" equals: " ,hd)
        hellinger[f] = hd
    
    hellinger_distance = sum(hellinger.values()) / len(hellinger)
    
    return hellinger_distance

def log_cluster(real_df,syn_df): #cluster metric

    real_df['Data'] = 'Real'
    syn_df['Data'] = 'Synthetic'

    #merged_df = 
    return

def KStest(real_df,syn_df,cat_columns,target_var): #statistical test
    """
    A function that returns the mean KS test for all numeric features.
    """
    features = real_df.columns.to_list()
    for c in cat_columns:
        features.remove(c)
    if target_var in features:
        features.remove(target_var)
    
    ks_statistic = {}

    for num in features:
        result = ss.ks_2samp(real_df[num].values,syn_df[num].values)
        statistic = result[0]
        #p_value = result[1]
        ks_statistic[num] = statistic
    
    mean_ks = sum(ks_statistic.values())/len(ks_statistic)

    return mean_ks

def unique_values_check(df1, df2):
    missing_cols = set(df1.columns.to_list()) - set(df2.columns.to_list())
    print(missing_cols)
    for m in missing_cols:
        df2[m] = 0

    return df2

def numerical_encoding(dataset,nominal_columns):
    converted_dataset = pd.DataFrame()
    
    for col in dataset.columns:
        if col not in nominal_columns:
            converted_dataset.loc[:, col] = dataset[col]
        else:
            unique_values = pd.unique(dataset[col])
            dummies = pd.get_dummies(dataset[col], prefix=col)
            converted_dataset = pd.concat(
                [converted_dataset, dummies], axis=1
            )
    
    return converted_dataset

def MLefficiency(syn_df, test_df, cat_cols, target_var, target_type='class', multi=False): #own metric to test the performance of synthetic dataset    
    cat_vars = cat_cols.copy()
    if not multi:
        cat_vars.remove(target_var)
        syn_df[target_var] = pd.to_numeric(syn_df[target_var])
        test_df[target_var] = pd.to_numeric(test_df[target_var])

    syn_data = numerical_encoding(syn_df, nominal_columns=cat_vars) #one-hot encoding of categorical variables
    test_data = numerical_encoding(test_df, nominal_columns=cat_vars)
    print(syn_data.head())
    print(test_data.head())
    syn_data = unique_values_check(test_data, syn_data)
    test_data = unique_values_check(syn_data,test_data)
    syn_data = syn_data[sorted(syn_data.columns)]
    test_data = test_data[sorted(test_data.columns)]

    if multi:
        all_columns = syn_data.columns.to_list()
        target_columns = list(filter(lambda x:target_var in x,all_columns))
        X_train = syn_data.loc[:,~syn_data.columns.isin(target_columns)].round(decimals=0)
        y_train = syn_data[target_columns].round(decimals=0).values

        X_test = test_data.loc[:,~test_data.columns.isin(target_columns)].round(decimals=0)
        y_test = test_data[target_columns].round(decimals=0).values
    else:
        X_train = syn_data.loc[:,syn_data.columns!=target_var].round(decimals=0)
        y_train = syn_data[target_var].values

        X_test = test_data.loc[:,test_data.columns!=target_var].round(decimals=0)
        y_test = test_data[target_var].values   


    performance_metrics = {}
    if target_type == 'regr':
        rf = RandomForestRegressor()
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)

        performance_metrics['Explained_variance'] = explained_variance_score(y_test, y_pred)
        performance_metrics['Mean_squared_error'] = mean_squared_error(y_test, y_pred)
        performance_metrics['R^2_score'] = r2_score(y_test, y_pred)
    else:
        rf = RandomForestClassifier()
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)

        if multi: #multi-class classification
            performance_metrics['AUC'] = roc_auc_score(y_test, y_pred,multi_class='ovr')
            performance_metrics['F1_score'] = f1_score(y_test, y_pred,average='weighted')
        else: #binary classification
            performance_metrics['AUC'] = roc_auc_score(y_test, y_pred)
            performance_metrics['F1_score'] = f1_score(y_test, y_pred,average='binary')
        
        performance_metrics['Accuracy'] = accuracy_score(y_test, y_pred)  
        

    return performance_metrics