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

from dython.nominal import numerical_encoding

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

def KStest(real_df,syn_df,cat_columns): #statistical test
    """
    A function that returns the mean KS test for all numeric features.
    """
    features = real_df.columns.to_list()
    for c in cat_columns:
        features.remove(c)
    
    ks_statistic = {}

    for num in features:
        result = ss.ks_2samp(real_df[num].values,syn_df[num].values)
        statistic = result[0]
        #p_value = result[1]
        ks_statistic[num] = statistic
    
    mean_ks = sum(ks_statistic.values())/len(ks_statistic)

    return mean_ks

def EStest(real_df,syn_df,cat_cols): #statistical test
    """
    A function that returns the mean ES test for all features.
    """
    features = real_df.columns.to_list()
    es_statistic = {}

    for f in features:
        #to extract distribution per feature
        p, bins0, ignored0 = plt.hist(real_df[f],10,density=True,color='b',alpha=0.3,label='Real') 
        q, bins2, ignored2 = plt.hist(syn_df[f],10,density=True,color='r',alpha=0.3,label='Synthetic')       
        result = ss.epps_singleton_2samp(np.array(p), np.array(q))
        statistic = result[0]
        #p_value = result[1]
        es_statistic[f] = statistic
    
    mean_es = sum(es_statistic.values())/len(es_statistic)
    
    return mean_es

def MLefficiency(syn_df, test_df, cat_cols, target_type='class',multi=False): #own metric to test the performance of synthetic dataset
    syn_data = numerical_encoding(syn_df, nominal_columns=cat_cols) #one-hot encoding of categorical variables
    test_data = numerical_encoding(test_df, nominal_columns=cat_cols)

    X_train = syn_df.iloc[:,:-1]
    y_train = syn_df.iloc[:,-1].round(decimals=0)

    X_test = test_df.iloc[:,:-1]
    y_test = test_df.iloc[:,-1].round(decimals=0).values
    
    performance_metrics = {}
    if target_type == 'regr':
        rf = RandomForestRegressor()
        rf.fit(X_train,y_train)
        y_pred = np.round(rf.predict(X_test),0)     

        performance_metrics['Explained_variance'] = explained_variance_score(y_test, y_pred)
        performance_metrics['Mean_squared_error'] = mean_squared_error(y_test, y_pred)
        performance_metrics['R^2_score'] = r2_score(y_test, y_pred)
    else:
        rf = RandomForestClassifier()
        rf.fit(X_train,y_train)
        y_pred = np.round(rf.predict(X_test),0)

        if multi: #multi-class classification
            performance_metrics['AUC'] = roc_auc_score(y_test, y_pred,multi_class='ovr')
            performance_metrics['F1_score'] = f1_score(y_test, y_pred,average='weighted')
        else: #binary classification
            performance_metrics['AUC'] = roc_auc_score(y_test, y_pred)
            performance_metrics['F1_score'] = f1_score(y_test, y_pred)
        
        performance_metrics['Accuracy'] = accuracy_score(y_test, y_pred)  
        

    return performance_metrics

#en dan met een andere evaluatie score zoals AUC