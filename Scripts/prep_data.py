import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dabl
import datasist as ds

#import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import os
data_folder = '../Data'

def main(data_path,dataset,cat_columns,path_to_save):
    datasets = ['bank','adult']
    df, num_vars = load_dataset(dataset,data_path,cat_columns)
    print(num_vars)
    df, target = dataframe_investigation(df, dataset)
    print(df.head())
    target_df = df[target]
    if cat_columns is not None:
        features_df = df[cat_columns+num_vars]
    else:
        features_df = df[num_vars]

    X_train, X_test, y_train, y_test = train_test_split(features_df,target_df,test_size=0.20,random_state=12) #voeg nog stratify parameter toe

    print(X_train.head())
    if dataset in [datasets]:
        X_train, X_test = feature_engineering(X_train, X_test, dataset, cat_vars=cat_columns)
    
    print(X_train.head())
    real_train_data = pd.concat([X_train.reset_index(drop=True),
                                    y_train.reset_index(drop=True)],axis=1)
    print(real_train_data.head())
    print(real_train_data.dtypes)
    real_train_data.to_csv(data_folder + path_to_save,index_label='Index')
    
    return real_train_data, target  


def load_dataset(data_type,data_path,cat_columns):
    file = (data_path)
    if data_type == 'heart':
        df = pd.read_csv(heart_data,sep=',',names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'],dtype={col:'object' for col in cat_columns})
    elif data_type == 'bank':
         df = pd.read_csv(file,encoding='utf8',sep=';',dtype={col:'object' for col in cat_columns})
    
    elif cat_columns is not None:
        df = pd.read_csv(file,encoding='utf8',sep=',',dtype={col:'object' for col in cat_columns})
    
    else:
        df = pd.read_csv(file,encoding='utf8',sep=',',dtype={'quality':'object'}) #wine dataset

    print('Size of dataset equals: ',df.shape)
    print('\n First five rows of dataset \n',df.head())
    print('\n Info about nan values: \n',df.info())
    num_cols = df.select_dtypes(exclude='object').columns.to_list()

    return df, num_cols

def check_null_values(dataframe):
    columns = dataframe.columns[dataframe.isnull().any()].to_list()

    if len(columns) > 0:
        for i in columns:
            mode_var = dataframe[i].mode()
            dataframe[i].fillna(mode_var, inplace=True) #missing values are filled using rolling mode method
    else:
        print("Data does not have missing values.")

    return dataframe

def dataframe_investigation(df,dataset):
    features = pd.value_counts(df.dtypes)
    print('Count of numerical and categorical columns: \n', features)   
    
    if dataset == 'adult':
        df[df =='?'] = np.nan
        #impute missing values with mode
        df = check_null_values(df)
        #for c in ['workclass','education','marital.status','occupation','relationship','race','sex']:
         #   df[c]=df[c].replace(df[c].unique(),[i for i in range(df[c].nunique())])
        target_var = 'income'   

        return df,target_var

    elif dataset == 'heart':
        target_var = 'num'
        df_plot = df.copy()
        di = {'1': 'Male', '0': 'Female'}
        dj = {'3':'normal','6':'fixed defect','7':'reversable defect'}
        dk = {'0':'Less chance of Heart Attack','1':'High Chance of Heart Attack'}

        df_plot['sex'].replace(di,inplace=True)
        df_plot['thal'].replace(dj,inplace=True)
        df_plot['num'].replace(dk,inplace=True)
       
        return df_plot,target_var
    
    elif dataset == 'bank':
        target_var = 'y'
        #impute missing values with mode
        df = check_null_values(df)

        return df,target_var        
    
    else: #wine
        target_var = 'quality'
        df[target_var].replace(df[target_var].unique(),[i for i in range(df[target_var].nunique())],inplace=True)

        #impute missing values with mode
        df = check_null_values(df)

        return df,target_var
    

def feature_engineering(train_df=None, test_df=None, dataset='bank', cat_vars=None):
    #add feature scaling?
    #only applied to bank and adult datasets
    #datasets = ['bank','adult']
    #if dataset in datasets:
    for feature in cat_vars:
        le = preprocessing.LabelEncoder()
        train_df[feature] = le.fit_transform(train_df[feature])
        test_df[feature] = le.transform(test_df[feature])
    
    print(train_df.head())
    print(test_df.head())
    return train_df,test_df
    
    #else: #wine and heart
       # return print('Not needed')
        
    
