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

def main(data_path,dataset,cat_columns):
    df, num_vars = load_dataset(dataset,data_path,cat_columns)
    df, target = dataframe_investigation(df, dataset)
    target_df = df[target]
    features_df = df[cat_columns+num_vars]
    if dataset == 'wine':
        target_df = feature_engineering(target_df,None,dataset,None)
        X_train, X_test, y_train, y_test = train_test_split(features_df,target_df,test_size=0.20,random_state=12) #voeg nog stratify parameter toe
        real_train_data = pd.concat([X_train.reset_index(drop=True),
                                        y_train.reset_index(drop=True)],axis=1)
        real_train_data.to_csv(data_folder + '/WINE_QUALITY/WINE_TRAIN_SET.csv',index_label='Index')

    elif dataset == 'bank':
        X_train, X_test, y_train, y_test = train_test_split(features_df,target_df,test_size=0.20,random_state=12)
        X_train, X_test = feature_engineering(X_train, X_test, dataset, cat_vars=cat_columns)
        real_train_data = pd.concat([X_train.reset_index(drop=True),
                                        y_train.reset_index(drop=True)],axis=1)
        real_train_data.to_csv(data_folder + '/BANK_MARKETING/BANK_TRAIN_SET.csv',index_label='Index')
    
    elif dataset == 'adult':
        X_train, X_test, y_train, y_test = train_test_split(features_df,target_df,test_size=0.20,random_state=12)
        X_train, X_test = feature_engineering(X_train, X_test, dataset, cat_vars=cat_columns)
        real_train_data = pd.concat([X_train.reset_index(drop=True),
                                        y_train.reset_index(drop=True)],axis=1)
        real_train_data.to_csv(data_folder + '/ADULT_CENSUS_INCOME/ADULT_TRAIN_SET.csv',index_label='Index')
    
    else: #heart dataset
        X_train, X_test, y_train, y_test = train_test_split(features_df,target_df,test_size=0.20,random_state=12)
        real_train_data = pd.concat([X_train.reset_index(drop=True),
                                        y_train.reset_index(drop=True)],axis=1)
        real_train_data.to_csv(data_folder + '/HEART_ATTACK_PREDICTION/HEART_TRAIN_SET.csv',index_label='Index')
    
    return real_train_data, target
        
  


def load_dataset(data_type,data_path,cat_columns):
    file = (data_path)
    if data_type == 'heart':
        df = pd.read_csv(heart_data,sep=',',header=0,names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'],dtype={col:'object' for col in cat_columns})
    else:
        df = pd.read_csv(file,encoding='utf8',sep=',',dtype={col:'object' for col in cat_columns})
    print('Size of dataset equals: ',df.shape)
    print('\n First five rows of dataset \n',df.head())
    print('\n Info about nan values: \n',df.info())
    
    num_cols = df.select_dtypes(exclude='object').columns.to_list()

    return df, num_vars

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
        #impute missing values with mode
        df = check_null_values(df)

        return df,target_var
    

def feature_engineering(train_df=None, test_df=None, dataset='wine', cat_vars=None):
    #add feature scaling?
    if dataset == 'wine':
        #label encoding for target variable
        #feature = 'quality'
        train_df.replace(train_df.unique(),[i for i in range(train_df.nunique())],inplace=True)
        return train_df

    
    elif dataset == 'heart':
        return print('Not needed')
    
    else: #bank and adult dataset
        for feature in cat_vars:
            le = preprocessing.LabelEncoder()
            train_df[feature] = le.fit_transform(train_df[feature])
            test_df[feature] = le.transform(test_df[feature])

        return train_df,test_df
    
