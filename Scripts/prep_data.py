import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import dabl
import zipfile

#import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

#import self-written library
import visuals

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import os
data_folder = '../Data'

def main(data_path,dataset,cat_columns,target,outliers_bool):
    features = cat_columns.copy()
    df, num_vars = load_dataset(dataset,data_path,cat_columns)
    df = dataframe_investigation(df, dataset, outliers_bool)
    if dataset != 'covertype':
        df = label_encoding(df, cat_columns)
    target_df = df[target]
    if dataset != 'metro':
        features.pop(-1)
    if cat_columns is not None:
        features_df = df[features+num_vars]

    X_train, X_test, y_train, y_test = train_test_split(features_df,target_df,test_size=0.20,shuffle=False) #voeg nog stratify parameter toe

    real_train_data = pd.concat([X_train.reset_index(drop=True),
                                    y_train.reset_index(drop=True)],axis=1)
    real_test_data = pd.concat([X_test.reset_index(drop=True),
                                    y_test.reset_index(drop=True)],axis=1)
    
    table_dict = create_metadata(real_train_data) #create metadata dictionary for CTGAN and TVAE

    real_train_data.to_csv(data_folder + '/' + dataset + '/' + dataset +'_TRAIN_SET.csv',index_label='Index')
    real_test_data.to_csv(data_folder + '/' + dataset + '/' + dataset +'_TEST_SET.csv',index_label='Index')


    return real_train_data, table_dict  


def load_dataset(data_type,data_path,cat_columns):
    file = (data_path)      
    types = ['metro','adult','iris']   

    if data_type in types:
        df = pd.read_csv(file,sep=',',index_col=0,dtype={col:'object' for col in cat_columns})
    
    elif data_type == 'covertype':
        zf = zipfile.ZipFile(file)
        df = pd.read_csv(zf.open(data_type+'.csv'),index_col=0,dtype={col:'object' for col in cat_columns})
        
    else: #bank dataset    
        df = pd.read_csv(file,encoding='utf8',sep=';',dtype={col:'object' for col in cat_columns})

    print('Size of dataset equals: ',df.shape)
    print('\n First five rows of dataset \n',df.head())
    print('\n Info about nan values: \n',df.info())

    num_cols = df.select_dtypes(exclude='object').columns.to_list()
    if data_type == 'metro':
        df.drop('date_time',axis=1,inplace=True)  
        num_cols.remove('traffic_volume')    
    
    if data_type == 'adult':
        df[df =='?'] = np.nan

    return df, num_cols

def check_null_values(dataframe):
    columns = dataframe.columns[dataframe.isnull().any()].to_list()
    if len(columns) > 0:
        for i in columns:
            mode_var = dataframe[i].mode()[0]
            dataframe[i].fillna(mode_var, inplace=True) #missing values are filled using rolling mode method
    else:
        print("Data does not have missing values.")

    return dataframe

def dataframe_investigation(df,dataset,outliers_bool):
    features = pd.value_counts(df.dtypes)
    print('Count of numerical and categorical columns: \n', features)   

    #impute missing values with mode
    df = check_null_values(df)
    features = df.columns.to_list()
    if outliers_bool:
        df = visuals.find_outliers(df,features,0.1)

    return df
    
def label_encoding(df,cat_columns):

    for cat in cat_columns:
        df[cat] = LabelEncoder().fit_transform(df[cat])
    
    return df

def create_metadata(df):    
    table_dict = dict()
    table_dict['METADATA_SPEC_VERSION'] = 'SINGLE_TABLE_V1'
    object_columns = df.dtypes[df.dtypes=='object'].index.to_list()
    int_columns = df.dtypes[df.dtypes=='int64'].index.to_list()
    float_columns = df.dtypes[df.dtypes=='float64'].index.to_list()
    all_columns = df.dtypes.index.to_list()
    col_dict = dict()

    for col in all_columns:
        #create for each column a dictionary and add to table_dict
        if col in float_columns:
            float_dict = dict()
            float_dict['sdtype'] = 'numerical'
            float_dict['computer_representation'] = 'Float'
            col_dict[col] = float_dict

        elif col in int_columns:
            int_dict = dict()
            int_dict['sdtype'] = 'numerical'
            int_dict['computer_representation'] = 'Int64'
            col_dict[col] = int_dict
        
        else: #col in object_columns
            object_dict = dict()
            object_dict['sdtype'] = 'categorical'
            col_dict[col] = object_dict

    table_dict['columns'] = col_dict

    return table_dict    

        
    
