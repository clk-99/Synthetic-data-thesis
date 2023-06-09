import pandas as pd
import re
import argparse, json
import visuals as vs
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Merge datasets to combine in one performance plot')
parser.add_argument("-l","--list",help='Which datasets in plot?',type=str)

args = parser.parse_args()
data_folder = '../Data' 
args.list = eval(args.list)
datasets = args.list

def merge_results_dataset(datasets,current_directory):
    merged_dfs = pd.DataFrame()
    for d in datasets:
        dataset_path = current_directory + '/' + d  
        with os.scandir(dataset_path) as it:
            output_path = dataset_path
            os.chdir(output_path)
            print(output_path)
            for result in it:
                print(result.name)  
                if re.search('final_results_SDG',result.name):
                    performance_df = pd.read_csv(result.name,delimiter=',',index_col=0)  
                    if "Dataset_x" in performance_df:
                        if "Dataset_y" in performance_df:
                            print("Something went wrong during merge!")
                            performance_df.rename(columns = {'Dataset_x':'Dataset'},inplace=True)  
                            print(performance_df.head()) 
                            performance_df.drop(columns=["Dataset_y"],inplace=True)
                    temp_df = pd.concat([merged_dfs,performance_df],ignore_index=True)
                    merged_dfs = temp_df
                else:
                    #to skip train and test set .csv files and break current for loop.
                    print('Incorrect file to proceed in current directory')                    
                    continue
        os.chdir('..')
        print(os.getcwd())
       

    return merged_dfs

if datasets:
    merged_performance = merge_results_dataset(datasets,data_folder)
    print(merged_performance.head())
    print(merged_performance.tail())
    print(merged_performance.shape)
    vs.create_performance_SDGs_all(merged_performance, datasets, data_folder)
    