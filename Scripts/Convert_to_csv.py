import pandas as pd
import argparse
import pyreadstat

parser = argparse.ArgumentParser(description='Convert data file to csv')
parser.add_argument("dataset", help='Name of dataset to convert filetype',type=str)
parser.add_argument("path_data", help="Directory containing folder where data file is saved", type=str)
parser.add_argument("path_csv", help="Directory containing folder where to save the data in csv format", type=str)

args = parser.parse_args()

if args.dataset:
    data_path = args.path_data
    dataset = args.dataset
    csv_path = args.path_csv
    df = pd.read_spss(data_path)
    print(df.head(5))
    df.to_csv(csv_path+"/"+dataset+".csv")