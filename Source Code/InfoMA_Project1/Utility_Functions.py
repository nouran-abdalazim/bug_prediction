import numpy as np
import pandas as pd
import os

def remove_list_duplications(my_list):
  return list(dict.fromkeys(my_list))

def convert_dataframe_value_type_to_int(dataframe):
    for column in dataframe.columns:
        if column == "method_name":
            continue
        dataframe[column] = dataframe[column].astype('int')
    return dataframe

def check_zero_columns (dataframe):
    to_drop_columns_name = []
    for column in dataframe.columns:
        if dataframe[column].to_numpy().sum() == 0:
            to_drop_columns_name.append(column)
    dataframe = dataframe.drop(columns=to_drop_columns_name)
    return dataframe

def read_csv_file_to_dataframe(file_path, index_col_falg = False):
    data_frame = pd.read_csv(file_path, index_col=index_col_falg)
    return data_frame

def get_file_name_from_file_path(file_path):
    file_name = os.path.basename(file_path).split('.')[0]
    return file_name

