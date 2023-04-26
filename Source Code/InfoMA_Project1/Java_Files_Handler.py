import javalang
import os
import pandas as pd

def collect_java_files():
    java_files_list = []
    for root, dirs, files in os.walk(os.path.abspath(os.pardir) + "/resources", topdown=False):
        for file_name in files:
            if file_name.endswith(".java"):
                file_path = f"{root}/{file_name}"
                java_files_list.append(file_path)
    return java_files_list


def parse_java_file(file_path):
    classes_list = []
    method_counter = 0
    class_name = ""
    with open(file_path, 'r') as f:
        tree = javalang.parse.parse(f.read())
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            method_counter = len(node.methods)
            classes_list.append((class_name,method_counter,file_path))
    return (classes_list)

def parse_java_collection(java_files):
    df = pd.DataFrame(columns= ["class_name","method_num", "class_path"])
    dataframe_row_index = 0
    for file in java_files:
        file_parsing_info = parse_java_file(file)
        for e in file_parsing_info:
            if e[0] != "":
                df.loc[dataframe_row_index] = list(e)
                dataframe_row_index += 1
    return df