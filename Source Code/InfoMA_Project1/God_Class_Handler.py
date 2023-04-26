import pandas as pd
import javalang
from Utility_Functions import check_zero_columns, convert_dataframe_value_type_to_int, remove_list_duplications

def compute_population_mean(dataframe_classes_methods):
    number_methods = dataframe_classes_methods['method_num'].sum()
    avg_number_methods = number_methods / len(dataframe_classes_methods)
    return avg_number_methods

def compute_population_standard_deviation(dataframe_classes_methods):
    return dataframe_classes_methods['method_num'].std()

def compute_god_class_threshold(mean, standard_deviation):
    threshold = mean + (6 * standard_deviation)
    return threshold

def identify_god_classes (dataframe_classes_methods, god_class_threshold):
    god_classes_dataframe = pd.DataFrame(columns=["class_name", "method_num", "class_path"])
    dataframe_row_index = 0
    for index, row in dataframe_classes_methods.iterrows():
        if row["method_num"] > god_class_threshold:
            god_classes_dataframe.loc[dataframe_row_index] = list([row["class_name"], row["method_num"], row["class_path"]])
            dataframe_row_index += 1
    return god_classes_dataframe


def get_methods_accessed_by_methods(method, features_list):
    methods_list = []
    for path, node in method:
        if type(node) == javalang.tree.MethodInvocation:
            if node.member in features_list:
                if node.member not in methods_list:
                    methods_list.append(node.member)
    return methods_list

def get_fields_accessed_by_methods(method, features_list):
    fields_list = []
    for path, node in method:
        if type(node) == javalang.tree.MemberReference:
            if node.member in features_list:
                if node.member not in fields_list:
                    fields_list.append(node.member)
    return fields_list

def get_fields(java_class_tree):
    fields_list = []
    for f in java_class_tree.types[0].fields:
        if f.declarators[0].name not in fields_list:
            fields_list.append(f.declarators[0].name)
    return fields_list


def get_methods(java_class_tree):
    methods_list = []
    for m in java_class_tree.types[0].methods:
        if m.name not in methods_list:
            methods_list.append(m.name)
    return methods_list


def extract_features_from_god_class(java_class_name, java_class_path):
    feature_list = []
    tree = extract_parse_tree(java_class_name, java_class_path)
    fields_list = remove_list_duplications(get_fields(tree))
    methods_list = remove_list_duplications(get_methods(tree))
    feature_list = fields_list + methods_list
    return remove_list_duplications(feature_list)

def extract_parse_tree(java_class_name, java_class_path):
    with open(java_class_path, 'r') as f:
        tree = javalang.parse.parse(f.read())
    return tree


def construct_feature_vectors_for_class_methods (java_class_name, java_class_path, features_list):
    tree = extract_parse_tree(java_class_name, java_class_path)
    god_classes_feature_vecror_dataframe = pd.DataFrame(columns=["method_name"] + features_list)
    for m in tree.types[0].methods:
        fields_list = get_fields_accessed_by_methods(m, features_list)
        methods_list = get_methods_accessed_by_methods(m, features_list)

        if god_classes_feature_vecror_dataframe['method_name'].isin([m.name]).any():
            method_row_index = god_classes_feature_vecror_dataframe.index[god_classes_feature_vecror_dataframe['method_name'] == m.name].tolist()[0]
            for f in fields_list:
                god_classes_feature_vecror_dataframe.at[method_row_index,f] = 1
            for method in methods_list:
                god_classes_feature_vecror_dataframe.at[method_row_index,method] = 1

        else:
            method_features_vector = [0] * len(features_list)
            for f in fields_list:
                feature_index = (features_list.index(f))
                method_features_vector[feature_index] = 1
            for method in methods_list:
                feature_index = (features_list.index(method))
                method_features_vector[feature_index] = 1
            row = [m.name] + method_features_vector
            god_classes_feature_vecror_dataframe.loc[len(god_classes_feature_vecror_dataframe)] = row

    god_classes_feature_vecror_dataframe = check_zero_columns(god_classes_feature_vecror_dataframe)
    return convert_dataframe_value_type_to_int(god_classes_feature_vecror_dataframe)





