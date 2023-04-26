import glob
import os
import pandas as pd


def read_ground_truth_keywords():
    ground_truth_keywords = []
    lines = []
    with open('keywords.txt') as f:
        lines = f.readlines()

    for line in lines:
        ground_truth_keywords.append(line.strip('\n'))

    return ground_truth_keywords

def assign_methods_to_ground_truth():
    ground_truth_keywords_list = read_ground_truth_keywords()
    path = "./Feature Vectors/*.csv"
    for fname in glob.glob(path):
        god_class_name = os.path.basename(fname).split('.')[0]
        god_class_data_frame = pd.read_csv(fname, index_col=False)
        methods_list = god_class_data_frame['method_name'].tolist()
        method_ground_truth_label = {}
        for m in methods_list:
            found_flag = False
            for k in ground_truth_keywords_list:
                if k.lower() in m.lower():
                    method_ground_truth_label[m] = ground_truth_keywords_list.index(k)
                    found_flag = True
                    break
            if found_flag == False:
                method_ground_truth_label[m] = len(ground_truth_keywords_list)

        with open("./Ground Truth/" + god_class_name + ".csv", 'w') as csvfile:
            csvfile.write("%s,%s\n" % ("method_name", "cluster_label"))
            for key in method_ground_truth_label.keys():
                csvfile.write("%s,%s\n" % (key, method_ground_truth_label[key]))



