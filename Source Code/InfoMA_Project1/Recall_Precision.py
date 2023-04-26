import pandas as pd
def compute_interpairs(grouped_data_frame):
    intrapairs_set = set()
    for index, row in grouped_data_frame.iterrows():
        method_list = row["method_name"].split(',')
        intrapairs = {(method_list[i], method_list[j]) for i in range(len(method_list)) for j in
                                     range(i + 1, len(method_list))}
        if len(intrapairs) != 0:
            intrapairs_set = intrapairs_set.union(intrapairs)

    return intrapairs_set


def calculate_recall_precision(ground_truth_file, clustering_file):
    ground_truth_file_data_frame = pd.read_csv(ground_truth_file, index_col=False)
    grouped_ground_truth_data_frame = ground_truth_file_data_frame.groupby( by="cluster_label", as_index = False).agg({'method_name': ','.join})
    ground_truth_intrapairs = compute_interpairs(grouped_ground_truth_data_frame)

    cluster_file_data_frame = pd.read_csv(clustering_file, index_col=False)
    grouped_clustering_data_frame = cluster_file_data_frame.groupby(by="cluster_label", as_index=False).agg(
        {'method_name': ','.join})
    clustering_intrapairs = compute_interpairs(grouped_clustering_data_frame)

    common_intrapirs = clustering_intrapairs.intersection(ground_truth_intrapairs)

    precision = round(len(common_intrapirs) / len(clustering_intrapairs), 4)
    recall =  round(len(common_intrapirs) / len(ground_truth_intrapairs), 4)

    return recall, precision



