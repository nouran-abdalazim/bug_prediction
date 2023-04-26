from sklearn.cluster import KMeans
from Utility_Functions import read_csv_file_to_dataframe
import pandas as pd

def cluster_feature_vectors_kmeans(feature_vector_file, k=5):

    god_class_data_frame = read_csv_file_to_dataframe(feature_vector_file)
    god_class_data_frame_without_method_column = god_class_data_frame.drop(['method_name'], axis=1)
    feartue_vector = god_class_data_frame_without_method_column.to_numpy()

    kmeans = KMeans(n_clusters=k).fit(feartue_vector)
    methods = god_class_data_frame['method_name'].tolist()
    cluster_labels_list = kmeans.labels_.tolist()
    method_label = {}
    for i in range(0, len(methods)):
        method_label[methods[i]] = cluster_labels_list[i]
    method_label = dict(sorted(method_label.items(), key=lambda item: item[1]))
    method_label_data_frame = pd.DataFrame(method_label.items(), columns=['method_name', 'cluster_label'])
    return method_label_data_frame, kmeans.labels_
