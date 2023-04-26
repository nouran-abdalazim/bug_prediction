from sklearn.metrics import silhouette_score
from Utility_Functions import read_csv_file_to_dataframe, get_file_name_from_file_path
from Clustering_Kmean import cluster_feature_vectors_kmeans
from Clustering_Agglomerative import  cluster_feature_vectors_agglomerative

def compute_silhouette_score(feature_vector, clustering_lables=None):
    score = silhouette_score(feature_vector, clustering_lables)
    return score



def compute_silhouette(feature_vector_file, clustering_file = None, k = None, clustering_labels = None):

    god_class_data_frame = read_csv_file_to_dataframe(feature_vector_file)
    god_class_data_frame_without_method_column = god_class_data_frame.drop(['method_name'], axis=1)
    feartue_vector = god_class_data_frame_without_method_column.to_numpy()

    if clustering_file is None and clustering_labels is None:
        kmeans_method_label_data_frame, kmeans_labels = cluster_feature_vectors_kmeans(feature_vector_file,k)
        kmeans_silhouette = compute_silhouette_score(feartue_vector, kmeans_labels)
        agglomerative_method_label_data_frame, agglomerative_labels = cluster_feature_vectors_agglomerative(feature_vector_file,k)
        agglomerative_silhouette = compute_silhouette_score(feartue_vector, agglomerative_labels)
        return  kmeans_silhouette, agglomerative_silhouette

    elif clustering_file is None and clustering_labels is not None :
        return compute_silhouette_score(feartue_vector, clustering_labels)

    else:
        clustering_labels_data_frame = read_csv_file_to_dataframe(clustering_file)
        clustering_labels = clustering_labels_data_frame['cluster_label'].tolist()
        return compute_silhouette_score(feartue_vector, clustering_labels)







