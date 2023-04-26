from Java_Files_Handler import collect_java_files, parse_java_collection
from God_Class_Handler import compute_population_mean, compute_population_standard_deviation, \
    compute_god_class_threshold, identify_god_classes, extract_features_from_god_class, \
    construct_feature_vectors_for_class_methods
import pandas as pd
from Clustering_Kmean import cluster_feature_vectors_kmeans
from Clustering_Agglomerative import cluster_feature_vectors_agglomerative
from Silhouette import compute_silhouette
from Ground_Truth import assign_methods_to_ground_truth
from Recall_Precision import calculate_recall_precision
from Utility_Functions import get_file_name_from_file_path, read_csv_file_to_dataframe
import numpy
import csv
import sys
import glob
import os

if __name__ == '__main__':


    ##==================================== Step 1 Starts ====================================##

    print("****** Project step 1 started ******")

    java_files_list = collect_java_files()
    dataframe_classes_methods = parse_java_collection(java_files_list)
    avg_number_methods = compute_population_mean(dataframe_classes_methods)
    methods_standard_deviation = compute_population_standard_deviation(dataframe_classes_methods)
    god_class_threshold= compute_god_class_threshold(avg_number_methods, methods_standard_deviation)
    god_classes_dataframe = identify_god_classes(dataframe_classes_methods,god_class_threshold)


    print("Java classes count = ",len(java_files_list))
    print("Average methods count = ", avg_number_methods)
    print("Methods standard deviation = ", methods_standard_deviation)
    print("God classes threshold = ", god_class_threshold)
    print("God classes count = ", len(god_classes_dataframe))
    print("God classes are : ")
    for index, row in god_classes_dataframe.iterrows():
        print("Name : ",row["class_name"], ", Method count = ",row["method_num"])
    print("****** Project step 1 ended ******")
    ##==================================== Step 1 Ends=======================================##

    ##==================================== Step 2 Starts ====================================##
    print("****** Project step 2 started ******")
    for index, row in god_classes_dataframe.iterrows():
         print("Extract feature list for god class", row["class_name"])
         features_list = extract_features_from_god_class(row["class_name"],row["class_path"])
         print("Features count without removing the all zeroes columns for class", row["class_name"], " = ", len(features_list))
         feature_vectors_data_frame = construct_feature_vectors_for_class_methods(row["class_name"],row["class_path"], features_list)
         print("Features count after removing the all zeroes columns for class", row["class_name"], " = ", len(feature_vectors_data_frame.columns))
         print("Features vectors count for class", row["class_name"], " = ", len(feature_vectors_data_frame))
         print("Generate feature vector csv file for god class", row["class_name"])
         feature_vectors_data_frame.to_csv("./Feature Vectors/"+row["class_name"]+".csv",",",".csv",index=False)
    print("****** Project step 2 ended ******")
    ##==================================== Step 2 Ends=======================================##

    ##==================================== Step 3 Starts ====================================##
    print("****** Project step 3 started ******")
    optimal_k_kmeans_cluster = {}
    optimal_k_agglomerative_cluster = {}
    path = "./Feature Vectors/*.csv"
    for file_path in glob.glob(path):
        god_class_name = os.path.basename(file_path).split('.')[0]

        optimal_k_score_kmeans = float('-inf')
        optimal_k_score_agglomerative = float('-inf')
        optimal_k_value_kmeans = float('-inf')
        optimal_k_value_agglomerative = float('-inf')
        optimal_kmeans_clustering_result= pd.DataFrame()
        optimal_agglomerative_clustering_result = pd.DataFrame()
        kmeans_silhouette_scores_dict = {}
        agglomerative_silhouette_scores_dict = {}


        print("Iteration over k values started")
        for k in range(2,61):
            print("Iteration K = ", k)

            print("Applying Kmeans clustering to god class", god_class_name)
            Kmeans_method_label_data_frame, Kmean_cluster_labels = cluster_feature_vectors_kmeans(feature_vector_file= file_path, k= k )
            print("Applying Silhouette for Kmeans clustering results for god class", god_class_name)
            current_kmeans_silhouette_score = compute_silhouette(feature_vector_file= file_path,clustering_file=None,k=k, clustering_labels=Kmean_cluster_labels)
            kmeans_silhouette_scores_dict[k] = current_kmeans_silhouette_score
            if current_kmeans_silhouette_score > optimal_k_score_kmeans:
                optimal_k_score_kmeans = current_kmeans_silhouette_score
                optimal_k_value_kmeans = k
                optimal_kmeans_clustering_result = Kmeans_method_label_data_frame

            print("Applying Agglomerative clustering to god class", row["class_name"])
            Agglomerative_method_label_data_frame, Agglomerative_cluster_labels = cluster_feature_vectors_agglomerative(feature_vector_file= file_path, k= k )
            print("Applying Silhouette for Agglomerative clustering results for god class", god_class_name)
            current_Agglomerative_silhouette_score = compute_silhouette(feature_vector_file= file_path,clustering_file=None,k=k, clustering_labels=Agglomerative_cluster_labels )
            agglomerative_silhouette_scores_dict[k] = current_Agglomerative_silhouette_score
            if current_Agglomerative_silhouette_score > optimal_k_score_agglomerative:
                optimal_k_score_agglomerative = current_Agglomerative_silhouette_score
                optimal_k_value_agglomerative = k
                optimal_agglomerative_clustering_result = Agglomerative_method_label_data_frame

        print("Iteration over k values ended")
        optimal_k_kmeans_cluster[row["class_name"]] = [optimal_k_value_kmeans, optimal_k_score_kmeans]
        optimal_k_agglomerative_cluster[row["class_name"]] = [optimal_k_value_agglomerative, optimal_k_score_agglomerative]
        print("Exporting Kmeans clustering optimal results csv file for god class", row["class_name"])
        optimal_kmeans_clustering_result.to_csv("./Clustering/Kmeans/" + god_class_name + ".csv", ",", ".csv",
                                              index=False)

        print("Exporting Agglomerative optimal clustering results csv file for god clas", god_class_name)
        optimal_agglomerative_clustering_result.to_csv("./Clustering/Agglomerative/" + god_class_name + ".csv", ",",
                                                     ".csv",
                                                     index=False)

        with open("./Silhouette/Kmeans/"+god_class_name+".csv", 'w') as csvfile:
            for key in kmeans_silhouette_scores_dict.keys():
                csvfile.write("%s,%s\n" % (key, kmeans_silhouette_scores_dict[key]))


        with open("./Silhouette/Agglomerative/" + god_class_name + ".csv", 'w') as csvfile:
            for key in agglomerative_silhouette_scores_dict.keys():
                csvfile.write("%s,%s\n" % (key, agglomerative_silhouette_scores_dict[key]))

    print("Final optimal results for kmeans clustering")
    print(optimal_k_kmeans_cluster)
    print("Final optimal results for agglomerative clustering")
    print(optimal_k_agglomerative_cluster)
    print("****** Project step 3 ended ******")
    ##==================================== Step 3 Ends=======================================##

    ##==================================== Step 4 Starts=======================================##
    assign_methods_to_ground_truth()
    god_class_evaluation_dict = {}
    path = "./Ground Truth/*.csv"
    for ground_truth_fname in glob.glob(path):
        god_class_name = get_file_name_from_file_path(ground_truth_fname)
        kmeans_file_name = "./Clustering/Kmeans/"+god_class_name+".csv"
        agglomerative_file_name = "./Clustering/Agglomerative/"+god_class_name+".csv"
        kmeans_recall, kmeans_precision = calculate_recall_precision(ground_truth_fname, kmeans_file_name)
        agglomerative_recall, agglomerative_precision = calculate_recall_precision(ground_truth_fname, agglomerative_file_name)

        god_class_evaluation_dict [god_class_name] = {"kmeans_recall": kmeans_recall, "kmeans_precision": kmeans_precision,
                                                      "agglomerative_recall": agglomerative_recall, "agglomerative_precision":agglomerative_precision}

        print("Kmeans: "+god_class_name+" precision = "+str(kmeans_precision)+" recall = "+str(kmeans_recall))
        print("Agglomerative: "+god_class_name+" precision = "+str(agglomerative_precision)+" recall = "+str(agglomerative_recall))

    with open("./god_class_evaluation.csv", 'w') as csvfile:
        for key in god_class_evaluation_dict.keys():
            csvfile.write("%s,%s\n" % (key, god_class_evaluation_dict[key]))
    ##==================================== Step 4 Ends=========================================##

    ##==================================== Step 5 Starts (Run some statistics to be used in the repotr)=========================================##
    path = "./Clustering/Kmeans/*.csv"
    evalutaion_file = open('./god_class_evaluation.csv', 'a')

    for kmeans_clustering_file in glob.glob(path):
        god_class_name = get_file_name_from_file_path(kmeans_clustering_file)
        evalutaion_file.write('god class: '+ god_class_name+'\n')
        evalutaion_file.write('kmeans result:\n')

        kmeans_clustering_dataframe = read_csv_file_to_dataframe(kmeans_clustering_file)
        cluster_label_list = numpy.unique(numpy.array(kmeans_clustering_dataframe['cluster_label'].tolist()))

        evalutaion_file.write('no. clusters = '+ str(len(cluster_label_list)) +'\n')
        evalutaion_file.write('clusters statistics :\n')


        for label in cluster_label_list:
            selected_methods_of_current_label = kmeans_clustering_dataframe.apply(lambda x: True
            if x['cluster_label'] == label else False, axis=1)
            num_rows = len(selected_methods_of_current_label[selected_methods_of_current_label == True].index)
            evalutaion_file.write('     label = ' + str(label) + 'count = '+ str(num_rows) +'\n')

        agglomerative_clustering_file = "./Clustering/Agglomerative/"+god_class_name+".csv"
        agglomerative_clustering_dataframe = read_csv_file_to_dataframe(agglomerative_clustering_file)
        cluster_label_list = numpy.unique(numpy.array(agglomerative_clustering_dataframe['cluster_label'].tolist()))

        evalutaion_file.write('agglomerative result:\n')
        evalutaion_file.write('no. clusters = '+ str(len(cluster_label_list)) +'\n')
        evalutaion_file.write('clusters statistics :\n')

        for label in cluster_label_list:
            selected_methods_of_current_label = agglomerative_clustering_dataframe.apply(lambda x: True
            if x['cluster_label'] == label else False, axis=1)
            num_rows = len(selected_methods_of_current_label[selected_methods_of_current_label == True].index)
            evalutaion_file.write('     label = ' + str(label) + 'count = ' + str(num_rows) + '\n')

    evalutaion_file.close()

    ##==================================== Step 5 Ends (Run some statistics to be used in the repotr)===========================================##



