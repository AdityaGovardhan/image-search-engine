import sys, os
sys.path.insert(0,'src/backend')

from database_connection import DatabaseConnection
from utils import get_svd_image_data_from_folder, get_image_names_in_a_folder, convert_folder_path_to_table_name, calculate_classification_accuracy
from singular_value_decomposition import SingularValueDecomposition
import numpy as np
import pprint
from scipy.cluster.vq import whiten

PALMAR = 'palmar'
DORSAL = 'dorsal'


class Image_Clustering:
    def __init__(self):
        self.db_conn = DatabaseConnection()
        self.no_of_dimensions = 10

    def intialize_cluster_centres(self, points, no_of_clusters):
        no_of_points = len(points)
        # first_point = round((np.random.random((1)) * no_of_points).tolist()[0])
        # print(first_point)
        # list_of_centre = [points[first_point]]
        list_of_centre = [points[0]]
        for centre in range(2, no_of_clusters):
            max_avg_dist = 0
            for point in points:
                if point in list_of_centre:
                    continue
                avg_dist = 0.0
                for centre in list_of_centre:
                    avg_dist += np.linalg.norm(np.subtract(np.array(centre), np.array(point)))
                avg_dist = avg_dist / len(list_of_centre)
                if avg_dist > max_avg_dist:
                    max_avg_dist = avg_dist
                    point_as_centre = point
            list_of_centre.append(point_as_centre)
        return list_of_centre

    def k_means(self, points, no_of_centres):

        # list_of_centre = (points.shape[0] * np.random.rand(no_of_centres,1)).tolist()
        points = points.tolist()
        list_of_centre = self.intialize_cluster_centres(points, no_of_clusters)
        clusters_points = {}
        old_clusters_centroid = {}

        clusters_centroid = { index:points[int(list_of_centre[index][0])] for index in range(len(list_of_centre))}

        while not clusters_centroid == old_clusters_centroid:

            # allocate points to cluster centroid
            clusters_points = {}
            for point in points:
                min_distance = np.inf
                point_in_cluster = -1

                for cluster_id in clusters_centroid:
                    centroid = clusters_centroid[cluster_id]
                    distance = np.linalg.norm(np.subtract(np.array(point), np.array(centroid)))

                    if distance < min_distance:
                        point_in_cluster = cluster_id
                        min_distance = distance

                if point_in_cluster in clusters_points:
                    clusters_points[point_in_cluster].append(point)
                else:
                    clusters_points[point_in_cluster] = [point]

            # Update cluster centroid
            old_clusters_centroid = clusters_centroid.copy()
            clusters_centroid = {}

            for cluster_id in clusters_points:
                list_of_points = np.array(clusters_points[cluster_id])
                clusters_centroid[cluster_id] = np.mean(list_of_points, axis=0).tolist()
                # print(cluster_id,clusters_centroid[cluster_id])

        return clusters_centroid  # ,clusters_centroid

    def cluster_images(self, no_of_clusters, relative_input_folder_path, relative_output_folder_path):
        base_tablename = "histogram_of_gradients"

        input_tablename = convert_folder_path_to_table_name(relative_input_folder_path, base_tablename)
        output_tablename = convert_folder_path_to_table_name(relative_output_folder_path, base_tablename)
        input_metadata_tablename = convert_folder_path_to_table_name(relative_input_folder_path)

        image_dict = self.db_conn.get_object_feature_matrix_from_db(input_tablename)
        data_matrix = image_dict['data_matrix']
        image_names = image_dict['images']
        svd_obj = SingularValueDecomposition()
        U, S, Vt = svd_obj.get_latent_semantics(data_matrix, self.no_of_dimensions)

        latent_semantic = np.matmul(U, S).tolist()

        list_of_labels = self.db_conn.get_correct_labels_for_given_images(image_names, "aspectofhand", input_metadata_tablename)
        # print(list_of_labels)
        dorsal_data_matrix = []
        palmar_data_matrix = []

        for image, label in list_of_labels:
            label_update = label.split(' ')[0]
            if label_update == DORSAL:
                dorsal_data_matrix.append(latent_semantic[image_names.index(image)])
            elif label_update == PALMAR:
                palmar_data_matrix.append(latent_semantic[image_names.index(image)])

        palmer_list_of_centers = self.k_means(np.array(palmar_data_matrix), no_of_clusters)
        dorsal_list_of_centers = self.k_means(np.array(dorsal_data_matrix), no_of_clusters)

        # print(palmer_list_of_centers, dorsal_list_of_centers)


        points_in_cluster = []

        for image_name, image_vector in zip(image_names,latent_semantic):
            min_distance = np.inf
            row = ()
            for center in dorsal_list_of_centers:
                distance = np.linalg.norm(np.subtract(np.array(image_vector), np.array(dorsal_list_of_centers[center])))
                if distance < min_distance:
                    row = (image_name, int(center) + 1)
                    min_distance = distance
            for center in palmer_list_of_centers:
                distance = np.linalg.norm(np.subtract(np.array(image_vector), np.array(palmer_list_of_centers[center])))
                if distance < min_distance:
                    row = (image_name, no_of_clusters+int(center) + 1)
                    min_distance = distance
            points_in_cluster.append(row)



        query_image_dict = self.db_conn.get_object_feature_matrix_from_db(output_tablename)
        query_data_matrix = query_image_dict['data_matrix']
        query_image_names = query_image_dict['images']
        query_latent = np.matmul(query_data_matrix,np.transpose(Vt))

        query_iterate = zip(query_image_names, query_latent.tolist())
        prediction = []

        for image, image_vector in query_iterate:
            min_distance = np.inf
            labelled = ""
            for cluster_centre in palmer_list_of_centers.values():
                distance = np.linalg.norm(np.subtract(np.array(image_vector),np.array(cluster_centre)))
                if distance < min_distance:
                    labelled = PALMAR
                    min_distance = distance
            for cluster_centre in dorsal_list_of_centers.values():
                distance = np.linalg.norm(np.subtract(np.array(image_vector),np.array(cluster_centre)))
                if distance<min_distance:
                    labelled = DORSAL
                    min_distance = distance
            prediction.append((image, labelled))

        query_list_of_labels = self.db_conn.get_correct_labels_for_given_images(query_image_names, "aspectofhand", "metadata")
        # print(query_list_of_labels)

        prediction = sorted(prediction, key=lambda k: k[0])
        query_list_of_labels = sorted(query_list_of_labels, key=lambda k: k[0])


        accuracy = 0.0
        for (image_name, predicted_label),(image2, correct_label) in zip(prediction, query_list_of_labels):
            # print(image_name, predicted_label,image2, correct_label)
            correct_label = correct_label.split(' ')[0]
            if(image_name == image2 and correct_label==predicted_label):
                accuracy += 1

        accuracy = 100*accuracy/float(len(query_image_names))
        print("The accuracy is "+ str(accuracy))

        points_in_cluster = sorted(points_in_cluster, key=lambda k: k[1])

        return points_in_cluster, prediction

if __name__ == "__main__":
    no_of_clusters = 5
    labelled_path = "/Labelled/Set1"
    unlabelled_folder_path = "/Unlabelled/Set1"

    clustering_obj = Image_Clustering()
    prediction = clustering_obj.cluster_images(no_of_clusters, labelled_path, unlabelled_folder_path)

    pprint.pprint(prediction)










