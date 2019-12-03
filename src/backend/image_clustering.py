import sys, os
sys.path.insert(0,'src/backend')

from database_connection import DatabaseConnection
from utils import get_svd_image_data_from_folder, get_image_names_in_a_folder, convert_folder_path_to_table_name
from singular_value_decomposition import SingularValueDecomposition
import numpy as np
import pprint
from scipy.cluster.vq import vq, kmeans, whiten

class Image_Clustering:
    def __init__(self):
        self.db_conn = DatabaseConnection()
        self.no_of_dimensions = 20

    def k_means(self, points, no_of_centres):

        list_of_centre = (points.shape[0] * np.random.rand(no_of_centres,1)).tolist()
        points = points.tolist()
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

    def cluster_images(self, no_of_clusters, relative_folder_path):
        tablename = convert_folder_path_to_table_name(relative_folder_path, "histogram_of_gradients")
        image_dict = self.db_conn.get_object_feature_matrix_from_db(tablename)
        image_names = image_dict['images']
        data_matrix = image_dict['data_matrix']
        svd_obj = SingularValueDecomposition()
        U, S, Vt = svd_obj.get_latent_semantics(data_matrix, self.no_of_dimensions)

        latent_semantic = np.matmul(U, S)
        normalized_data = whiten(latent_semantic)
        clustered_data, error = kmeans(normalized_data, no_of_clusters)

        list_of_images = latent_semantic.tolist()
        list_of_centers = clustered_data.tolist()

        points_in_cluster = []
        sorted_image_list = sorted(zip(image_names, list_of_images), key=lambda k: k[0])
        for image_name, image_vector in sorted_image_list:
            min_distance = np.inf
            row = ()
            for center in list_of_centers:
                distance = np.linalg.norm(np.subtract(np.array(image_vector),np.array(center)))
                if distance < min_distance:
                    row = (image_name, int(list_of_centers.index(center))+1)
                    min_distance = distance
            points_in_cluster.append(row)


        prediction = sorted(points_in_cluster, key=lambda k: k[1])

        pprint.pprint(prediction)

        return prediction

if __name__ == "__main__":
    no_of_clusters = 12
    relative_folder_path = "/Labelled/Set1"

    clustering_obj = Image_Clustering()
    prediction = clustering_obj.cluster_images(no_of_clusters, relative_folder_path)









