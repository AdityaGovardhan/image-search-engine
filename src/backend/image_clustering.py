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
        for image_name, image_vector in zip(image_names, list_of_images):
            min_distance = np.inf
            row = ()
            for center in list_of_centers:
                distance = np.linalg.norm(np.subtract(np.array(image_vector),np.array(center)))
                if distance < min_distance:
                    row = (image_name, int(list_of_centers.index(center))+1)
                    min_distance = distance
            points_in_cluster.append(row)

        prediction = sorted(points_in_cluster, key=lambda k: k[1])

        #pprint.pprint(prediction)

        return prediction

if __name__ == "__main__":
    no_of_clusters = 12
    relative_folder_path = "/Labelled/Set1"

    clustering_obj = Image_Clustering()
    prediction = clustering_obj.cluster_images(no_of_clusters, relative_folder_path)
    pprint.pprint(prediction)








