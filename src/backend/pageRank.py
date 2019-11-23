import sys, os
sys.path.insert(0,'src/backend')

from database_connection import DatabaseConnection
from utils import get_svd_image_data_from_folder
from singular_value_decomposition import SingularValueDecomposition
import numpy as np
from pprint import pprint
from sklearn.preprocessing import normalize
from utils import (ALPHA, read_from_pickle, get_pickle_directory, save_to_pickle,
                           PICKLE_FILE_NAME)


class PageRank:
    def __init__(self):
        db_conn = DatabaseConnection()
        all_data_dict = db_conn.get_object_feature_matrix_from_db("histogram_of_gradients")
        self.data_matrix = all_data_dict['data_matrix']
        self.imageIDs = np.array(all_data_dict['images'])

    def get_image_similarity_matrix(self, k, svd_image_data):
        image_distance_matrix = np.array([])
        for each_row in svd_image_data:
            sub_matrix = np.subtract(svd_image_data, each_row)
            euc_dist = np.linalg.norm(sub_matrix, axis = 1, keepdims=True)
            if(image_distance_matrix.size == 0):
                image_distance_matrix = euc_dist
            else:
                image_distance_matrix = np.concatenate((image_distance_matrix, euc_dist), axis = 1)

        image_distance_matrix[image_distance_matrix == 0] = 1

        n = image_distance_matrix.shape[1]
        image_similarity_matrix = np.reciprocal(image_distance_matrix)

        threshold_col = np.partition(image_similarity_matrix, n-k-1, axis = 1)[:, n-k-1]
        threshold_col = threshold_col.reshape(threshold_col.shape[0], 1)
        image_similarity_matrix[image_similarity_matrix < threshold_col] = 0
        image_similarity_matrix = normalize(image_similarity_matrix, axis=1, norm='l1')

        return image_similarity_matrix

    def calculate_intermediate_page_rank_matrix(self, image_similarity_matrix):
        I = np.identity(image_similarity_matrix.shape[0])
        alpha = ALPHA
        interim = (np.linalg.inv(I - (alpha * image_similarity_matrix))) * (1 - alpha)
        return interim

    def get_page_rank_eigen_vector(self, image_similarity_matrix, S, pickle_file_name = PICKLE_FILE_NAME):
        pickle_dir = get_pickle_directory()
        interim_file_path = os.path.join(pickle_dir, pickle_file_name)
        if(os.path.exists(interim_file_path)):
            interim = read_from_pickle(pickle_file_name)
        else:
            interim = self.calculate_intermediate_page_rank_matrix(image_similarity_matrix)
            save_to_pickle(interim, pickle_file_name)
        pie = np.matmul(interim, S)
        return pie

    def get_seed_vector(self, imageIDs):
        S = np.zeros(len(self.imageIDs))
        total_user_images = len(imageIDs)

        for image in imageIDs:
            index = np.where(self.imageIDs == image)[0][0]
            S[index] = 1/total_user_images

        S = np.array(S)
        S = S.reshape(S.shape[0], 1)
        return S

    def get_top_K_images_based_on_scores(self, matrix, image_ids, K):
        similarity_scores = list(matrix.flatten())
        images, scores = list(zip(*sorted(zip(similarity_scores, image_ids), reverse=True)))
        dominant_images = list(zip(images[:K], scores[:K]))
        return dominant_images




    def get_K_dominant_images(self, k, K, imageIDs, relative_folder_path):
        S = self.get_seed_vector(imageIDs)
        svd_image_data, image_names = get_svd_image_data_from_folder(relative_folder_path)
        image_similarity_matrix = self.get_image_similarity_matrix(k, svd_image_data)
        pie = self.get_page_rank_eigen_vector(image_similarity_matrix, S, pickle_file_name="page_rank_interim_task3.pickle")
        # print("**************PIE******************")
        # pprint(pie)
        image_ids = self.imageIDs.copy()
        dominant_images = self.get_top_K_images_based_on_scores(pie, image_ids, K)
        # pprint(dominant_images)
        return dominant_images

if __name__ == "__main__":
    pg_obj = PageRank()
    pg_obj.get_K_dominant_images(5, 4, ['Hand_0011685.jpg', 'Hand_0011694.jpg','Hand_0009446.jpg'], "/Labelled/Set2")