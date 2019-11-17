from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from database_connection import DatabaseConnection
import numpy as np

from utils import get_euclidian_distance, plot_the_result, get_image_directory, visualize_data_latent_semantics, sift_euclidean_comparison, transform_cm, plot_scree_test

import matplotlib.pyplot as plt
import pprint
from math import sqrt
import cv2
import os

# from scale_invariant_feature_transformation import Sift

class PrincipleComponentAnalysis:
    def __init__(self):
        pass

    # get latent semantics of data matrix formed by database images
    def get_latent_semantics(self, data_matrix, n_components):
        # standard_scalar = StandardScaler()
        # sc_matrix = standard_scalar.fit_transform(data_matrix)
        self.pca = PCA(n_components=n_components)
        u = self.pca.fit_transform(data_matrix)
        s = self.pca.singular_values_
        vt = self.pca.components_

        return u, s, vt
    
    # By Aditya
    # get image vector in transformed space formed using latent semantics
    # TODO: to be shifted to Utils? According to other DR algos?
    def transform_query_image(self, image_vector, data_matrix, n_components):

        u, s, vt = self.get_latent_semantics(data_matrix, n_components)

        transformed_image = np.dot(image_vector, np.transpose(vt))

        return transformed_image

# TODO: clean up testing!
if __name__ == "__main__":

    # By Ketan
    database_connection = DatabaseConnection()
    all_data = database_connection.get_object_feature_matrix_from_db(tablename='color_moments')
    # u, s, vt = pca.get_latent_semantics(data_matrix, 5)
    # print(u, s, vt)
    data_matrix = all_data['data_matrix']
    # print()
    transformed_cm = transform_cm(data_matrix)
    pprint.pprint(transformed_cm)
    # # By Aditya
    # db = DatabaseConnection()

    # data_matrix = db.get_object_feature_matrix_from_db(tablename='scale_invariant_feature_transformation')
    # image_vector = db.get_feature_data_for_image(tablename='scale_invariant_feature_transformation', imageName='Hand_0011682.jpg')

    # print("original data matrix shape =", data_matrix["data_matrix"].shape)
    # print("original query image shape =", image_vector.shape)

    # pca = PrincipleComponentAnalysis()

    # u,s,vt = pca.get_latent_semantics(data_matrix["data_matrix"], 10)
    # t_image = pca.transform_query_image(image_vector, data_matrix["data_matrix"], 10)

    # print("================================================================================================")

    # i = data_matrix["images"].index('Hand_0011682.jpg')


    # t_db = np.dot(data_matrix["data_matrix"], np.transpose(vt))

    # d = t_db[i * 32 : i * 32 + 32]

    # print("transformed data matrix shape =", t_db.shape)
    # print("transformed query image shape =",t_image.shape)

    # sift_euclidean_comparison(t_db, t_image, data_matrix["images"])
