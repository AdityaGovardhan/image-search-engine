from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from database_connection import DatabaseConnection
import numpy as np
from utils import get_euclidian_distance, plot_the_result,  get_image_directory, visualize_data_latent_semantics
import matplotlib.pyplot as plt
import pprint
from math import sqrt
import os
import time
import pickle


class NonNegativeMatrixFactorization:
    def __init__(self):
        self.data_directory = "./../Data/latent_semantics/"
    def get_latent_semantics(self, data_matrix, n_components, feature_model = "local_binary_pattern"):
        """if we initialize randomly then NMF is stucking in
                local minima as it is returning bad results"""
        start_time = time.perf_counter()
        #Use init = 'nndsvda' for LBP model for better results
        if(feature_model == "local_binary_pattern"):
            self.nmf = NMF(n_components=n_components, init='nndsvda',
                      tol=5e-3)  # Note nndsvd is behaving better with HOG, not much improvement with changing the tolerance
        else:
            self.nmf = NMF(n_components=n_components, init='nndsvd', tol=5e-3)
        print("Total training time", time.perf_counter() - start_time)
        W = self.nmf.fit_transform(data_matrix)
        H = self.nmf.components_
        return W, H
    def get_tansformed_query_image(self, query_image_vector, feature_model, dimensionality_reduction_tech):
        latent_semantic_file_path = self.data_directory + "transformed_query" + feature_model + '_' + dimensionality_reduction_tech + '.pickle'
        file = open(latent_semantic_file_path, 'wb')
        transformed_query = self.nmf.transform(query_image_vector)
        print("in fxn")
        pprint.pprint(transformed_query)
        pickle.dump(transformed_query, file)
    """
    Author: Vibhu Varshney
    """
    def test_nmf(self, feature_model, K, m, query_image):
        """
        :param feature_model: The Feature model to be used for feature extraction
        :param K: Number of latent features to use
        :param m: Number of most similar images to be outputted
        :return:
        """
        db_conn = DatabaseConnection()
        all_data_with_images = db_conn.get_object_feature_matrix_from_db(feature_model)
        imageNames = all_data_with_images['images']
        feat_matr = all_data_with_images['data_matrix']
        print(feat_matr.shape)
        #Applying NMF to get the decomposed matrices
        W, H = self.get_latent_semantics(feat_matr, K)
        print(W.shape)
        print(H.shape)
        visualize_data_latent_semantics(H, imageNames, K, 5, "Latent Semantics in terms of Features", isData = False,
                                        data_in_latent_space = np.dot(feat_matr, np.transpose(H)))


if __name__ == "__main__":
    #By Vibhu
    nmf = NonNegativeMatrixFactorization()
    nmf.test_nmf("local_binary_pattern", 20, 10, 'Hand_0000012.jpg')
    # nmf.test_nmf("histogram_of_gradients", 20, 10, 'Hand_0000039.jpg')
