from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from database_connection import DatabaseConnection
import numpy as np

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

    pass
