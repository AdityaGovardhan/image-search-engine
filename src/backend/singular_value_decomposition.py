import sys
sys.path.insert(0, '../backend')

from database_connection import DatabaseConnection
from histogram_of_gradients import HistogramOfGradients
# from . import database_connection, histogram_of_gradients
import pprint
import numpy as np
from utils import plot_scree_test
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


class SingularValueDecomposition:

    def __init__(self):
        self.dbconnection = DatabaseConnection()
        pass

    def get_latent_semantics(self, data_matrix, n_components):
        u, s, vt = self.get_svd_decomposition(data_matrix)
        u = np.array(u[:,:n_components])
        s = np.diag(s[:n_components])
        vt = np.array(vt[:n_components,:])
        return u, s, vt

    def get_svd_decomposition(self, data_matrix):
        u, s, vt = np.linalg.svd(data_matrix, full_matrices=False)
        return u, s, vt


if __name__ == "__main__":
    svd_object = SingularValueDecomposition()
    svd_object.task6_using_svd(27)