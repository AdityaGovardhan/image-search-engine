from database_connection import DatabaseConnection
from histogram_of_gradients import HistogramOfGradients
# from . import database_connection, histogram_of_gradients
import pprint
import numpy as np
from utils import plot_scree_test
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


# Is a temperarory class design
class SingularValueDecomposition:

    def __init__(self):
        self.dbconnection = DatabaseConnection()
        pass

    def get_latent_semantics(self, data_matrix, n_components):
        u, s, vt = np.linalg.svd(data_matrix,full_matrices=False)
        u = np.matrix(u[:,:n_components])
        s = np.diag(s[:n_components])
        vt = np.matrix(vt[:n_components,:])
        # plot_scree_test(np.diagonal(s))
        return u, s, vt


if __name__ == "__main__":
    svd_object = SingularValueDecomposition()
    svd_object.task6_using_svd(27)