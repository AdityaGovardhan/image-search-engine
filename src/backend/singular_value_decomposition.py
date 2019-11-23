import sys
# sys.path.insert(0, '../src/backend')

from database_connection import DatabaseConnection
import numpy as np


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

    def get_transformed_data(self, data_matrix):
        u, s, vt = self.get_svd_decomposition(data_matrix)
        u = np.array(u)
        s = np.diag(s)
        transformed_data = np.matmul(u, s)
        return transformed_data


if __name__ == "__main__":
    svd_object = SingularValueDecomposition()
    svd_object.task6_using_svd(27)