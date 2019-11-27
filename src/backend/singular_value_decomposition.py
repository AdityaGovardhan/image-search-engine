import numpy as np


# Is a temperarory class design
class SingularValueDecomposition:

    def __init__(self, num_dims=10):
        self.U = None
        self.S = None
        self.Vt = None
        self.num_dims = num_dims

    def get_latent_semantics(self, data_matrix, n_components):
        u, s, vt = self.get_svd_decomposition(data_matrix)
        u = np.array(u[:,:n_components])
        s = np.diag(s[:n_components])
        vt = np.array(vt[:n_components,:])
        return u, s, vt

    def get_svd_decomposition(self, data_matrix):
        u, s, vt = np.linalg.svd(data_matrix, full_matrices=False)
        return u, s, vt

    def get_transformed_data(self, data_matrix, k=10):
        u, s, vt = self.get_svd_decomposition(data_matrix)
        u = np.array(u[:,:k])
        s = np.diag(s[:k])
        transformed_data = np.matmul(u, s)
        return transformed_data

    def fit(self, data_matrix):
        self.U, self.S, self.Vt = np.linalg.svd(data_matrix)

    def fit_transform(self, data_matrix):
        self.U, self.S, self.Vt = np.linalg.svd(data_matrix)
        vt = self.Vt[:self.num_dims, :]
        return data_matrix.dot(vt.T)

    def transform(self, data_matrix):
        vt = self.Vt[:self.num_dims, :]
        return data_matrix.dot(vt.T)

if __name__ == "__main__":

    A = np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [11,12,13,14,15,16,17,18,19,20],
    [21,22,23,24,25,26,27,28,29,30]])

    B = np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [11,12,13,14,15,16,17,18,19,20]])

    svd_object = SingularValueDecomposition(2)
    tf_A = svd_object.fit_transform(A)
    print(tf_A)

    tf_B = svd_object.transform(B)
    print(tf_B)