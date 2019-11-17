from database_connection import DatabaseConnection
import pprint
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans
from utils import visualize_images, transform_cm


# Is a temperarory class design
class LatentDirichletAllocation_:

    def __init__(self):
        self.dbconnection = DatabaseConnection()
        self.model = None
        self.imagenames = []

    def get_latent_semantics(self, data_matrix, n_components, feature_model):
        self.imagenames = data_matrix['images']
        image_visual_words_vector = []
        image_visual_words_vector = data_matrix['data_matrix']
        if (not feature_model == "scale_invariant_feature_transformation"):
            if (feature_model == "color_moments"):
                image_visual_words_vector = transform_cm(image_visual_words_vector)

            (x, y) = image_visual_words_vector.shape

            image_visual_words_vector = self.generate_cluster_histogram(image_visual_words_vector, x - 20)
            print("after transform ", image_visual_words_vector.shape)

        self.lda = LatentDirichletAllocation(n_components=n_components)  # To be experimented the features
        W = self.lda.fit_transform(image_visual_words_vector)  # doc-topic
        H = self.lda.components_  # topic- word
        return W, H

    def generate_cluster_histogram(self, data_matrix, num_cluster):
        kmeans = MiniBatchKMeans(n_clusters=num_cluster, random_state=0).fit(data_matrix)
        self.model = kmeans
        feature_vectors = []
        for i in range(len(data_matrix.tolist())):
            predict_kmeans = kmeans.predict(np.matrix(data_matrix[i]))
            # calculates the histogram
            hist, bin_edges = np.histogram(predict_kmeans, bins=40)

            feature_vectors.append(hist)

        # kmeans = MiniBatchKMeans(n_clusters=num_cluster, init='k-means++', init_size=30000, max_iter=300, n_init=10,
        #                          random_state=0, verbose=0)
        # descriptor_indices = kmeans.fit_predict(data_matrix)
        #
        # feature_vectors = []
        # for i, image_file in enumerate(descriptor_indices):
        #     histogram, binedges = np.histogram(descriptor_indices[i], bins=num_cluster)
        #     feature_vectors.append(histogram.tolist())

        return np.matrix(feature_vectors)

    def find_similar_images(self, w, image_vector, feature_model, m):
        image_visual_words_vector = image_vector
        if (not feature_model == "scale_invariant_feature_transformation"):
            predict_kmeans = self.model.predict(image_vector)
            # calculates the histogram
            image_visual_words_vector, bin_edges = np.histogram(predict_kmeans, bins=40)

        image_visual_words_vector = self.lda.transform(image_visual_words_vector.reshape(1, -1))
        image_visual_words_vector = np.array(image_visual_words_vector)
        ranking = {}
        for i_comp_vector in range(len(w.tolist())):
            image_name = self.imagenames[i_comp_vector]
            comp_vector_np = np.array(w.tolist()[i_comp_vector])
            ranking[image_name] = np.linalg.norm(image_visual_words_vector - comp_vector_np)

        sorted_k_values = sorted(ranking.items(), key=lambda kv: kv[1])
        # pprint.pprint(sorted_k_values)
        top_m_tuples = sorted_k_values[:m]
        return top_m_tuples

    def process_images_using_sift(self, filename):
        feature_vector, file_hist, all_image_names = get_feature_vector(40, filename)
        w, h = self.get_latent_semantics(feature_vector, 9)

        file_hist = np.asarray(file_hist)

        input_image_vector = self.lda.transform(file_hist.reshape(1, -1))

        ranking = {}
        for i_comp_vector in range(len(w.tolist())):
            image_name = all_image_names[i_comp_vector]
            comp_vector_np = np.array(w.tolist()[i_comp_vector])
            ranking[image_name] = np.linalg.norm(input_image_vector - comp_vector_np)
        sorted_k_values = sorted(ranking.items(), key=lambda kv: kv[1])
        pprint.pprint(sorted_k_values)
        top_m_tuples = sorted_k_values[:9 + 1]
        pprint.pprint(top_m_tuples)
        visualize_images(top_m_tuples, len(top_m_tuples))


if __name__ == "__main__":
    dbconnection = DatabaseConnection()
    lda_object = LatentDirichletAllocation_()
    lda_object.process_images_using_sift("Hand_0000012.jpg")
