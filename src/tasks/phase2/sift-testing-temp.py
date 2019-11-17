import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
from utils import get_image_directory, visualize_images

class SIFT:
    def __init__(self):
        return

    def get_image_descriptors(self, image_path):
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        return descriptors.tolist()


    def get_clustered_descriptor_indices(self, scaled_all_descriptors, cluster_num):
        kmeans = MiniBatchKMeans(n_clusters=cluster_num, init='k-means++', init_size=30000, max_iter=300, n_init=10, random_state=0, verbose=0)
        descriptor_indices = kmeans.fit_predict(scaled_all_descriptors)

        return descriptor_indices


    def get_histogram(self, descriptor_indices, cluster_num):

        histogram = [0] * cluster_num
        for i in descriptor_indices:
            histogram[i] += 1

        return histogram


    def plot_variance_vs_k_pca(self, scaled_all_descriptors):

        sum_s = []
        for k in range(1, 128):

            pca = PCA(n_components=k)
            scaled_all_descriptors = pca.fit_transform(scaled_all_descriptors)
            explained_var = pca.explained_variance_ratio_
            print(k, sum(explained_var))

            sum_s.append(sum(explained_var) * 100)

        pca = PCA(n_components=None)
        scaled_all_descriptors = pca.fit_transform(scaled_all_descriptors)
        explained_var = pca.explained_variance_ratio_
        print(128, sum(explained_var))
        sum_s.append(sum(explained_var) * 100)

        plt.figure()
        plt.plot(sum_s)
        plt.xlabel("number of components")
        plt.ylabel("variance (%)")
        plt.show()


    def plot_max_eigen_vs_k_svd(self, scaled_all_descriptors):


        svd = TruncatedSVD(n_components=10)
        scaled_all_descriptors = svd.fit_transform(scaled_all_descriptors)

        plt.figure()
        plt.plot(svd.singular_values_)
        plt.xlabel("number of components")
        plt.ylabel("singular values")
        plt.show()

    def plot_score_vs_k_lda(self, scaled_all_descriptors):

        sum_s = []
        for k in range(1, 128):

            lda = LatentDirichletAllocation(n_components=k)
            scaled_all_descriptors = lda.fit_transform(scaled_all_descriptors)
            explained_var = lda.score(scaled_all_descriptors)
            # print(k, sum(explained_var))

            sum_s.append(explained_var)

        lda = LatentDirichletAllocation(n_components=None)
        scaled_all_descriptors = lda.fit_transform(scaled_all_descriptors)
        explained_var = lda.score(scaled_all_descriptors)
        # print(128, sum(explained_var))
        sum_s.append(explained_var)

        plt.figure()
        plt.plot(sum_s)
        plt.xlabel("number of components")
        plt.ylabel("variance (%)")
        plt.show()

    def plot_reconstruction_err_vs_k_nmf(self, scaled_all_descriptors):

        sum_s = []
        for k in range(1, 129):

            nmf = NMF(n_components=k)
            scaled_all_descriptors = nmf.fit_transform(scaled_all_descriptors)
            reconstruction_err = nmf.reconstruction_err_
            print(k, reconstruction_err)

            sum_s.append(reconstruction_err)

        # nmf = NMF(n_components=None)
        # all_descriptors_np = nmf.fit_transform(scaled_all_descriptors)
        # explained_var = nmf.explained_variance_ratio_
        # print(128, sum(explained_var))
        # sum_s.append(sum(explained_var) * 100)

        plt.figure()
        plt.plot(sum_s)
        plt.xlabel("number of components")
        plt.ylabel("reconstruction error")
        plt.show()


def main():

    ##################################### SIFT step ######################################
    sift = SIFT()

    image_dir = get_image_directory()
    image_files = os.listdir(image_dir)

    
    all_descriptors = list()
    image_wise_descriptor_count = dict()

    for image_file in image_files:
        image_path = image_dir + "/" + image_file
        descriptors = sift.get_image_descriptors(image_path)
        before = len(all_descriptors)
        all_descriptors.extend(descriptors)
        after = len(all_descriptors)
        image_wise_descriptor_count[image_file] = [before, after]

    # print(descriptors)

    all_descriptors_np = np.asarray(all_descriptors)

    # ##################################### PCA step ######################################

    # sc = StandardScaler()
    # all_descriptors_np = sc.fit_transform(all_descriptors_np)

    # # sift.plot_variance_vs_k_pca(all_descriptors_np)
    # # exit()

    # pca = PCA(n_components=10)

    # all_descriptors_np = pca.fit_transform(all_descriptors_np)
    # # print(all_descriptors_np.tolist())

    # ##################################### SVD step ######################################

    # sc = StandardScaler()
    # all_descriptors_np = sc.fit_transform(all_descriptors_np)

    # sift.plot_max_eigen_vs_k_svd(all_descriptors_np)
    # exit()

    # svd = TruncatedSVD(n_components=10)

    # all_descriptors_np = svd.fit_transform(all_descriptors_np)
    # # print(all_descriptors_np.tolist())

    # ##################################### LDA step ######################################

    # sc = StandardScaler()
    # all_descriptors_np = sc.fit_transform(all_descriptors_np)

    # # sift.plot_score_vs_k_lda(all_descriptors_np)
    # # exit()

    # lda = LatentDirichletAllocation(n_components=10)

    # all_descriptors_np = lda.fit_transform(all_descriptors_np)
    # # print(all_descriptors_np.tolist())

    # ##################################### NMF step ######################################

    # sc = StandardScaler()
    # all_descriptors_np = sc.fit_transform(all_descriptors_np)

    # # sift.plot_reconstruction_err_vs_k_nmf(all_descriptors_np)
    # # exit()

    # nmf = NMF(n_components=10)

    # all_descriptors_np = nmf.fit_transform(all_descriptors_np)
    # # print(all_descriptors_np.tolist())

    ################################### BOW step #########################################

    cluster_num = 40
    descriptor_indices = sift.get_clustered_descriptor_indices(all_descriptors_np, cluster_num)

    image_wise_histogram = list()
    all_histograms = list()

    for image_file in image_files:
        a, b = image_wise_descriptor_count[image_file]
        histogram = sift.get_histogram(descriptor_indices[a:b], cluster_num)
        image_wise_histogram.append({"imageName": image_file, "histogram": histogram})
        all_histograms.append(histogram)

    all_histograms_np = np.asarray(all_histograms)

    # print(all_histograms_np.shape)

    # # ##################################### PCA step ######################################

    # # sift.plot_variance_vs_k_pca(all_histograms_np)
    # # exit()

    # pca = PCA(n_components=30)

    # all_histograms_np = pca.fit_transform(all_histograms_np)
    # # print(all_histograms_np.tolist())

    # image_wise_histogram = list()
    
    # for i, image_file in enumerate(image_files):
    #     image_wise_histogram.append({"imageName": image_file, "histogram": all_histograms_np[i]})

    # ##################################### SVD step ######################################

    # # sift.plot_max_eigen_vs_k_svd(all_histograms_np)
    # # exit()

    # svd = TruncatedSVD(n_components=10)

    # all_histograms_np = svd.fit_transform(all_histograms_np)
    # # print(all_histograms_np.tolist())

    # image_wise_histogram = list()
    
    # for i, image_file in enumerate(image_files):
    #     image_wise_histogram.append({"imageName": image_file, "histogram": all_histograms_np[i]})

    # ##################################### LDA step ######################################

    # # sift.plot_score_vs_k_lda(all_histograms_np)
    # # exit()

    # lda = LatentDirichletAllocation(n_components=50)

    # all_histograms_np = lda.fit_transform(all_histograms_np)
    # # print(all_histograms_np.tolist())

    # image_wise_histogram = list()

    # for i, image_file in enumerate(image_files):
    #     image_wise_histogram.append({"imageName": image_file, "histogram": all_histograms_np[i]})

    ##################################### NMF step ######################################

    # sift.plot_reconstruction_err_vs_k_nmf(all_histograms_np)
    # exit()

    nmf = NMF(n_components=10)

    all_histograms_np = nmf.fit_transform(all_histograms_np)
    # print(all_histograms_np.tolist())

    image_wise_histogram = list()

    for i, image_file in enumerate(image_files):
        image_wise_histogram.append({"imageName": image_file, "histogram": all_histograms_np[i]})

    ################################### test step #########################################

    histogram_dict = {image_info['imageName']:image_info['histogram'] for image_info in image_wise_histogram}

    query_histogram = histogram_dict["Hand_0009445.jpg"]

    # similarity_dict = dict()

    # for image_file in image_files:
    #     db_histogram = histogram_dict[image_file]
    #     similarity = cosine_similarity(np.asarray(query_histogram).reshape(1, -1), np.asarray(db_histogram).reshape(1, -1))
    #     similarity_dict[image_file] = similarity[0][0]

    # # print
    # for w in sorted(similarity_dict, key=similarity_dict.get, reverse=True):
    #     print(w, "=", similarity_dict[w])

    # # visualize
    # sorted_k_values = sorted(similarity_dict.items(), key=lambda kv: kv[1], reverse=True)
    # visualize_images(sorted_k_values, len(sorted_k_values))

    euclidean_dict = dict()

    for image_file in image_files:
        db_histogram = histogram_dict[image_file]
        euclidean_dictance = np.linalg.norm(np.asarray(query_histogram) - np.asarray(db_histogram))
        euclidean_dict[image_file] = euclidean_dictance

    # print
    for w in sorted(euclidean_dict, key=euclidean_dict.get):
        print(w, "=", euclidean_dict[w])

    # visualize
    sorted_k_values = sorted(euclidean_dict.items(), key=lambda kv: kv[1])
    visualize_images(sorted_k_values, len(sorted_k_values))


if __name__ == "__main__":
    main()