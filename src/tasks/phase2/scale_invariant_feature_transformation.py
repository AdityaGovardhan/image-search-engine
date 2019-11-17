import os
from operator import itemgetter

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Manager, Process
import pprint
import pickle
from utils import get_image_directory, visualize_images
from principle_component_analysis import PrincipleComponentAnalysis
from singular_value_decomposition import SingularValueDecomposition
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

np.set_printoptions(suppress=True)

class Sift:

    def __init__(self):
        return

    # def get_image_descriptors(self, x, file_path, file_name): # MULTIPROCESSING UNCOMMENT
    def get_image_descriptors(self, file_path):

        img = cv.imread(file_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        return descriptors.tolist()


    def perform_clustering(self, scaled_all_descriptors, cluster_num):
        
        kmeans = MiniBatchKMeans(n_clusters=cluster_num, init='k-means++', init_size=30000, max_iter=300, n_init=10,
                                 random_state=0, verbose=0)
        descriptor_indices = kmeans.fit_predict(scaled_all_descriptors)

        centroid_vectors = kmeans.cluster_centers_

        return descriptor_indices, centroid_vectors

    # def get_histogram_from_descriptor_indices(self, descriptor_indices, cluster_num):

    #     histogram = [0] * cluster_num
    #     for i in descriptor_indices:
    #         histogram[i] += 1

    #     return histogram

    def get_histogram_from_cluster_vectors(self, descriptors, cluster_vectors):
        
        return

    def get_image_vectors(self, folder_path):

        # # MULTIPROCESSING UNCOMMENT
        # files = os.listdir(folder_path)

        # manager = Manager()
        # x = manager.list()
        # p = []

        # for i, file_name in enumerate(files):
        #     file_path = folder_path + '/' + file_name
        #     p.append(Process(target=self.get_image_descriptors, args=(x, file_path, file_name)))
        #     p[i].start()

        # for i in range(len(p)):
        #     p[i].join()

        # return list(x)

        image_files = os.listdir(folder_path)

        all_descriptors = list()
        image_wise_descriptor_count = dict()

        for image_file in image_files:
            image_path = folder_path + "/" + image_file
            descriptors = self.get_image_descriptors(image_path)
            print(image_file, "has", len(descriptors), "descriptors")
            before = len(all_descriptors)
            all_descriptors.extend(descriptors)
            after = len(all_descriptors)
            image_wise_descriptor_count[image_file] = [before, after]

        all_descriptors_np = np.asarray(all_descriptors)

        cluster_num = 150
        descriptor_indices, centroid_vectors = self.perform_clustering(all_descriptors_np, cluster_num)

        image_wise_histogram = list()

        for image_file in image_files:
            a, b = image_wise_descriptor_count[image_file]
            # histogram = self.get_histogram_from_descriptor_indices(descriptor_indices[a:b], cluster_num)
            histogram, binedges = np.histogram(descriptor_indices[a:b], bins=cluster_num)
            image_wise_histogram.append({"imageName": image_file, "features": pickle.dumps([histogram.tolist()])})
            # image_wise_histogram.append({"imageName": image_file, "features": [histogram.tolist()]})

        return image_wise_histogram

class Sift_BoW:

    def __init__(self):
        return

    # get descriptors of an image
    def sift_implementation(self, file_path):

        img = cv.imread(file_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        return descriptors.tolist()

    def get_svd_features_per_image(self, features_per_image, feature_dataset_array):

        # count of descriptors per image
        dcount_per_image = [{"imageName": image_info["imageName"], "dcount": len(image_info["features"])} for image_info in features_per_image]
        # print(dcount_per_image)

        svd = SingularValueDecomposition()

        # feature_dataset_array = normalize(feature_dataset_array)
        u,s,vt = svd.get_latent_semantics(feature_dataset_array, 10)
        feature_dataset_array = np.dot(feature_dataset_array, np.transpose(vt))

        i = 0
        features_per_image = []

        for image_info in dcount_per_image:
            j = i + image_info["dcount"]
            features_per_image.append({"imageName": image_info["imageName"], "features": feature_dataset_array[i:j]})
            i = j

        return features_per_image, feature_dataset_array

    def get_pca_features_per_image(self, features_per_image, feature_dataset_array):

        # count of descriptors per image
        dcount_per_image = [{"imageName": image_info["imageName"], "dcount": len(image_info["features"])} for image_info in features_per_image]
        # print(dcount_per_image)

        pca = PrincipleComponentAnalysis()

        # feature_dataset_array = normalize(feature_dataset_array)
        u,s,vt = pca.get_latent_semantics(feature_dataset_array, 20)
        feature_dataset_array = np.dot(feature_dataset_array, np.transpose(vt))

        i = 0
        features_per_image = []

        for image_info in dcount_per_image:
            j = i + image_info["dcount"]
            features_per_image.append({"imageName": image_info["imageName"], "features": feature_dataset_array[i:j]})
            i = j

        return features_per_image, feature_dataset_array


    # get bow histogram for list of images
    def get_bow_histogram_per_image(self, features_per_image, feature_dataset_array):

        # average number of descriptors to get number of cluster centers
        # TODO: improve this, use WCSS
        # reference: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
        k = feature_dataset_array.shape[0] // len(features_per_image)
        # print("average number of features are", k)

        # initialize kmeans cluster
        kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', init_size=3*k, max_iter=300, n_init=10, random_state=0, verbose=0)

        # fit all descriptors to get n cluster centers
        kmeans = kmeans.fit(feature_dataset_array)

        # list of cluster centers is called codebook in literature
        # will have to store this in DB
        # TODO: ask prof if updated every time new image comes in real life appln
        codebook = kmeans.cluster_centers_
        # print(codebook.tolist())
        # print("==========================================================")

        bow_histogram_per_image = []
        bow_histogram_dataset = []

        for image_info in features_per_image:

            descriptors = image_info["features"]

            # from descriptors to indices of codebook
            # that index is assigned to a descriptor which is closest to it
            # TODO: computationally better to use labels_?
            codebook_indices_per_descriptor = kmeans.predict(descriptors).tolist()

            # histogram logic
            bow_histogram = [0] * k
            for index in codebook_indices_per_descriptor:
                bow_histogram[index] += 1 / len(descriptors)

            # print(image_info["imageName"], bow_histogram)

            bow_histogram_per_image.append({"imageName": image_info["imageName"], "features": [bow_histogram]})
            bow_histogram_dataset.append(bow_histogram)

        bow_histogram_dataset_array = np.array(bow_histogram_dataset)

        return bow_histogram_per_image, bow_histogram_dataset_array

    # get sift bow vectors for images in a folder
    def get_features_per_image(self, folder_path):

        files = os.listdir(folder_path)

        # list of descriptors mapped to each image
        features_per_image = []

        # list of all descriptors across all images
        feature_dataset = []

        # count of descriptors per image
        dcount_per_image = []

        # get descriptors per image
        for i, file_name in enumerate(files):
            file_path = folder_path + '/' + file_name
            descriptors = self.sift_implementation(file_path)

            features_per_image.append({"imageName": file_name, "features": descriptors})
            dcount_per_image.append({"imageName": file_name, "dcount": len(descriptors)})
            print(file_name, "has", len(descriptors), "features")

            feature_dataset.extend(descriptors)

        feature_dataset_array = np.array(feature_dataset)
        # feature_dataset_array = normalize(feature_dataset_array)

        # i = 0
        # features_per_image = []

        # for image_info in dcount_per_image:
        #     j = i + image_info["dcount"]
        #     features_per_image.append({"imageName": image_info["imageName"], "features": feature_dataset_array[i:j].tolist()})
        #     i = j

        return features_per_image, feature_dataset_array

    def plot_variance_vs_k(self, folder_path, dr_algorithm, mink=1, maxk=128):

        files = os.listdir(folder_path)

        # list of all descriptors across all images
        feature_dataset = []

        # list of descriptors mapped to each image
        features_per_image = []

        # count of descriptors per image
        dcount_per_image = []

        # get descriptors per image
        for i, file_name in enumerate(files):
            file_path = folder_path + '/' + file_name
            descriptors = self.sift_implementation(file_path)
            features_per_image.append({"imageName": file_name, "features": descriptors})
            dcount_per_image.append({"imageName": file_name, "dcount": len(descriptors)})
            print(file_name, "has", len(descriptors), "features")

            feature_dataset.extend(descriptors)

        feature_dataset_array = np.array(feature_dataset)

        # for image in features_per_image:
        #     print(image["imageName"])
        #     print(image["features"])
        #     print("===============================================================")

        sum_s = []
        for k in range(mink, maxk + 1):

            if dr_algorithm == "pca":
                pca = PrincipleComponentAnalysis()

                u,s,vt,ratio = pca.get_latent_semantics(feature_dataset_array, k)
                t_db = np.dot(feature_dataset_array, np.transpose(vt))

                sum_s.append(sum(ratio) * 100)


        plt.figure()
        plt.plot(sum_s)
        plt.xlabel("number of components")
        plt.ylabel("variance (%)")
        plt.show()


# testing
if __name__ == "__main__":

    # SIFT
    # ====

    sift = Sift()
    output = sift.get_image_vectors(get_image_directory())
    print(output)

    # # SIFT BOW
    # # ========

    # sift_bow = Sift_BoW()

    # features_per_image, feature_dataset_array = sift_bow.get_features_per_image(get_image_directory())
    # # print(feature_dataset_array.shape)

    # features_per_image, feature_dataset_array = sift_bow.get_bow_histogram_per_image(features_per_image, feature_dataset_array)
    # # print(feature_dataset_array.shape)

    # # features_per_image, feature_dataset_array = sift_bow.get_pca_features_per_image(features_per_image, feature_dataset_array)
    # # features_per_image, feature_dataset_array = sift_bow.get_svd_features_per_image(features_per_image, feature_dataset_array)
    # # print(feature_dataset_array.shape)

    

    

    # # for image in features_per_image:
    # #     print(image["imageName"])
    # #     print(image["features"])
    # #     print("===============================================================")

    # # exit()

    # # (list of dict) -> dict conversion
    # features_dict = {image_info['imageName']:image_info['features'] for image_info in features_per_image}

    # # test image
    # query_histogram = features_dict["Hand_0009445.jpg"]

    # dist_dict = {}

    # # dict with key as image name and value as its distance from query image
    # dist_dict = {image_info['imageName']:np.linalg.norm(np.array(query_histogram) - np.array(image_info['features'])) for image_info in features_per_image}

    # # print
    # for w in sorted(dist_dict, key=dist_dict.get):
    #     print(w, "=", dist_dict[w])

    # # visualize
    # sorted_k_values = sorted(dist_dict.items(), key=lambda kv: kv[1])
    # visualize_images(sorted_k_values, len(sorted_k_values))