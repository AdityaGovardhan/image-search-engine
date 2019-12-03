import numpy as np
import os
import cv2
from sklearn.cluster import MiniBatchKMeans
import pickle
from multiprocessing import Manager, Pool


class SIFT:
    def __init__(self, INPUT_DATA_PATH):
        self.input_path = INPUT_DATA_PATH
        # print(self.input_path)
        self.x = None
        self.sift_keypoints = None
        self.kmeans = None
        self.feature_vectors = None

    def sift_implmentation(self, file):
        # print("reading file: " + file)
        image = cv2.imread(self.input_path + '/' + file, 1)
        # Convert them to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image, None)
        # append the descriptors to a list of descriptors
        self.sift_keypoints.append(descriptors)

    def read_and_clusterize(self, num_cluster):
        manager = Manager()
        self.sift_keypoints = manager.list()
        processors = os.cpu_count() - 2
        pool = Pool(processes=processors)
        pool.map(self.sift_implmentation, os.listdir(self.input_path))
        pool.close()
        pool.join()
        # print(self.sift_keypoints)
        # keypoints = np.asarray(self.sift_keypoints, dtype=np.float32)
        if self.sift_keypoints:
            keypoints = np.concatenate(self.sift_keypoints, axis=0)
            # with the descriptors detected, lets clusterize them
            print("Training kmeans")
            self.kmeans = MiniBatchKMeans(n_clusters=num_cluster, random_state=0).fit(keypoints)

    def calculate_centroids_histogram_implemtation(self, file):
        # print("reading file in calculate_centroids_histogram_implemtation: " + file)
        image = cv2.imread(self.input_path + '/' + file, 1)
        # Convert them to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image, None)
        # append the descriptors to a list of descriptors
        predict_kmeans = self.kmeans.predict(descriptors)
        # calculates the histogram
        hist, bin_edges = np.histogram(predict_kmeans, bins=150)
        self.feature_vectors.append({"imageName": file, "features": pickle.dumps(hist)})

    def calculate_centroids_histogram(self):
        manager = Manager()
        self.feature_vectors = manager.list()
        processors = os.cpu_count() - 2
        pool = Pool(processes=processors)
        pool.map(self.calculate_centroids_histogram_implemtation, os.listdir(self.input_path))
        pool.close()
        pool.join()
        return self.feature_vectors

    def get_feature_vector(self, num_clusters):
        self.read_and_clusterize(num_clusters)
        return self.calculate_centroids_histogram()

# # this function will get SIFT descriptors from training images and
# # train a k-means classifier
# def read_and_clusterize(folder_path, num_cluster):
#     sift_keypoints = []
#
#     for i, file in enumerate(os.listdir(folder_path)):
#         print("reading file: " + file)
#         image = cv2.imread(folder_path + '/' + file, 1)
#         # Convert them to grayscale
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # SIFT extraction
#         sift = cv2.xfeatures2d.SIFT_create()
#         kp, descriptors = sift.detectAndCompute(image, None)
#         # append the descriptors to a list of descriptors
#         sift_keypoints.append(descriptors)
#
#     sift_keypoints = np.asarray(sift_keypoints)
#     sift_keypoints = np.concatenate(sift_keypoints, axis=0)
#     # with the descriptors detected, lets clusterize them
#     print("Training kmeans")
#     kmeans = MiniBatchKMeans(n_clusters=num_cluster, random_state=0).fit(sift_keypoints)
#     # return the learned model
#     return kmeans
#
#
# # with the k-means model found, this code generates the feature vectors
# # by building an histogram of classified keypoints in the kmeans classifier
# def calculate_centroids_histogram(folder_path, model):
#     feature_vectors = []
#     class_vectors = []
#     all_image_names = []
#     file_hist = []
#     for i, file in enumerate(os.listdir(folder_path)):
#         # print(line)
#         # read image
#         all_image_names.append(file)
#         image = cv2.imread(folder_path + '/' + file, 1)
#         # Convert them to grayscale
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # SIFT extraction
#         sift = cv2.xfeatures2d.SIFT_create()
#         kp, descriptors = sift.detectAndCompute(image, None)
#         # classification of all descriptors in the model
#         predict_kmeans = model.predict(descriptors)
#         # calculates the histogram
#         hist, bin_edges = np.histogram(predict_kmeans, bins=150)
#         feature_vectors.append({"imageName": file, "features": pickle.dumps(hist)})
#
#     return feature_vectors
#
#
# def get_feature_vector(num_clusters, filename):
#     images_folder_path = os.getcwd()[:-4] + '/Data/images2'
#     model = read_and_clusterize(images_folder_path, num_clusters)
#     return calculate_centroids_histogram(images_folder_path, model, filename)


