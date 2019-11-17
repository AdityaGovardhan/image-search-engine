import numpy as np
import csv
from sklearn.decomposition import PCA
from phase1.vibhu.task1 import calculate_feature_descriptor_using_LBP, calculate_feature_descriptor_using_CM
import matplotlib.pyplot as plt
import cv2
import numpy.linalg.svd as svd


class PCA_test:

    def __init__(self):
        cm_features = self.load_csv("vibhu/out_task1.csv")
        self.perform_pca(cm_features)


    def load_csv(self, path):
        # with open(path, "r") as f:
        #     # cm_feature = csv.reader(f)
        #     # next(cm_feature)
        #     # data = list(cm_feature)
        #     data_np = np.genfromtxt(f, dtype=float, delimiter=',', names=True)
        # data_np = calculate_feature_descriptor_using_CM("../input/Hand_0000012.jpg")
        data_np = calculate_feature_descriptor_using_LBP("../input/Hand_0000012.jpg")
        # print(cm_feature)
        # print(data_np)
        # print(list(data_np))
        return data_np

    def perform_pca(self, vector):
        print(vector.shape)
        # vector = np.reshape(vector, (192,16,16))
        # print(vector)
        pca = PCA(0.95)

        pca.fit(vector)
        lat_fea = pca.transform(vector)
        # print("la", lat_fea)
        print(lat_fea.shape)


        # w = 100
        # h = 100
        # fig = plt.figure(figsize=(1,1))
        # columns = 16
        # rows = 12
        # for i in range(1, columns * rows+1):
        #     # img = np.random.randint(10, size=(h, w))
        #     fig.add_subplot(rows, columns, i)
        #     # rgb_img = cv2.cvtColor(vector[i-1], cv2.COLOR_RGB2GRAY)
        #     plt.imshow(lat_fea[i-1], cmap='gray')
        # plt.show()
        # lat_fea

if __name__=="__main__":
    pca = PCA_test()