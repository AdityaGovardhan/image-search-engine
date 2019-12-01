import cv2, os
import numpy as np
from skimage import feature
import pickle
from multiprocessing import Manager, Pool


class LocalBinaryPattern:

    def __init__(self, INPUT_DATA_PATH):
        self.input_path = INPUT_DATA_PATH
        self.x = None
        self.points = 8
        self.radius = 1

    def get_feature_descriptor_using_LBP(self, file):
        imgsrc = self.input_path + '/' + file
        rgb_img = cv2.imread(imgsrc)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        step = 100
        window_size = [100, 100]  # [row, column] i.e. [height, width]
        img_height = gray_img.shape[0]
        img_width = gray_img.shape[1]
        res = []
        bins = 2 ** self.points

        for y in range(0, img_height, step):
            for x in range(0, img_width, step):
                img_block = gray_img[y:y + window_size[0], x:x + window_size[1]]  # getting window of 100 x 100 size
                lbp = feature.local_binary_pattern(img_block, self.points, self.radius, method="default")
                (hist, _) = np.histogram(lbp.flatten(),
                                         bins=bins,
                                         range=(0, bins - 1))
                hist = hist / np.linalg.norm(hist, ord=1)
                res += hist.tolist()

        self.x.append({"imageName": file, "features": pickle.dumps([res])})

    def get_image_vectors(self):
        manager = Manager()
        self.x = manager.list()
        processors = os.cpu_count() - 2
        pool = Pool(processes=processors)

        pool.map(self.get_feature_descriptor_using_LBP, os.listdir(self.input_path))
        pool.close()
        return self.x


if __name__ == "__main__":
    pass
