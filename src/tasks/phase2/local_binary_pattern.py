import cv2, os
import numpy as np
from scipy.stats import moment
from skimage import feature
import pprint
import time
import concurrent.futures
from itertools import repeat
import pickle
from utils import get_image_directory

class LocalBinaryPattern:

    def __init__(self):
        pass

    def get_feature_descriptor_using_LBP(self, relative_folder_path, image_id, points=8, radius=1):
        imgsrc = relative_folder_path + '/' + image_id
        rgb_img = cv2.imread(imgsrc)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        step = 100
        window_size = [100, 100]  # [row, column] i.e. [height, width]
        img_height = gray_img.shape[0]
        img_width = gray_img.shape[1]
        res = []
        bins = 2 ** points

        block_index = 1  # denote the window number

        for y in range(0, img_height, step):
            for x in range(0, img_width, step):
                img_block = gray_img[y:y + window_size[0], x:x + window_size[1]]  # getting window of 100 x 100 size
                lbp = feature.local_binary_pattern(img_block, points, radius, method="default")
                (hist, _) = np.histogram(lbp.flatten(),
                                         bins=bins,
                                         range=(0, bins - 1))
                hist = hist / np.linalg.norm(hist, ord=1)
                res += hist.tolist()

        # return res
        return {"imageName": image_id, "features": pickle.dumps([res])}

    def get_image_vectors(self, folder_path):
        image_vectors = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(self.get_feature_descriptor_using_LBP, repeat(folder_path), os.listdir(folder_path)):
                image_vectors.append(result)

        # for image_name in os.listdir(folder_path):
        #     # print(image_name)
        #     features = self.get_feature_descriptor_using_LBP(folder_path, image_name)
        #     image_vectors.append(features)
        # pprint.pprint(image_vectors)
        return image_vectors

if __name__=="__main__":
    localBinPattern = LocalBinaryPattern()
    localBinPattern.INPUT_DATA_PATH = get_image_directory()
    start = time.perf_counter()
    vec = localBinPattern.get_image_vectors(localBinPattern.INPUT_DATA_PATH)
    fin = time.perf_counter()
    # print("ccc", vec)
    print("time = ", fin - start)
    with open('out_lbp.csv', 'wb') as output_file:
        np.savetxt(output_file, np.array(vec), fmt='%s', delimiter=',', comments='')
        output_file.close()