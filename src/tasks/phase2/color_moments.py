import cv2, os
import numpy as np
from scipy.stats import moment
from skimage import feature
import pprint
import concurrent.futures
import time
from itertools import repeat
import pickle
from utils import get_image_directory


class ColorMoments:

    def __init__(self):
        pass

    def get_feature_descriptor_using_CM(self, relative_folder_path, image_id):
        imgsrc = relative_folder_path + '/' + image_id
        rgb_img = cv2.imread(imgsrc)
        yuv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YUV)
        step = 100
        window_size = [100, 100]  # [row, column] i.e. [height, width]
        img_height = yuv_img.shape[0]
        img_width = yuv_img.shape[1]
        res = []
        block_index = 1  # denote the window number

        for y in range(0, img_height, step):
            for x in range(0, img_width, step):
                img_block = yuv_img[y:y + window_size[0], x:x + window_size[1], :]  # getting window of 100 x 100 size
                y_b, u_b, v_b = cv2.split(img_block)
                y_mean = np.mean(y_b)
                u_mean = np.mean(u_b)
                v_mean = np.mean(v_b)
                y_std = np.std(y_b)
                u_std = np.std(u_b)
                v_std = np.std(v_b)
                y_skew = np.cbrt(moment(y_b.flatten(), moment=3))
                u_skew = np.cbrt(moment(u_b.flatten(), moment=3))
                v_skew = np.cbrt(moment(v_b.flatten(), moment=3))
                res += [y_mean, u_mean, v_mean, y_std, u_std, v_std, y_skew, u_skew, v_skew]
                # Need to work on dimensions of the res data marix
        return {"imageName": image_id, "features": pickle.dumps([res])}

    def get_image_vectors(self, folder_path):
        image_vectors = []
        #
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(self.get_feature_descriptor_using_CM, repeat(folder_path), os.listdir(folder_path)):
                image_vectors.append(result)
        # return image_vectors
        # for image_name in os.listdir(folder_path):
        #     print(image_name)
        #     image_vectors.append(self.get_feature_descriptor_using_CM(folder_path, image_name))
        # pprint.pprint(image_vectors)
        return image_vectors


if __name__=="__main__":
    color_moments = ColorMoments()
    color_moments.INPUT_DATA_PATH = get_image_directory()
    start = time.perf_counter()
    vec = color_moments.get_image_vectors(color_moments.INPUT_DATA_PATH)
    fin = time.perf_counter()
    print("ccc", vec.shape)
    # print("time took = ", fin - start)
    # with open('out_cm.csv', 'wb') as output_file:
    #     np.savetxt(output_file, np.array(vec), fmt='%s', delimiter=',', comments='')
    #     output_file.close()
