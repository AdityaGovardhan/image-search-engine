from skimage import feature, io
from skimage.transform import rescale
import numpy as np
import pprint
import os
from multiprocessing import Manager, Process
import warnings
import pickle
warnings.filterwarnings("ignore")


class HistogramOfGradients:
    def hog_implementation(self, x, image_path, file):
        hand_image = io.imread(image_path)

        hand_image = rescale(hand_image, 0.1, anti_aliasing=False)

        (HoG_descriptors, hogImage) = feature.hog(hand_image, orientations=9, pixels_per_cell=(8, 8),
                                                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                                  visualize=True)
        x.append({"imageName": file, "features": pickle.dumps([HoG_descriptors.tolist()])})

    def get_image_vectors(self, folder_path):
        manager = Manager()
        x = manager.list()
        p = []
        for i, file in enumerate(os.listdir(folder_path)):
            p.append(Process(target=self.hog_implementation, args=(x, folder_path + '/' + file, file)))
            p[i].start()

        for i in range(len(p)):
            p[i].join()
        return list(x)


if __name__ == "__main__":
    hog = HistogramOfGradients()
    folder_path = "./../Data/CSE 515 Fall19 - Smaller Dataset"
    image_dict = hog.get_image_vectors(folder_path)
    pprint.pprint(image_dict)

