from skimage import feature, io
from skimage.transform import rescale
import os
from multiprocessing import Manager, Pool
import warnings
import pickle
warnings.filterwarnings("ignore")


class HistogramOfGradients:
    def __init__(self, INPUT_DATA_PATH):
        self.input_path = INPUT_DATA_PATH
        self.x = None

    def hog_implementation(self, file):
        file_path = self.input_path + '/' + file
        hand_image = io.imread(file_path)

        hand_image = rescale(hand_image, 0.1, anti_aliasing=False)

        (HoG_descriptors, hogImage) = feature.hog(hand_image, orientations=9, pixels_per_cell=(8, 8),
                                                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                                                  visualize=True)
        self.x.append({"imageName": file, "features": pickle.dumps([HoG_descriptors.tolist()])})

    def get_image_vectors(self):
        manager = Manager()
        self.x = manager.list()
        processors = os.cpu_count() - 2
        pool = Pool(processes=processors)
        pool.map(self.hog_implementation, os.listdir(self.input_path))
        pool.close()
        return self.x


if __name__ == "__main__":
    hog = HistogramOfGradients()
