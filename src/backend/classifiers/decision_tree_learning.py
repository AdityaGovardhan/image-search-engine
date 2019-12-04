import sys

sys.path.append('..')

from database_connection import DatabaseConnection
from singular_value_decomposition import SingularValueDecomposition
from utils import get_train_and_test_dataframes_from_db, get_result_metrics

from numpy.linalg import svd
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import pprint
import operator
from collections import Counter

class Node(object):
    def __init__(self, tempid_to_image_vector, tempid_to_image_label):
        self.left = None
        self.right = None
        self.tempid_to_image_vector = tempid_to_image_vector
        self.data = tempid_to_image_label
        self.dominant_label = False
        self.parent = None
        self.feature_index = self.find_best_feature()
        self.mean_value = self.find_mean()

    def find_mean(self):
        if self.feature_index is None:
            return None
        vectors_of_images = [self.tempid_to_image_vector[each][self.feature_index] for each in self.data.keys()]
        matrix_of_images = np.vstack(vectors_of_images)
        mean_of_images = np.nanmean(matrix_of_images)

        return mean_of_images

    def calculate_fsr(self, feature_index, images_of_class1, images_of_class2):
        vectors_of_class1_images = [self.tempid_to_image_vector[each] for each in images_of_class1]
        vectors_of_class2_images = [self.tempid_to_image_vector[each] for each in images_of_class2]
        matrix_of_class1_images = np.vstack(vectors_of_class1_images)
        matrix_of_class2_images = np.vstack(vectors_of_class2_images)
        mean_of_class1 = np.nanmean(matrix_of_class1_images)
        mean_of_class2 = np.nanmean(matrix_of_class2_images)
        variance_of_class1 = np.nanvar(matrix_of_class1_images)
        variance_of_class2 = np.nanvar(matrix_of_class2_images)

        try:
            fsr = ((mean_of_class1 - mean_of_class2) ** 2) / ((variance_of_class1 ** 2) + (variance_of_class2 ** 2))
        except ZeroDivisionError:
            fsr = float("-inf")

        return fsr

    def find_best_feature(self):
        labels = list(set(list(self.data.values())))
        if len(labels) < 2:
            return None
        tempids = list(self.data.keys())

        label_image_dict = {}
        for tempid in tempids:
            label = self.data[tempid]
            if label in label_image_dict.keys():
                label_image_dict[label].append(tempid)
            else:
                label_image_dict[label] = [tempid]

        fsr_list = []
        length = len(list(self.tempid_to_image_vector.values())[0])
        for feature_index in range(length):
            fsr = 0
            count = 0
            for i in range(0, len(labels) - 1):
                for j in range(i+1, len(labels)):
                    fsr += self.calculate_fsr(feature_index, label_image_dict[labels[i]], label_image_dict[labels[j]])
                    count += 1
            avg_fsr = fsr / float(count)
            fsr_list.append(avg_fsr)
        best_feature_index, value = max(enumerate(fsr_list), key=operator.itemgetter(1))

        return best_feature_index

    def check_dominancy(self, image_label_dict_values):
        label_count_dict = Counter(image_label_dict_values)
        label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        (dominant_label, dominant_count) = label_count_dict_sorted[0]
        dominancy = float(dominant_count) / len(image_label_dict_values)
        if dominancy > 0.85:
            return dominant_label
        else:
            return False

    def construct_tree(self):
        image_label_dict_values = self.data.values()
        if len(image_label_dict_values) == 0:
            return None
        dominant_label = self.check_dominancy(image_label_dict_values)
        if dominant_label:
            self.dominant_label = dominant_label
            return self
        left_tempid_label_dict = {}
        left_tempid_vector_dict = {}
        right_tempid_label_dict = {}
        right_tempid_vector_dict = {}
        for (tempid, label) in self.data.items():
            image_value_for_tag = self.tempid_to_image_vector[tempid][self.feature_index]
            if image_value_for_tag > self.mean_value:
                right_tempid_label_dict[tempid] = label
                right_tempid_vector_dict[tempid] = self.tempid_to_image_vector[tempid]
            else:
                left_tempid_label_dict[tempid] = label
                left_tempid_vector_dict[tempid] = self.tempid_to_image_vector[tempid]
        self.left = Node(left_tempid_vector_dict, left_tempid_label_dict).construct_tree()
        self.right = Node(right_tempid_vector_dict, right_tempid_label_dict).construct_tree()

        return self


class DecisionTreeLearning:
    def __init__(self, random_state=0, min_samples_leaf=10):
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.tree = None
        pass

    def fit_sklearn(self, X, y):
        self.model = DecisionTreeClassifier(random_state=self.random_state, min_samples_leaf=self.min_samples_leaf)
        print(X, y)
        self.model.fit(X, y)

    def predict_sklearn(self, u):
        y = list()

        for i in range(len(u)):
            pred_label = self.model.predict(u)
            y.append(pred_label[0])
        print(y)
        return y

    def predict(self, tree, X):
        if tree.dominant_label:
            return tree.dominant_label
        image_value_for_tag = X[tree.feature_index]
        if image_value_for_tag > tree.mean_value:
            return self.predict(tree.right, X)
        else:
            return self.predict(tree.left, X)

    def fit(self, X, y, X_test):
        tempid_to_image_vector = dict()
        for i, vector in enumerate(X): 
            tempid_to_image_vector[i] = vector
        tempid_to_image_label = dict()
        for i, label in enumerate(y): 
            tempid_to_image_label[i] = label
        node = Node(tempid_to_image_vector, tempid_to_image_label)
        
        self.tree = node.construct_tree()

        predicted_labels = list()
        for X_test_i in X_test:
            predicted_labels.append(self.predict(self.tree, X_test_i))

        return predicted_labels

# testing
if __name__ == "__main__":
    k = 15

    train_table = 'histogram_of_gradients_labelled_set1'
    train_table_metadata = 'metadata_labelled_set1'
    test_table = 'histogram_of_gradients_unlabelled_set1'

    train_df, test_df = get_train_and_test_dataframes_from_db(train_table, train_table_metadata, test_table, num_dims=k)
    X_train, y_train = np.vstack(train_df['hog_svd_descriptor'].values), train_df['label'].to_numpy(dtype=int)

    # dtl = DecisionTreeLearning()
    # dtl.fit_sklearn(X_train, y_train)
    # test_df['predicted_label'] = dtl.predict_sklearn(np.vstack(test_df['hog_svd_descriptor'].values))

    dtl2 = DecisionTreeLearning()
    test_df['predicted_label'] = dtl2.fit(X_train, y_train, np.vstack(test_df['hog_svd_descriptor'].values))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(test_df.sort_values('imagename'))

    get_result_metrics('dtl', test_df['expected_label'], test_df['predicted_label'])
