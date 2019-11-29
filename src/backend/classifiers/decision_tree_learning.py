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

class DecisionTreeLearning:
    def __init__(self, random_state=0, min_samples_leaf=10):
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        pass

    def fit(self, X, y):
        self.model = DecisionTreeClassifier(random_state=self.random_state, min_samples_leaf=self.min_samples_leaf)
        self.model.fit(X, y)

    def predict(self, X):

        y = list()

        for i in range(len(X)):
            pred_label = self.model.predict(X)
            y.append(pred_label[0])

        return y

# testing
if __name__ == "__main__":

    k = 15

    train_table = 'histogram_of_gradients_labelled_set1'
    train_table_metadata = 'metadata_labelled_set1'
    test_table = 'histogram_of_gradients_unlabelled_set1'

    train_df, test_df = get_train_and_test_dataframes_from_db(train_table, train_table_metadata, test_table, num_dims=k)
    X_train, y_train = np.vstack(train_df['hog_svd_descriptor'].values), train_df['label'].to_numpy(dtype=int)

    dtl = DecisionTreeLearning()
    dtl.fit(X_train, y_train)
    test_df['predicted_label'] = dtl.predict(np.vstack(test_df['hog_svd_descriptor'].values))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(test_df.sort_values('imagename'))

    get_result_metrics('dtl', test_df['expected_label'], test_df['predicted_label'])