import sys
sys.path.append('..')

from database_connection import DatabaseConnection
from singular_value_decomposition import SingularValueDecomposition

from numpy.linalg import svd
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import pprint

class DecisionTreeLearning:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, u):
        pass

# testing
if __name__ == "__main__":

    k = 15

    train_table = 'histogram_of_gradients_labelled_set1'
    train_table_metadata = 'metadata_labelled_set1'

    test_table = 'histogram_of_gradients_unlabelled_set1'
    test_table_metadata = 'metadata_unlabelled_set1'

    label_map = {"dorsal": 0, "palmar": 1}

    db = DatabaseConnection()
    train_dataset = db.get_object_feature_matrix_from_db(train_table)
    test_dataset = db.get_object_feature_matrix_from_db(test_table)

    train_data = train_dataset['data_matrix']
    test_data = test_dataset['data_matrix']

    svd = SingularValueDecomposition(k)
    tf_train_data = svd.fit_transform(train_data)
    tf_test_data = svd.transform(test_data)

    train_labels_map = dict(db.get_correct_labels_for_given_images(train_dataset['images'], 'aspectOfHand', train_table_metadata))

    col_names = ['imagename', 'hog_svd_descriptor', 'label']
    train_df = pd.DataFrame(columns=col_names)

    for i, image in enumerate(train_dataset['images']):
        temp = train_labels_map[image]
        label = temp.split(' ')[0]

        train_df.loc[len(train_df)] = [image, tf_train_data[i], label_map[label]]

    print(train_df)
    print("======================================================================")

    X_train, y_train = tf_train_data, train_df['label'].to_numpy(dtype=int)

    model = DecisionTreeClassifier(random_state=0, min_samples_leaf=10)
    model.fit(X_train, y_train)
    pprint.pprint(plot_tree(model))


    test_df = pd.DataFrame(columns=col_names)

    for i, image in enumerate(test_dataset['images']):
        label = model.predict(tf_test_data[i].reshape(1, -1))

        test_df.loc[len(test_df)] = [image, tf_test_data[i], label]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(test_df.sort_values('imagename'))

    print("======================================================================")

    train_df_2 = pd.DataFrame(columns=col_names)

    for i, image in enumerate(train_dataset['images']):
        label = model.predict(tf_train_data[i].reshape(1, -1))

        train_df_2.loc[len(train_df_2)] = [image, tf_train_data[i], label]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(train_df_2.sort_values('imagename'))