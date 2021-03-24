import numpy as np
from sklearn.model_selection import train_test_split

def shuffle_data(features, Y):
    '''
        Receive two np.arrays and return both array shuffled with the same order
    '''
    assert features.shape[0] == features.shape[0], assert_error("Both np arrays must have the same number of rows")

    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)

    return features[indices], Y[indices]


def split_data_kfold(data, k=1, test_percent=20):
    '''
        Receive a np.arrays 'data' and return:
            1- A np array, the test set, with a percent 'test_percent' of the data  
            2- A list of k np arrays corresponding to the classification k fold data 
    '''

    n = data.shape[0]
    test_len = int(test_percent*n/100)
    k_len = int((n - test_len)/k)

    k_dataset = [[] for x in range(int(n/k_len))]


def assert_error(msg):
    return f"ASSERT ERROR: {msg}"


def split_data(X, y, test_percent=0.3, random_=None):
    '''
        Receive two np.arrays 'X' and 'y' and returns training and testing data for both
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=random_)

    return X_train, y_train, X_test, y_test
