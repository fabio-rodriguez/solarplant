import keras
import numpy as np
import tensorflow as tf

#from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model


def split_data(X, y, test_percent=0.3, random_=None):
    '''
        Receive two np.arrays 'X' and 'y' and returns training and testing data for both
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=random_)

    return X_train, y_train, X_test, y_test


def test_model(X, y, path):

    model = keras.models.load_model(path)

    y_pred = model.predict_classes(X)

    print_confusion_matrix(y, y_pred)

    plot_model(model, to_file='model.png')


def plot_history(record, path=""):

    print(record)
    print(record.history.keys())

    plt.plot(record.history['accuracy'])
    plt.plot(record.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(path+"accuracy_hist.png")
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(path+"loss_hist.png")
    plt.close()


def print_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print_table_in_cmd(["1","0"], [
        [tp, fp],
        [fn, tn],
    ])

    print()
    print(f"Accuraccy: {round((tp + tn)/(tn + fp + fn + tp),2)}")
    print(f"Sensitivity: {round(tp/(tp + fp),2)}")
    print(f"Specificity: {round(tn/(tn + fn),2)}")
    print()


def print_table_in_cmd(headers, data):

    row_format ="{:>7}" * (len(headers) + 1)
    print(row_format.format("", *headers))
    for team, row in zip(headers, data):
        print(row_format.format(team, *row))
