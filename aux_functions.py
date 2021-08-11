import keras
import numpy as np
import statistics
import tensorflow as tf
import warnings

#from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.backend import reverse



def mean_sd_bycolumn(data):
    '''
    Calcular la media y la desviacion estandar de cada feature de los datos
    '''

    n = len(data[0])
    means = np.zeros(n)
    sds = np.zeros(n)

    for i in range(n):
        columni = [column[i] for column in data]
        means[i] = sum(columni)/len(columni)
        sds[i] = statistics.pstdev(columni)

    return means, sds


def split_data(X, y, test_percent=0.3, random_=None):
    '''
        Receive two np.arrays 'X' and 'y' and returns training and testing data for both
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=random_)

    return X_train, y_train, X_test, y_test


def test_model_from_path(X, y, path, reg=None, mean_y = None, std_y = None, Time = None, picture_id = ""):
    '''
        Given X and y from the test set creates the confusion matrix from a model readed from path 
    '''

    model = keras.models.load_model(path)
    if reg:
        y_pred = model.predict(X)
        if mean_y and std_y:
            y = [y*std_y+mean_y for yi in y]
            y_pred = y_pred*std_y+mean_y

        try: 
            plt.plot(Time, y, label="real values", color="blue")
            plt.plot(Time, y_pred, label="predicted values", color="red")
            plt.legend()
            plt.savefig(path+f"predictions {picture_id}.jpg")
            plt.close()
        except:
            pass

        mean = mean_absolute_error(y, y_pred=y_pred)
        
        y_pred = [x[0] for x in y_pred]
        return mean, np.sqrt(sum((y-y_pred-np.full(len(y), mean))**2)/len(y)) 
        #print(list(zip(y, list(y_pred[:, 0]))))
    else:
        y_pred = model.predict_classes(X)
        print_confusion_matrix(y, y_pred)



def test_model(X, y, model):
    '''
        Given X and y from the test set creates the confusion matrix from a model readed from path 
    '''

    y_pred = model.predict(X)
    print_confusion_matrix(y, y_pred)


def plot_history(record, path="", reg=None):
    '''
        Given a model's record, plot the accuracy and loss graphics per epoch
    '''

    if reg:
        plt.plot([x for i, x in enumerate(record.history['mean_absolute_error']) if i%10==0])
        plt.plot([x for i, x in enumerate(record.history['val_mean_absolute_error']) if i%10==0])
        plt.title('model mean absolute error')
        plt.ylabel('mean absolute error')
    else:        
        plt.plot([x for i, x in enumerate(record.history['accuracy']) if i%10==0])
        plt.plot([x for i, x in enumerate(record.history['val_accuracy']) if i%10==0])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
    
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(path+"accuracy_hist.png")
    plt.close()

    plt.plot([x for i, x in enumerate(record.history['loss']) if i%10==0])
    plt.plot([x for i, x in enumerate(record.history['val_loss']) if i%10==0])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(path+"loss_hist.png")
    plt.close()


def print_confusion_matrix(y_true, y_pred):
    '''
        Prints confusion matrix from real and predicted data sets
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print_table_in_cmd(["1","0"], [
        [tp, fp],
        [fn, tn],
    ])

    print()
    print(f"Accuraccy: {round((tp + tn)/(tn + fp + fn + tp),5)}")
    print(f"Sensitivity: {round(tp/(tp + fp),5)}")
    print(f"Specificity: {round(tn/(tn + fn),5)}")
    print()


def print_table_in_cmd(headers, data):
    '''
        Auxiliar method for printing confusion matrix on console
    '''

    row_format ="{:>7}" * (len(headers) + 1)
    print(row_format.format("", *headers))
    for team, row in zip(headers, data):
        print(row_format.format(team, *row))


def save_model_info(model, hist, path):
    '''
        Saves the model and the accuracy and loss info in the path direction 
    '''

    model.save(path)
    plot_history(hist, path)


def study_plot(X,y):

    ## Respecto al tiempo
    #good = [(x[0], x[4]) for x, y in zip(X, y) if y]
    #fail = [(x[0], x[4]) for x, y in zip(X, y) if not y]

    #good = [(x[0], x[5]) for x, y in zip(X, y) if y]
    #fail = [(x[0], x[5]) for x, y in zip(X, y) if not y]

    ## Relacion entre variables
    #good = [(x[2], x[4]) for x, y in zip(X, y) if y]
    #fail = [(x[2], x[4]) for x, y in zip(X, y) if not y]

    ## Caudal es la clave
    good = [(x[0], x[3]) for x, y in zip(X, y) if y]
    fail = [(x[0], x[3]) for x, y in zip(X, y) if not y]

    print(len(good))
    print(len(fail))
    goodX, goodY = zip(*good)        
    failX, failY = zip(*fail)

    plt.plot(goodX, goodY, "b.", markersize=7)
    plt.plot(failX, failY, "r.",markersize=2)
    plt.show()        




def study_plot_multiclass(X,y):
    
    classes = set(y)

    for size, _class in enumerate(classes):
        ##Irradiancia -> No sense
        #points = [(x[0], x[1]) for x, y in zip(X, y) if y==_class]
        
        ##T ambiente -> No sense
        #points = [(x[0], x[2]) for x, y in zip(X, y) if y==_class]
        
        ##Caudal
        #points = [(x[0], x[3]) for x, y in zip(X, y) if y==_class]
        
        ##T entrada -> No sense
        #points = [(x[0], x[4]) for x, y in zip(X, y) if y==_class]
        
        ##T salida -> No sense
        #points = [(x[0], x[5]) for x, y in zip(X, y) if y==_class]
        
        ##offset
        #points = [(x[0], x[6]) for x, y in zip(X, y) if y==_class]
        

        ##Caudal vs T Salida
        points = [(x[3], x[4], x[5]) for x, y in zip(X, y) if y==_class]
        
        Xs, Ys, Zs = zip(*points)        
        plt.plot(Xs, Ys, Zs, ".", markersize=len(classes)-size)
    
    plt.show()        



def k_fold_classify(X_test, y_test, model):

    means, sds = model["means"], model["sds"]
    sd_factor = model["sd_factor"]
    is_within_bounds = lambda value, i: ((means[i] - sd_factor*sds[i]) <= value) and (value <= (means[i] + sd_factor*sds[i])) 
    is_positive = lambda X: not bool(len(X) - len([xi for i, xi in enumerate(X) if is_within_bounds(xi, i)]))

    true_positives, true_negatives, false_negatives, false_positives = 0, 0, 0, 0

    for X, y in zip(X_test, y_test):
        if y:
            if is_positive(X):
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if is_positive(X):
                false_positives += 1
            else:
                true_negatives += 1

    print(" ", 1, "   ", 0)
    print(1, true_positives, false_negatives)
    print(0, false_positives, true_negatives)
    print()

    precision =  (true_positives + true_negatives)/ (false_positives + true_negatives + true_positives + false_negatives) 
    sensitivity =  (true_positives)/ (true_positives + false_negatives) 
    specificity =  (true_negatives)/ (true_negatives + false_positives) 

    print("precision", round(precision,2))
    print("sensitivity", round(sensitivity,2))
    print("specificity", round(specificity,2))
