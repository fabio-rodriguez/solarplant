import json
from matplotlib.pyplot import hist
import numpy
import statistics

from aux_functions import plot_history, mean_sd_bycolumn, print_confusion_matrix, save_model_info
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score 
from sklearn.preprocessing import LabelEncoder


def sd_classify(good_data, fail_data, sd_factor=2.5):
    
    #n = len(good_data[0])
    means, sds = mean_sd_bycolumn(good_data)

    is_within_bounds = lambda value, i: ((means[i] - sd_factor*sds[i]) <= value) and (value <= (means[i] + sd_factor*sds[i])) 
    is_positive = lambda X: not bool(len(X) - len([xi for i, xi in enumerate(X) if is_within_bounds(xi, i)]))

    ## Testing with the good data

    true_positives = 0
    false_negatives = 0

    for data in good_data:
        if is_positive(data):
            true_positives += 1
        else:
            false_negatives += 1

    ## Classification

    true_negatives = 0
    false_positives = 0

    for data in fail_data:
        if is_positive(data):
            false_positives += 1
        else:
            true_negatives += 1

    # Print Results
    
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


def sd_k_fold(X_train, y_train, sd_factor=2.5, n_splits=5, random_=None, path=None):
    
    if random_:
        skf = StratifiedKFold(n_splits = n_splits, random_state = random_, shuffle = True)
    else:
        skf = StratifiedKFold(n_splits = n_splits)

    models = []

    for train, test in skf.split(X_train, y_train):

        y = y_train[train]
        data = [x for i, x in enumerate(X_train[train]) if y[i] == 1]
        means, sds = mean_sd_bycolumn(data)

        is_within_bounds = lambda value, i: ((means[i] - sd_factor*sds[i]) <= value) and (value <= (means[i] + sd_factor*sds[i])) 
        is_positive = lambda X: not bool(len(X) - len([xi for i, xi in enumerate(X) if is_within_bounds(xi, i)]))

        ## Validation

        X_val = X_train[test]
        y_val = y_train[test]
        true_positives, true_negatives, false_negatives, false_positives = 0, 0, 0, 0

        for X, y in zip(X_val, y_val):
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

        models.append({
            "means": means.tolist(),
            "sds": sds.tolist(),
            "sd_factor": sd_factor,
            "precision": (true_positives + true_negatives)/ (false_positives + true_negatives + true_positives + false_negatives),
            "sensitivity": (true_positives)/ (true_positives + false_negatives), 
            "specificity":   (true_negatives)/ (true_negatives + false_positives)     
        })
        
    with open(path if path else "models/sd_kfold.json", "w+") as file:
        file.write(json.dumps(models))

    return models


def basic_nn_classify(x_train, y_train, x_test, y_test, path="models/mymodel"):

    model=basic_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_split=0.3, epochs=200, batch_size=20)

    plot_history(history)

    _, accuracy = model.evaluate(x_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))

    print()
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save(path)


def basic_model():

    model = Sequential()
    model.add(Dense(10, input_dim=7, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def basic_regression_model(input_size):

    model = Sequential()
    model.add(Dense(40, input_dim=input_size, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model


def k_fold(X_train, y_train, create_model=basic_model, n_splits=5, random_=None, epochs_permodel=150, batch_size_permodel=10, path=None):
    
    if random_:
        skf = StratifiedKFold(n_splits = n_splits, random_state = random_, shuffle = True)
    else:
        skf = StratifiedKFold(n_splits = n_splits)

    models = []

    for train, test in skf.split(X_train, y_train):

        X = X_train[train]
        y = y_train[train]
        X_val = X_train[test]
        y_val = y_train[test]

        model=create_model()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        record = model.fit(X, y, validation_data=(X_val,y_val), epochs=epochs_permodel, batch_size=batch_size_permodel)
        
        models.append((record.history["val_accuracy"], model, record))

    models.sort()

    if path:
        model.save("models/dnn_k_fold_test")

    return models[0][1], models[0][2]


def build_sequential_dnn(layers):
    '''
        Builds a sequential neural network model given its layers
    '''

    model = Sequential()

    for layer in layers:
        model.add(layer)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    return model


def build_dense_sequential_dnn(nodes_by_layer, layers_activation, input_dim):
    '''
        Build a sequential dense dnn based on lists of node numbers and layers activation
    '''

    assert len(nodes_by_layer) == len(layers_activation), "The list of node numbers and activation functions must have len n_layers" 
        
    layers = [Dense(nodes_by_layer[0], input_dim=input_dim, activation=layers_activation[0])]
    layers += [Dense(nodes, activation=activation) for nodes, activation in zip(nodes_by_layer[1:], layers_activation[1:])]

    return build_sequential_dnn(layers)


def k_fold_generic(X_train, y_train, model=None, path=None, random_=None, n_splits=5):

    if random_:
        skf = StratifiedKFold(n_splits = n_splits, random_state = random_, shuffle = True)
    else:
        skf = StratifiedKFold(n_splits = n_splits)

    models = []

    for train, test in skf.split(X_train, y_train):

        X = X_train[train]
        y = y_train[train]
        X_val = X_train[test]
        y_val = y_train[test]

        record = model.fit(X, y, validation_data=(X_val,y_val))        
        models.append((record.history["val_accuracy"], model, record))

    models.sort()

    if path:
        model.save("models/dnn_k_fold_test")

    return models[0][1], models[0][2]


def basic_model_mult_classes():

    return build_dense_sequential_dnn([36,18,9], ['relu', 'relu', 'softmax'], 5)


def k_fold_multiclass(X, y, create_model=basic_model, n_splits=4, seed=None, epochs_permodel=150, batch_size_permodel=10, path=None):

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    one_hot_y = np_utils.to_categorical(encoded_Y)
    
    #model = create_model()
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    if seed:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        kfold = KFold(n_splits=n_splits)

    results = list()
    for train_ix, test_ix in kfold.split(X):
            # prepare data
            X_train, X_val = X[train_ix], X[test_ix]
            y_train, y_val = one_hot_y[train_ix], one_hot_y[test_ix]
            # define model
            model = create_model()
            # fit model
            record = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
            # make a prediction on the test set
            results.append((record.history["val_accuracy"], model, record))
    
    results.sort(key= lambda x: x[0])
    _, m, h = results[0]

    if not path:
        plot_history(h)
        save_model_info(m, h, "models/dnn_k_fold_multiclass")
   
    return results


def k_fold_regression_NN(X_train, y_train, input_size, create_model=basic_regression_model, n_splits=5, seed=None, epochs_permodel=150, batch_size_permodel=10, path=None):
    
    if seed:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        kfold = KFold(n_splits=n_splits)


    models = []

    for train, test in kfold.split(X_train, y_train):

        X = X_train[train]
        y = y_train[train]
        X_val = X_train[test]
        y_val = y_train[test]

        model=create_model(input_size)
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        
        #record = model.fit(X, y, validation_data=(X_val,y_val), epochs=epochs_permodel, batch_size=batch_size_permodel, verbose=0)
        record = model.fit(X, y, validation_data=(X_val,y_val), epochs=epochs_permodel, batch_size=1000) # verbose=0
        models.append((record.history['val_mean_absolute_error'], model, record))

    models.sort()

    if path:
        model.save(path)
    else:
        model.save("models/dnn_k_fold_test")

    return models[0][1], models[0][2]
