
import json
import numpy
import statistics

from aux_functions import plot_history, mean_sd_bycolumn
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold


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

    history = model.fit(x_train, y_train, validation_split=0.3, epochs=15, batch_size=20)

    plot_history(history)

    _, accuracy = model.evaluate(x_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))

    print()
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save(path)


def basic_model():

    model = Sequential()
    model.add(Dense(20, input_dim=5, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

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

    for layer in nodes:
        model.add(layer)
        
    return model


def build_dense_sequential_dnn(n_layers, n_layer_nodes, layers_activation, input_size, output_size):
    '''
        Build a sequential dense dnn based on lists of node numbers and layers activation
    '''

    assert n_layers == len(n_layer_nodes) ==  len(layers_activation), "The list of node numbers and activation functions must have len n_layers" 
    assert output_size == n_layer_nodes[-1], "Last element of n_layer_nodes corresponds tothe last layer, so its value muts be equal to output_size" 
        
    layers = [Dense(n_layer_nodes[0], layers_activation[0], input_dim=input_size)]
    layers += [Dense(nodes, activation) for nodes, activation in zip(n_layer_nodes, layers_activation)]

    return build_sequential_dnn(layers)
    



