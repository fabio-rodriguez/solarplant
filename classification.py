
import numpy
import statistics

from aux_functions import plot_history
from keras.models import Sequential
from keras.layers import Dense


def sd_classify(good_data, fail_data, sd_factor=2.5):
    
    n = len(good_data[0])
    means = numpy.zeros(n)
    sds = numpy.zeros(n)

    # Calcular la media y la desviacion estandar de cada campo de los datos buenos   
    for i in range(n):
        columni = [column[i] for column in good_data]
        means[i] = sum(columni)/len(columni)
        sds[i] = statistics.pstdev(columni)

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


def basic_nn_classify(x_train, y_train, x_test, y_test, path="models/"):

    model=basic_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_split=0.3, epochs=10, batch_size=20)

    plot_history(history)

    _, accuracy = model.evaluate(x_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))

    print()
    _, accuracy = model.evaluate(x_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))

    model.save(path)


def basic_model():

    model = Sequential()
    model.add(Dense(12, input_dim=5, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

