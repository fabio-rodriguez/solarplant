import numpy as np
import os
import scipy.io
import statistics
import sys

from aux_functions import *
from classification import *
from read_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def read_data():

    dirname = os.path.dirname(__file__)
    
    # Grupo 1, Caudal ascendente
    
    relative_path = 'data/Recirculación/Grupo_1_170809/Caudal_ascendente/'
    no_fail_path = 'Datos_sin_fallo/G1_sin_fallos.mat'
    pos_offset_path = '1_Offset positivo en caudal/G1_con_fallos.mat'
    neg_offset_path = '2_Offset negativo en caudal/G1_con_fallo2.mat'

    G1_AS_good = scipy.io.loadmat(os.path.join(dirname, relative_path+no_fail_path))
    G1_AS_positive_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+pos_offset_path))
    G1_AS_negative_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+neg_offset_path))
    
    # Grupo 1, Caudal descendente
    
    relative_path = 'data/Recirculación/Grupo_1_170809/Caudal_descendente/'
    no_fail_path = 'Datos sin fallo/G1_sin_fallos_dcr.mat'
    pos_offset_path = '1_Offset positivo en caudal/G1_con_fallo1_dcr.mat'
    neg_offset_path = '2_Offset negativo en caudal/G1_con_fallo2_dcr.mat'

    G1_DES_good = scipy.io.loadmat(os.path.join(dirname, relative_path+no_fail_path))
    G1_DES_positive_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+pos_offset_path))
    G1_DES_negative_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+neg_offset_path))
    
    # Grupo 2, Caudal ascendente
    
    relative_path = 'data/Recirculación/Grupo_2_290609/'
    no_fail_path = 'Datos sin fallo/G2_sin_fallos.mat'
    pos_offset_path = '1_Offset positivo/G2_offset_positivo.mat'
    neg_offset_path = '2_Offset negativo/G2_offset_negativo.mat'

    G2_good = scipy.io.loadmat(os.path.join(dirname, relative_path+no_fail_path))
    G2_positive_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+pos_offset_path))
    G2_negative_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+neg_offset_path))

    return G1_AS_good, G1_AS_positive_offset, G1_AS_negative_offset, G1_DES_good, G1_DES_positive_offset, G1_DES_negative_offset, G2_good, G2_positive_offset, G2_negative_offset 


def delete_columns(array, indexes):
    return np.delete(array, indexes, 1)


def test1(g1as_good, g1as_pos, g1as_neg, g1des_good, g1des_pos, g1des_neg, g2_good, g2_pos, g2_neg):

    print("Grupo 1 Caudal Ascendente (Offset Positivo)")
    sd_classify(g1as_good, g1as_pos)
    print("*************************")
    print()
    
    print("Grupo 1 Caudal Ascendente (Offset Negativo)")
    sd_classify(g1as_good, g1as_neg)
    print("*************************")
    print()
    
    print("Grupo 1 Caudal Descendente (Offset Positivo)")
    sd_classify(g1des_good, g1des_pos)
    print("*************************")
    print()
 
    print("Grupo 1 Caudal Descendente (Offset Negativo)")
    sd_classify(g1des_good, g1des_neg)
    print("*************************")
    print()
 
    print("Grupo 2 (Offset Positivo)")
    sd_classify(g2_good, g2_pos)
    print("*************************")
    print()
 
    print("Grupo 2 (Offset Negativo)")
    sd_classify(g2_good, g2_neg)
    print("*************************")
    print()

    print("Grupo 1 Caudal Ascendente (Offset Positivo y Negativo)")
    sd_classify(g1as_good, np.concatenate((g1as_pos, g1as_neg), axis=0))
    print("*************************")
    print()
    
    print("Grupo 1 Caudal Descendente (Offset Positivo y Negativo)")
    sd_classify(g1des_good, np.concatenate((g1des_pos, g1des_neg), axis=0))
    print("*************************")
    print()
    

    print("Grupo 2 (Offset Positivo y Negativo)")
    sd_classify(g2_good, np.concatenate((g2_pos, g2_neg), axis=0))
    print("*************************")
    print()
    
    print("TODOS")
    sd_classify(np.concatenate((g1as_good, g1des_good, g2_good), axis=0), 
                np.concatenate((g1as_pos, g1as_neg, g1des_pos, g1des_neg, g2_pos, g2_neg), axis=0))
    print("*************************")
    print()


def merge_data(g1as_good, g1as_pos, g1as_neg, g1des_good, g1des_pos, g1des_neg, g2_good, g2_pos, g2_neg):
    
    X_good = np.concatenate((g1as_good, g1des_good, g2_good), axis=0)
    X_fail = np.concatenate((g1as_pos, g1as_neg, g1des_pos, g1des_neg, g2_pos, g2_neg), axis=0)
    
    y_good = np.ones(X_good.shape[0])
    y_fail = np.zeros(X_fail.shape[0])

    return np.concatenate((X_good, X_fail), axis=0), np.concatenate((y_good, y_fail), axis=0)


def test_networks():
    
    X  = read_quality_data()
    Time = X[:, 0]

    mean_y_flow = sum(X[:, 3])/len(X[:, 3])
    std_y_flow = statistics.pstdev(X[:, 3])

    mean_y_irr = sum(X[:, 1])/len(X[:, 1])
    std_y_irr = statistics.pstdev(X[:, 1])

    X = preprocessing.scale(X) 

    y_flow = X[:, 3]
    y_irr = X[:, 1]

    ##Flow From Four
    Xs = np.delete(X, [0, 3, 6], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_flow, random_=2**11)
    model, hist = k_fold_regression_NN(X_train, y_train, 4, epochs_permodel=500, n_splits=5, seed=2**11, path="models/test/flow-from-4/")

    plot_history(hist, path="models/test/flow-from-4/", reg=1)

    with open("data.txt", "w+") as file:
        error = test_model_from_path(X_test, y_test, "models/test/flow-from-4/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow)
        file.write(f"Validation Set Error flow-from-4: {error} \n")
        error = test_model_from_path(Xs, y_flow, "models/test/flow-from-4/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow, Time=Time)
        file.write(f"Total Set Error flow-from-4: {error} \n")

    ##Flow From Three
    Xs = np.delete(X, [0, 1, 3, 6], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_flow, random_=2**11)
    model, hist = k_fold_regression_NN(X_train, y_train, 3, epochs_permodel=500, n_splits=5, seed=2**11, path="models/test/flow-from-3/")

    plot_history(hist, path="models/test/flow-from-3/", reg=1)

    with open("data.txt", "a+") as file:
        error = test_model_from_path(X_test, y_test, "models/test/flow-from-3/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow)
        file.write(f"Validation Set Error flow-from-3: {error} \n")
        error = test_model_from_path(Xs, y_flow, "models/test/flow-from-3/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow, Time=Time)
        file.write(f"Total Set Error flow-from-3: {error} \n")

    ##Irr From Four
    Xs = np.delete(X, [0, 1, 6], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_irr, random_=2**11)
    model, hist = k_fold_regression_NN(X_train, y_train, 4, epochs_permodel=500, n_splits=5, seed=2**11, path="models/test/Irr-from-4/")

    plot_history(hist, path="models/test/Irr-from-4/", reg=1)

    with open("data.txt", "a+") as file:
        error = test_model_from_path(X_test, y_test, "models/test/Irr-from-4/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr)
        file.write(f"Validation Set Error Irr-from-4: {error} \n")
        error = test_model_from_path(Xs, y_irr, "models/test/Irr-from-4/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr, Time=Time)
        file.write(f"Total Set Error Irr-from-4: {error} \n")

    ##Irr From Three
    Xs = np.delete(X, [0, 1, 3, 6], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_irr, random_=2**11)
    model, hist = k_fold_regression_NN(X_train, y_train, 3, epochs_permodel=500, n_splits=5, seed=2**11, path="models/test/Irr-from-3/")

    plot_history(hist, path="models/test/Irr-from-3/", reg=1)

    with open("data.txt", "a+") as file:
        error = test_model_from_path(X_test, y_test, "models/test/Irr-from-3/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr)
        file.write(f"Validation Set Error Irr-from-3: {error} \n")
        error = test_model_from_path(Xs, y_irr, "models/test/Irr-from-3/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr, Time=Time)
        file.write(f"Total Set Error Irr-from-3: {error} \n")


def test_networks2():
    
    X  = read_NEW_DATA()
    N = len(X)
    print(N)

    mean_y_flow = sum(X[:, 1])/len(X[:, 1])
    std_y_flow = statistics.pstdev(X[:, 1])


    mean_y_irr = sum(X[:, 0])/len(X[:, 0])
    std_y_irr = statistics.pstdev(X[:, 0])

    X = preprocessing.scale(X) 

    y_flow = X[:, 1]
    y_irr = X[:, 0]

    path = "models/dnn_regression"

    ##Flow From Four
    name = "flow-from-4"
    Xs = np.delete(X, [1], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_flow, test_percent=0.3, random_=2**11)
    print(len(X_train))
    model, hist = k_fold_regression_NN(X_train, y_train, 4, epochs_permodel=500, n_splits=5, seed=2**11, path=f"{path}/{name}/")

    plot_history(hist, path= f"{path}/{name}/", reg=1)

    with open("data.txt", "w+") as file:
        error = test_model_from_path(X_test, y_test, f"{path}/{name}/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow)
        file.write(f"Validation Set Error flow-from-4: {error} \n")
        error = test_model_from_path(Xs, y_flow, f"{path}/{name}/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow, Time=list(range(N)))
        file.write(f"Total Set Error flow-from-4: {error} \n")

    ##Flow From Three
    name = "flow-from-3"
    Xs = np.delete(X, [0, 1], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_flow, test_percent=0.3, random_=2**11)
    model, hist = k_fold_regression_NN(X_train, y_train, 3, epochs_permodel=600, n_splits=5, seed=2**11, path=f"{path}/{name}/")

    plot_history(hist, path=f"{path}/{name}/", reg=1)

    with open("data.txt", "a+") as file:
        error = test_model_from_path(X_test, y_test, f"{path}/{name}/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow)
        file.write(f"Validation Set Error flow-from-3: {error} \n")
        error = test_model_from_path(Xs, y_flow, f"{path}/{name}/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow, Time=list(range(N)))
        file.write(f"Total Set Error flow-from-3: {error} \n")

    ##Irr From Four
    name = "Irr-from-4"
    Xs = np.delete(X, [0], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_irr, test_percent=0.3, random_=2**11)
    model, hist = k_fold_regression_NN(X_train, y_train, 4, epochs_permodel=500, n_splits=5, seed=2**11, path=f"{path}/{name}/")

    plot_history(hist, path=f"{path}/{name}/", reg=1)

    with open("data.txt", "a+") as file:
        error = test_model_from_path(X_test, y_test, f"{path}/{name}/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr)
        file.write(f"Validation Set Error Irr-from-4: {error} \n")
        error = test_model_from_path(Xs, y_irr, f"{path}/{name}/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr, Time=list(range(N)))
        file.write(f"Total Set Error Irr-from-4: {error} \n")

    ##Irr From Three
    name = "Irr-from-3"
    Xs = np.delete(X, [0, 1], 1)
    X_train, y_train, X_test, y_test = split_data(Xs, y_irr, test_percent=0.3, random_=2**11)
    model, hist = k_fold_regression_NN(X_train, y_train, 3, epochs_permodel=600, n_splits=5, seed=2**11, path=f"{path}/{name}/")

    plot_history(hist, path=f"{path}/{name}/", reg=1)

    with open("data.txt", "a+") as file:
        error = test_model_from_path(X_test, y_test, f"{path}/{name}/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr)
        file.write(f"Validation Set Error Irr-from-3: {error} \n")
        error = test_model_from_path(Xs, y_irr, f"{path}/{name}/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr, Time=list(range(N)))
        file.write(f"Total Set Error Irr-from-3: {error} \n")


def test_networks3():
    
    X  = read_NEW_DATA()
    path = "models/dnn_regression"
    #train_all_models(np.copy(X), 0, path, "Irr", epochs_permodel= 700)
    train_all_models(np.copy(X), 1, path, "Flow", epochs_permodel= 700)
    train_all_models(np.copy(X), 2, path, "Tamb", epochs_permodel= 700)
    train_all_models(np.copy(X), 3, path, "Tin", epochs_permodel= 700)
    train_all_models(np.copy(X), 4, path, "Tout", epochs_permodel= 700)


def train_all_models(X, yindex, path, name, test_percent=0.3, epochs_permodel=600, n_splits=5, seed=2**11, batch_size=1000):

    name = name if name else yindex
    
    N = len(X)
    mean_y = sum(X[:, yindex])/len(X[:, yindex])
    std_y = statistics.pstdev(X[:, yindex])

    X = preprocessing.scale(X) 
    y = X[:, yindex]

    ##Flow From Three
    for i in range(X.shape[1]):

        if i == yindex:
            methodName = f"{name}-from-4"
            Xs = np.delete(X, [yindex], 1)
        #    inputNodes = X.shape[1] - 1 
        else:
            methodName = f"{name}-from-3-removing({i})"
            Xs = np.delete(X, [i, yindex], 1)
        #    inputNodes = X.shape[1] - 2

        X_train, y_train, X_test, y_test = split_data(Xs, y, test_percent=test_percent, random_=seed)
        #model, hist = k_fold_regression_NN(X_train, y_train, inputNodes, epochs_permodel=epochs_permodel, batch_size_permodel= batch_size, n_splits=n_splits, seed=seed, path=f"{path}/{methodName}/")

        #plot_history(hist, path=f"{path}/{methodName}/", reg=1)

        with open("data.txt", "a+") as file:
            error, std = test_model_from_path(X_test, y_test, f"{path}/{methodName}/", reg=1, mean_y=mean_y, std_y=std_y)
            file.write(f"Validation Set on {methodName}: square_mean_error {error}, standar_deviation {std} \n")
            error, std = test_model_from_path(Xs, y, f"{path}/{methodName}/", reg=1, mean_y=mean_y, std_y=std_y, Time=list(range(N)))
            file.write(f"Total Set on {methodName}: square_mean_error {error}, standar_deviation {std} \n")
            file.write("\n")


def check_trained_models():
    X  = read_NEW_DATA()
    N = len(X)
    print(N)

    mean_y_flow = sum(X[:, 1])/len(X[:, 1])
    std_y_flow = statistics.pstdev(X[:, 1])

    mean_y_irr = sum(X[:, 0])/len(X[:, 0])
    std_y_irr = statistics.pstdev(X[:, 0])

    X = preprocessing.scale(X) 

    y_flow = X[:, 1]
    y_irr = X[:, 0]

    Xs= np.delete(X, [1], 1)
    with open("testing.txt", "w+") as file:
        error = test_model_from_path(Xs, y_flow, "models/test2/flow-from-4/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow)
        file.write(f"Total Set Error flow-from-4: {error} \n")
    
    Xs= np.delete(X, [0, 1], 1)
    with open("testing.txt", "a+") as file:
        error = test_model_from_path(Xs, y_flow, "models/test2/flow-from-3/", reg=1, mean_y=mean_y_flow, std_y=std_y_flow)
        file.write(f"Total Set Error flow-from-3: {error} \n")
        

    Xs= np.delete(X, [0], 1)
    with open("testing.txt", "a+") as file:
        error = test_model_from_path(Xs, y_irr, "models/test2/Irr-from-4/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr)
        file.write(f"Total Set Error Irr-from-4: {error} \n")
    
    Xs= np.delete(X, [0, 1], 1)
    with open("testing.txt", "a+") as file:
        error = test_model_from_path(Xs, y_irr, "models/test2/Irr-from-3/", reg=1, mean_y=mean_y_irr, std_y=std_y_irr)
        file.write(f"Total Set Error Irr-from-3: {error} \n")
        

if __name__ == "__main__":

    G1_AS_good, G1_AS_positive_offset, G1_AS_negative_offset, G1_DES_good, G1_DES_positive_offset, G1_DES_negative_offset, G2_good, G2_positive_offset, G2_negative_offset = read_data()
    
    data_list = [
        G1_AS_good[list(G1_AS_good.keys())[-1]], 
        G1_AS_positive_offset[list(G1_AS_positive_offset.keys())[-1]], 
        G1_AS_negative_offset[list(G1_AS_negative_offset.keys())[-1]], 
        G1_DES_good[list(G1_DES_good.keys())[-1]], 
        G1_DES_positive_offset[list(G1_DES_positive_offset.keys())[-1]], 
        G1_DES_negative_offset[list(G1_DES_negative_offset.keys())[-1]], 
        G2_good[list(G2_good.keys())[-1]], 
        G2_positive_offset[list(G2_positive_offset.keys())[-1]], 
        G2_negative_offset[list(G2_negative_offset.keys())[-1]]
    ]

    #study_plot(*merge_data(*data_list))

    #TEST_DATA = [delete_columns(data, [0,6,7]) for data in data_list]

    ## Mean and SD classification
    #  
    #test1(*TEST_DATA)

    ## DNN classification

    #X, y = merge_data(*TEST_DATA)
    #X_train, y_train, X_test, y_test = split_data(X, y, random_=2**11)

    #assert not set(map(tuple, X_train.tolist())).intersection(set(map(tuple, X_test.tolist()))), "Training and testing sets must be complement" 

    #basic_nn_classify(X_train, y_train, X_test, y_test)
    #test_model_from_path(X_test,y_test,"models/mymodel")


    ## Test k-fold

    #model, hist = k_fold(X_train, y_train, epochs_permodel=300, n_splits=5, random_=2**11)
    #plot_history(hist)
    #model.save("models/k_fold_test3/")
    
    #test_model_from_path(X_test, y_test, "models/k_fold_test3/")

    #test_model_from_path(X_test, y_test, "models/k_fold_test/")

    ##Save model
    #save_model_info(model, hist, "models/save_test/")

    ## SD k-fold
    #models = sd_k_fold(X_train, y_train, random_=1024, path="models/sd_k_fold/sd_k_fold.json", sd_factor=2.2)
    #models.sort(key=lambda x: x["precision"])
    #k_fold_classify(X_test, y_test, models[0])

    ## RANDOM FOREST
    
    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)

    #cl = RandomForestClassifier(n_estimators=20, random_state=0)
    #cl.fit(X_train, y_train)
    #y_pred = cl.predict(X_test)    

    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    #print(accuracy_score(y_test, y_pred))


    ## KFOLD RANDOM FOREST - ARREGLAR!!!

    #k_fold_generic(X_train, y_train, model=RandomForestClassifier(n_estimators=20, random_state=2**11), path="models/RF", n_splits=5, random_=2**11)


    ## NN for multiple classes

    #X, y  = read_FAULT_DATA("multiclass")
    #X = np.delete(X, [0, 6], 1)
    #X_train, y_train, X_test, y_test = split_data(X, y, random_=2**11)

    #print(len(set(map(tuple, X_train.tolist())).intersection(set(map(tuple, X_test.tolist())))))
    #assert not set(map(tuple, X_train.tolist())).intersection(set(map(tuple, X_test.tolist()))), "Training and testing sets must be complement" 

    #k_fold_multiclass(X_train, y_train, create_model=basic_model_mult_classes ,epochs_permodel=300, n_splits=5, seed =2**11)

    ## NN for new data binary classification

    #X, y  = read_FAULT_DATA("binary")
    #X = np.delete(X, [0, 6], 1)
    #X_train, y_train, X_test, y_test = split_data(X, y, random_=2**11)

    #model, hist = k_fold(X_train, y_train, epochs_permodel=100, n_splits=4, random_=2**11)
    #plot_history(hist)
    #model.save("models/k_fold_test3/")
    
    #test_model_from_path(X_test, y_test, "models/k_fold_test3/")


    ## NN Regression for FLOW prediction
    
    #X  = read_quality_data()
    #mean_y = sum(X[:, 3])/len(X[:, 3])
    #std_y = statistics.pstdev(X[:, 3])

    #X = preprocessing.scale(X) 
    #y = X[:, 3]
    #X = np.delete(X, [0, 3, 6], 1)
    #X = np.delete(X, [0, 1, 3, 6], 1)

    #X_train, y_train, X_test, y_test = split_data(X, y, random_=2**11)
    #model, hist = k_fold_regression_NN(X_train, y_train, epochs_permodel=500, n_splits=5, seed=2**11, path="models/k_fold_regression_flow_norm-Irr/")

    #plot_history(hist, reg=1)
    #test_model_from_path(X_test, y_test, "models/k_fold_regression_flow_norm-Irr/", reg=1)
    #test_model_from_path(X_test, y_test, "models/k_fold_regression_flow_norm-Irr/", reg=1, mean_y=mean_y, std_y=std_y)


    ## NN Regression for Irradiance prediction
    
    #X = read_quality_data()
    #mean_y = sum(X[:, 1])/len(X[:, 1])
    #std_y = statistics.pstdev(X[:, 1])
    
    #X = preprocessing.scale(X)  
    #y = X[:, 1]
    #X = np.delete(X, [0, 1, 6], 1)
    #X = np.delete(X, [0, 1, 3, 6], 1)

    #X_train, y_train, X_test, y_test = split_data(X, y, random_=2**11)
    #model, hist = k_fold_regression_NN(X_train, y_train, epochs_permodel=500, n_splits=5, seed=2**11, path="models/k_fold_regression_Irr_Standarization-Flow/")

    #plot_history(hist, reg=1)
    #test_model_from_path(X_test, y_test, "models/k_fold_regression_Irr_Standarization-Flow/", reg=1, mean_y=mean_y, std_y=std_y)

    #test_networks()
    
    #test_networks2()

    test_networks3()
    