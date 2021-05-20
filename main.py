import numpy as np
import os
import scipy.io
import sys

from aux_functions import *
from classification import *
from read_data import read_FAULT_DATA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


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

    TEST_DATA = [delete_columns(data, [0,6,7]) for data in data_list]

    ## Mean and SD classification
    #  
    #test1(*TEST_DATA)

    ## DNN classification

    X, y = merge_data(*TEST_DATA)
    X_train, y_train, X_test, y_test = split_data(X, y, random_=2**11)

    assert not set(map(tuple, X_train.tolist())).intersection(set(map(tuple, X_test.tolist()))), "Training and testing sets must be complement" 

    #basic_nn_classify(X_train, y_train, X_test, y_test)
    #test_model_from_path(X_test,y_test,"models/mymodel")


    ## Test k-fold

    #model, hist = k_fold(X_train, y_train, epochs_permodel=300, n_splits=5, random_=2**11)
    #plot_history(hist)
    #model.save("models/k_fold_test2/")
    
    #test_model_from_path(X_test, y_test, "models/k_fold_test2/")

    #test_model_from_path(X_test, y_test, "models/k_fold_test/")

    ##Save model
    #save_model_info(model, hist, "models/save_test/")

    ## SD k-fold
    #models = sd_k_fold(X_train, y_train, random_=1024, path="models/sd_k_fold/sd_k_fold.json", sd_factor=2.2)
    #models.sort(key=lambda x: x["precision"])
    #k_fold_classify(X_test, y_test, models[0])

    ## RANDOM FOREST
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #cl = RandomForestClassifier(n_estimators=20, random_state=0)
    #cl.fit(X_train, y_train)
    #y_pred = cl.predict(X_test)    

    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    #print(accuracy_score(y_test, y_pred))


    ## KFOLD RANDOM FOREST - ARREGLAR!!!

    #k_fold_generic(X_train, y_train, model=RandomForestClassifier(n_estimators=20, random_state=2**11), path="models/RF", n_splits=5, random_=2**11)


    ## NN for multiple classes

    X, y  = read_FAULT_DATA("multiclass")
    X_train, y_train, X_test, y_test = split_data(X, y, random_=2**11)

    #print(len(set(map(tuple, X_train.tolist())).intersection(set(map(tuple, X_test.tolist())))))
    #assert not set(map(tuple, X_train.tolist())).intersection(set(map(tuple, X_test.tolist()))), "Training and testing sets must be complement" 

    k_fold_multiclass(X_train, y_train, create_model=basic_model_mult_classes ,epochs_permodel=300, n_splits=5, seed =2**11)

