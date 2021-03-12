import os
import scipy.io
import sys

from classification import sd_classify


if __name__ == "__main__":

    dirname = os.path.dirname(__file__)
    
    # Grupo 1, Caudal ascendente
    
    relative_path = 'data/Recirculación/Grupo_1_170809/Caudal_ascendente/'
    no_fail_path = 'Datos_sin_fallo/G1_sin_fallos.mat'
    neg_offset_path = '1_Offset positivo en caudal/G1_con_fallos.mat'
    pos_offset_path = '2_Offset negativo en caudal/G1_con_fallo2.mat'

    G1_good = scipy.io.loadmat(os.path.join(dirname, relative_path+no_fail_path))

    G1_positive_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+pos_offset_path))

    G1_negative_offset = scipy.io.loadmat(os.path.join(dirname, relative_path+neg_offset_path))
    
    print(G1_good)
    print(G1_positive_offset)
    print(G1_negative_offset)