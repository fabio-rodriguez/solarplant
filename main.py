from classification import sd_classify


if __name__ == "__main__":

    
    # Grupo 1, Caudal ascendente
    
    root_path = "./data/Recirculación/Grupo_1_170809/Caudal_ascendente"

    with open(root_path + "/Datos_sin_fallo/G1_sin_fallos_dcr.mat", "r") as file:
        print(file.read())

    G1_good = None

    G1_positive_offset = None

    G1_negative_offset = None
    