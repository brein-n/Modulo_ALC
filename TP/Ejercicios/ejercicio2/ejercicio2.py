import numpy as np
from alc import svd_reducida

#a)
# Le saqué el limite de 12, porque la funcion esta definida para n puntos y yo habia
#Especificado para una matriz de 24x12
def obtenerQ3d(imgPts, objPts) -> np.ndarray:
    """
    Arma la matriz Q del sistema homogeneo Qm = 0 del DLT 3D (ecuación 6).

    Parámetros:
        imgPts: array (n, 2) con las coordenadas (u_i, v_i) de cada punto en la imagen, en píxeles.
        objPts: array (n, 3) con las coordenadas (U_i, V_i, W_i) de cada punto en el mundo real.
                El punto i de imgPts debe corresponder al punto i de objPts.
    Retorna:
        Q: array (2n, 12). Las filas 2i y 2i+1 son las dos ecuaciones del punto i.
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)
    n = objPts.shape[0]

    if imgPts.shape != (n, 2) or objPts.shape != (n, 3):
        raise ValueError("imgPts debe ser (n, 2) y objPts (n, 3), con el mismo n")

    Q = np.zeros((2 * n, 12))

    for i in range(n):
        # Coordenadas del punto i en la imagen
        ui = imgPts[i, 0]
        vi = imgPts[i, 1]

        # Coordenadas del punto i en el mundo real
        Ui = objPts[i, 0]
        Vi = objPts[i, 1]
        Wi = objPts[i, 2]

        # Ecuación de u: [P_i^T, 0^T, -u_i P_i^T]
        Q[2 * i] = [
            Ui, Vi, Wi, 1,
            0, 0, 0, 0,
            -ui * Ui, -ui * Vi, -ui * Wi, -ui
        ]

        # Ecuación de v: [0^T, P_i^T, -v_i P_i^T]
        Q[2 * i + 1] = [
            0, 0, 0, 0,
            Ui, Vi, Wi, 1,
            -vi * Ui, -vi * Vi, -vi * Wi, -vi
        ]

    return Q


#b)
def resolverDLT3D(Q):
    # el vector que minimiza ||Qm||^2 es el autovector asociado al menor valor singular
    # que este es el último autovector de la matriz V
    _,_,V = svd_reducida(Q)
    M = V[:-1].reshape((3,4))
    # reshape ((3,4)) toma la matriz [1,2,3,4,5,6,7,8,9,10,11,12]
    # y la convierte en una matriz de 3x4 de la forma [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    return  M
