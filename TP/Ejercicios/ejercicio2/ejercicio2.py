import numpy as np
from TP.entrega.alc import svd_reducida, inversa,calculaQR,traspuesta,arroba, norma


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


# b)
def resolverDLT3D(Q):
    # el vector que minimiza ||Qm||^2 es el autovector asociado al menor valor singular
    # que este es el último autovector de la matriz V
    _, _, V = svd_reducida(Q)
    M = V[:,-1].reshape((3, 4))
    # reshape ((3,4)) toma la matriz [1,2,3,4,5,6,7,8,9,10,11,12]
    # y la convierte en una matriz de 3x4 de la forma [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    return M


# c)

def devolverMatricesCalibracion3D(M):
    M_monio = M / norma(M[2, 0:3])
    M1 = M_monio[0:3,0:3]
    M2 = M_monio[:,3]
    M1Inversa = inversa(M1)
    R_traspuesta,K_inversa = calculaQR(M1Inversa,'GS')
    R = traspuesta(R_traspuesta)
    K = inversa(K_inversa)
    return R,t,K

# Cada fila de este archivo contiene las coordenadas de los puntos detectados en la imagen
# de la forma u1 v1 u2 v2 u3 v3 ... u12 v12
detectedCorners = np.loadtxt('/home/luis/PycharmProjects/Modulo_ALC/TP/calib/3d/detected_corners.txt')
# Con una imagen alcanza: tomamos la primera fila y la pasamos a (12, 2),
# reshape la recorre por filas, asi que queda la fila i = (u_i, v_i)
imgPts = detectedCorners[0].reshape((-1, 2))
print(imgPts.shape)
print(imgPts)

# Cada fila de este archivo contiene las coordenadas U_i, V_i, W_i del punto i en el mundo real
objPts = np.loadtxt('/home/luis/PycharmProjects/Modulo_ALC/TP/calib/3d/p_W_corners.txt',delimiter=',')
print(objPts.shape)
print(objPts)


Q3D = obtenerQ3d(imgPts,objPts)

print(Q3D.shape)
print(Q3D)
M = resolverDLT3D(Q3D)
print(M)
# Los primeros 3 valores de la ultima fila de M son la ultima fila de R (pag. 11), pero M sale
# con ||m|| = 1, no con esa fila de norma 1. Dividimos M por la norma 2 de esa fila para que
# sea consistente con R ortonormal (multiplicar por un escalar no cambia la solucion de Qm = 0)
MPrima = M / norma(M[2, 0:3])
print(MPrima)

R, t , K = devolverMatricesCalibracion3D(MPrima)
print(R)
print(t)
print(K)