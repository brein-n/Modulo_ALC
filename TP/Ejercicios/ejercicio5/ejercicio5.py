import numpy as np
import matplotlib.pyplot as plt

rgb= plt.imread('/home/luis/PycharmProjects/Modulo_ALC/TP/salguero/img_10_18_47_351.jpg')

plt.imshow(rgb)

X = np.array([281, 889, 1139, 418,833])
Y = np.array([474,521, 493, 385,262])

plt.plot(X, Y, 'ro', markersize = 2)
plt.show()
#
# import numpy as np
# import ast
#
# # Import "trucho" de obtenerQ3d: con un import normal se corre todo el script de ejercicio2.py
# # (carga calib/3d e imprime la calibracion del 2). Leemos el archivo, nos quedamos solo con sus
# # import y sus definiciones de funciones, y ejecutamos eso, sin el resto del script
# rutaEjercicio2 = '/home/luis/PycharmProjects/Modulo_ALC/TP/Ejercicios/ejercicio2/ejercicio2.py'
# with open(rutaEjercicio2) as archivoEjercicio2:
#     arbolEjercicio2 = ast.parse(archivoEjercicio2.read())
# arbolEjercicio2.body = [nodo for nodo in arbolEjercicio2.body
#                         if isinstance(nodo, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
# funcionesEjercicio2 = {}
# # compile con la ruta del archivo, asi si algo falla el error muestra la linea de ejercicio2.py
# exec(compile(arbolEjercicio2, rutaEjercicio2, 'exec'), funcionesEjercicio2)
# obtenerQ3d = funcionesEjercicio2['obtenerQ3d']
#
# # a) Carga de los puntos de la esquina
#
# # Cada fila es un punto: pts;X;Y;Z;u_0;v_0;u_1;v_1;u_2;v_2;u_3;v_3
# # (X, Y, Z) en metros y (u_k, v_k) su posicion en la vista k, en pixeles
# datos = np.loadtxt('/home/luis/PycharmProjects/Modulo_ALC/TP/salguero/world_img_pts.csv', delimiter=';', skiprows=1)
#
# # Coordenadas del mundo real: una tira por coordenada, con un valor por punto
# X = datos[:, 1]
# Y = datos[:, 2]
# Z = datos[:, 3]
# print("X =", X)
# print("Y =", Y)
# print("Z =", Z)
#
# # Coordenadas de la imagen: despues de pts, X, Y, Z vienen dos columnas (u_k, v_k) por cada vista,
# # asi que u[k] y v[k] son las tiras de la vista k
# cantVistas = (datos.shape[1] - 4) // 2
# u = []
# v = []
# for k in range(cantVistas):
#     u.append(datos[:, 4 + 2 * k])
#     v.append(datos[:, 5 + 2 * k])
#     print("vista", k, ": u =", u[k], " v =", v[k])
#
# # Formato de obtenerQ3d (ejercicio2.py): objPts (n, 3) con una fila (X_i, Y_i, Z_i) por punto
# # e imgPts (n, 2) con una fila (u_i, v_i) por punto. vstack apila las tiras como filas
# # y .T las pasa a columnas
# objPts = np.vstack((X, Y, Z)).T
# print("objPts", objPts.shape)
# print(objPts)
#
# # Un imgPts por vista, porque cada vista tiene su propia imagen de los mismos puntos
# imgPtsVistas = []
# for k in range(cantVistas):
#     imgPtsVistas.append(np.vstack((u[k], v[k])).T)
#     print("imgPts de la vista", k, imgPtsVistas[k].shape)
#     print(imgPtsVistas[k])
#
#
# print(objPts)
# print(imgPtsVistas)
#
# # Una Q por vista: los mismos objPts con los imgPts de cada vista
# QVistas = []
# for k in range(cantVistas):
#     QVistas.append(obtenerQ3d(imgPtsVistas[k], objPts))
#     print("Q de la vista", k, QVistas[k].shape)
#     print(QVistas[k])
#
#
