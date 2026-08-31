import numpy as np  

# Laboratorio 1

def error(x, y):
    x = np.float64(x)
    y = np.float64(y)
    
    error = abs(x - y)
    
    return error

def error_relativo(x, y):
    x = np.float64(x)
    y = np.float64(y)
    
    numerador = abs(x - y)
    denominador = abs(x)
    
    error_relativo = numerador/denominador
    
    return error_relativo

def matricesIguales(A, B):
    
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    
    # Aca suponemos que las dimensiones son iguales
    for n in range (len(A)):
        # no necesariamente es cuadrada 
        for m in range(len(A[0])):
            # if A[n][m] != B[n][m]:
            # Con el distinto no me pasa por lo que lo panteo con error
            # Si uso error funciona a partir de 1e-07 pero con resta a partir de 1e-08
            if error(A[n][m], B[n][m]) > 1e-07:
                return False
            
    return True

# Laboratorio 2

def rota(theta):
    # Para esta matriz tenemos que la forma es [[cos, sin], [-sin,cos]]
    # usamos las funciones trigonometricas de numpy
    
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                 [np.sin(theta), np.cos(theta)]])
    
    return R

def escala(s):
    n = len(s)
    S = np.zeros((n,n))
    
    for i in range(len(s)):
        S[i][i] = s[i]
        
    return S


def multiplicacionMatricial(A,B):
    
    if len(A[0]) !=  len(B):
        return None
    
    res = np.zeros((len(A), len(B[0])), dtype = float)
    suma = 0
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            # Este k puede iterar tanto en la len de las columans de A
            # como en la len de las filas de B
            for k in range(len(A[0])):
                elem_A = A[i][k]
                elem_B = B[k][j]
                suma += elem_A * elem_B
            
            res [i][j] = suma
            suma = 0
                  
    return res


def rota_y_escala(theta, s):
    # ojo que para que la matriz sea de 2*2, s tiene que tener len(s) = 2
    if len(s) != 2:
        return None
    R = rota(theta)
    S = escala(s)
    
    # Si primero rota y despues escala es RoS
    res = multiplicacionMatricial(S,R)
    
    return res

def afin(theta,s,b):
    if len(s) != 2 or len(b) != 2:
        return None
    
    R_aux = rota(theta)
    S_aux = escala(s)
    
    multiplicacion = multiplicacionMatricial(S_aux, R_aux)
    
    res = np.zeros((3,3))
    
    for i in range(2):
        for j in range(2):
            res[i][j] = multiplicacion[i][j]
            
    for i in range(len(b)):
        res[i][2] = b[i]
        
    res[2][2] = 1
    
    return res


def multiplicacionMatrizVector(A, v):
    
    if len(A[0]) != len(v):
        return None

    res = np.zeros(len(A))
    
    for i in range(len(A[0])):
        suma = 0
        for k in range(len(v)):
            suma += A[i][k] * v[k]
            
        res[i] = suma
        
    return res


def trans_afin(v, theta, s, b):
    A = afin(theta, s, b)
    
    nuevo_v = np.zeros(3)
    
    for i in range(len(v)):
        nuevo_v[i] = v[i]
    nuevo_v[2] = 1
    A_v = multiplicacionMatrizVector(A, nuevo_v)
    
    res = A_v[:2]
    
    
    return res


