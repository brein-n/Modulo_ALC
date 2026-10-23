import numpy as np  

#Laboratorio 0

def traspuesta(a):
    filas,columnas = a.shape
    # inicializamos la matriz vacia
    new = np.empty((columnas, filas), dtype=a.dtype)
    for i in range(filas):
        new[:, i] = a[i]
    #     agarra la fila i de a y guardala como la columna i de new
    return new


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
    
    a_filas, a_columnas = A.shape
    b_filas , b_columnas = B.shape
    if a_filas != b_filas or a_columnas != b_columnas:
        return False
    # Aca ya sabemos  que las dimensiones son iguales
    for n in range(0, a_filas):
        # no necesariamente es cuadrada 
        for m in range(0, a_columnas):
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


# Laboratorio 3



def norma(vector,p):
    """
    Devuelve la norma p del vector x.
    """
    if p == 'inf':
        return normaInfinito(vector)
    acum = 0
    for i in range(len(vector)):
        acum = abs(vector[i])**p + acum
    return acum**(1/p)

def normaliza(vectores,p):
    copia = vectores.copy()
    for i in range(len(vectores)):
        copia[i] = copia[i]/norma(copia[i],p)
    return copia

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q,p} y el vector x en el cual
    se alcanza el máximo.
    """
    maximo_encontrado = 0
    maximo_xs = None
    for _ in range(0,Np):
        x = np.random.normal(0, 1, A.shape[1])
        xs = x/norma(x,p)
        ys = A@xs
        ys_max = norma(ys,q)
        if ys_max > maximo_encontrado:
            maximo_encontrado = ys_max
            maximo_xs = xs
    return maximo_encontrado,maximo_xs

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    """
    absA = np.abs(A)
    norma1 = absA.sum(axis=0).max()
    normaInf = absA.sum(axis=1).max()
    if p == 1:
        return norma1
    if p == 'inf':
        return normaInf
    if p == [1, 'inf']:
        return [norma1, normaInf]
    return None


def normaInfinito(vector):
    return abs(vector).max()




def condMC(A,p,Np=10000):
    """
    Devuelve el número de condición de A usando la norma inducida p.
    """
    a_norm = normaMatMC(A,p,p,Np)
    a_inv_norm = normaMatMC(np.linalg.inv(A),p,p,Np)

    return a_norm[0]*a_inv_norm[0]

def condExacta(A,p):
    """
    Devuelve el número de condición de A a partir de la fórmula
    de la ecuación (1) usando la norma p.
    """
    a_norm = normaExacta(A,p)
    a_inv_norm = normaExacta(np.linalg.inv(A),p)
    return a_norm*a_inv_norm
