import numpy as np  

#Laboratorio 0

def arroba(A,B):
    if A.ndim == 1:
        A = A.reshape(1, -1)  # Vector → matriz fila
    if B.ndim == 1:
        B = B.reshape(-1, 1)  # Vector → matriz columna
    if A.shape[1] != B.shape[0]:
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    filas = A.shape[0]
    columnas = B.shape[1]
    C = np.zeros((filas,columnas))
    for i in range (filas):
        for j in range(columnas):
            C[i,j] = sum(A[i] * B[:,j])
    return C

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

def matricesIguales(a,b,atol=1e-8):
    filas_a,columnas_a = a.shape
    filas_b,columnas_b = b.shape
    if filas_a != filas_b or columnas_a != columnas_b:
        return False
    for i in range(filas_a):
        for j in range(columnas_a):
            if not np.isclose(a[i,j],b[i,j],atol=atol):
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

def normaliza_vector(vector, p=2):
    return vector / norma(vector, p)

def norma(vector,p=2):
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



# Laboratorio 4



def calculaLU(A):
    """
    Calcula la factorización LU de la matriz A y retorna las matrices L
    y U, junto con el número de operaciones realizadas.
    En caso de que la matriz no pueda factorizarse, retorna None.
    """
    if A is None or A.shape[0] != A.shape[1]:
        return None, None, 0
    cant_op = 0
    m, n = A.shape
    Ac = A.copy()

    for k in range(0, n - 1):
        if Ac[k, k] == 0:
            return None, None, 0

        for i in range(k + 1, n):
            Ac[i, k] = Ac[i, k] / Ac[k, k]
            cant_op += 1

            for j in range(k + 1, n):
                Ac[i, j] = Ac[i, j] - Ac[i, k] * Ac[k, j]
                cant_op += 2

    if Ac[n - 1, n - 1] == 0:
        return None, None, 0

    L = np.zeros(A.shape)
    for i in range(n):
        L[i, i] = 1
        for j in range(i):
            L[i, j] = Ac[i, j]

    U = np.zeros(A.shape)
    for i in range(n):
        U[i, i] = Ac[i, i]
        for j in range(i + 1, n):
            U[i, j] = Ac[i, j]

    return L, U, cant_op


def res_tri(L, b, inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular.
    Se puede indicar si es triangular inferior o superior usando
    el argumento inferior (por defecto, asumir que es triangular inferior).
    """
    vector_solucion = np.zeros_like(b, dtype=float)
    if inferior:
        for i in range(0, len(b)):
            acum = 0
            for j in range(0, i):
                acum = acum + L[i, j] * vector_solucion[j]
            vector_solucion[i] = (b[i] - acum) / L[i, i]
        return vector_solucion
    else:
        for i in range(len(b) - 1, -1, -1):
            acum = 0
            for j in range(i + 1, len(b)):
                acum = acum + L[i, j] * vector_solucion[j]
            vector_solucion[i] = (b[i] - acum) / L[i, i]
        return vector_solucion


def inversa(A):
    """
    Calcula la inversa de A empleando la factorización LU
    y las funciones que resuelven sistemas triangulares.
    """
    L, U, _ = calculaLU(A)
    if L is None:
        return None

    n = A.shape[0]
    canonicos = np.eye(n)
    # la fila i de inv_t es la solucion de A x = e_i, o sea la columna i de la inversa
    inv_t = np.zeros((n, n))
    for i in range(0, n):
        y = res_tri(L, canonicos[i])
        inv_t[i] = res_tri(U, y, inferior=False)

    return traspuesta(inv_t)


def calculaLDV(A):
    """
    Calcula la factorización LDV de la matriz A, de forma tal que
    A = LDV, con L triangular inferior, D diagonal y V triangular superior.
    En caso de que la matriz no pueda factorizarse, retorna None.
    """
    L, U, _ = calculaLU(A)
    if L is None:
        return None, None, None

    U_t = traspuesta(U)

    V_t, D, _ = calculaLU(U_t)

    V = traspuesta(V_t)

    return L,D,V

def esSDP(A,atoi=1e-10):
    """
    Determina si una matriz es semidefinida positiva.
    """
    if not matricesIguales(A,traspuesta(A),atoi):
        return False

    _,D,_ = calculaLDV(A)
    if D is None:
        return False

    for i in range(0,D.shape[0]):
        if D[i,i] <= atoi:
            return False
    return True




def producto_interno(v1,v2):
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("Las dimensiones de los vectores no coinciden")
    return sum(v1 * v2)

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    n, m = A.shape
    if n != m:
        return None
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    cant_op = 0
    q1 = A[:, 0]
    norma_q1 = norma(q1)
    q1 = q1 / norma_q1
    cant_op += 3 * n - 1
    Q[:, 0] = q1
    R[0, 0] = norma_q1
    for j in range(1, n):
        qj = A[:, j]
        for k in range(j):
            R[k, j] = producto_interno(Q[:, k], qj)
            qj = qj - R[k, j] * Q[:, k]
        R[j, j] = norma(qj)
        qj = qj / R[j, j]
        Q[:, j] = qj
    if retorna_nops:
        return Q, R, cant_op
    else:
        return Q, R


def QR_con_HH(A, tol=1e-12):
    m, n = A.shape

    if m < n:
        return None

    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        alfa = -np.sign(x[0])*norma(x)
        u = x - alfa*np.eye(m-k)[0]
        norma_2 = norma(u)
        if norma_2 > tol:
            u = u/norma(u)
            Hk = calcula_Householder_k(u)
            i_m = np.eye(m)
            h_k_monio = np.eye(m)
            h_k_monio[0:k,0:k] = i_m[0:k,0:k]
            h_k_monio[k:m,k:m] = Hk
            R = h_k_monio @ R
            Q = Q @ traspuesta(h_k_monio)

    return Q, R


def calcula_Householder_k(u):
    n = u.shape[0]
    Hk = np.eye(n)
    for i in range(n):
        for j in range(n):
            Hk[i,j] = Hk[i,j] - 2*u[i]*u[j]
    return Hk

def calculaQR(A,metodo=['GS','RH'],tol=1e-12):
    if metodo == 'GS':
        return QR_con_GS(A,tol)
    elif metodo == 'RH':
        return QR_con_HH(A,tol)
    else:
        return None


#Laboratorio 6


#TODO: usar nuestro arroba en vez de @
def f_A(v,A):
    w_prima = A @ v
    if norma(w_prima) == 0:
        return np.zeros(v.shape[0])
    return normaliza_vector(w_prima)

def f_A2(v,A):
    return f_A(f_A(v,A),A)

def metpot2k(A,tol=1e-15,K=1000):

    v = normaliza_vector(np.random.uniform(0, 1, size=A.shape[0]))

    v_monio = f_A2(v,A)
    e = producto_interno(v_monio,v)
    k = 0
    while (abs(e-1) > tol) and (k < K):
        v = v_monio
        v_monio = f_A2(v,A)
        e = producto_interno(v_monio,v)
        k = k+1
    lamd = v_monio @ A @ v_monio
    return v_monio, lamd, k


def diagRH(A,tol=1e-15,K=1000):
    """ A simetrica tolerancia tol y entero K cantidad de iteraciones"""
    v_monio, lamd, _ = metpot2k(A,tol,K)
    tamanio = A.shape[0]
    e_1 = np.eye(tamanio)[0]
    Hv_1 = calcula_Householder_k((e_1-v_monio))
    if tamanio == 2:
        S = Hv_1
        D = Hv_1 @ A @ Hv_1
        return S,D
    else:
        B = Hv_1@A@(traspuesta(Hv_1))
        A_monio = B[1:,1:]
        S_monio, D_monio = diagRH(A_monio,tol,K)
        D = np.zeros((tamanio,tamanio))
        D[0,0] = lamd
        D[1:,1:] = D_monio
        S_aux = np.eye(tamanio)
        S_aux[1:,1:] = S_monio
        S = Hv_1 @ S_aux
        return S, D