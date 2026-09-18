#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np
from queue import Queue as Cola

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    # Entonces tengo que agarrar cada columna, dividirla por el pivot
    # y restarsela a np.eye y eso va a ser mi L
    # Antes de pasar a la siguiente iteracion tengo que hacer la multi-
    # plicacion de A para ir sacando U. Creo que teoricamente podes hacer
    # lo de los calumnas de una y despues calculas U.
    # ASUMO QUE LAS MATRICES SON CUADRADAS. aunque podes hacer LU de una A
    # no cuadrada. Ja lo dice el primer if

    k = len(A[0])
    I = np.eye(k)
    # Me podria guardar los M tambien 
    Ms = Cola()
    L = np.eye(k)
    for col in range(k - 1):
        pivot = Ac[col][col]
        if pivot == 0:
            return None
        aux = np.zeros((k,k))
        
        for fila in range(col + 1, m):
            factor = Ac[fila][col]/pivot
            cant_op += 1
            
            aux[fila][col] = factor
            L[fila][col] = factor
            
        M = I - aux
        cant_op += 1
        Ms.put(M)
        # Actualizamos Ac
        Ac = M @ Ac
        # CONTAR TODAS LAS OPERACIONES QUE HACE MULTIPLICACIONES
        # Me falta calcular la cantidad de operaciones de @
        # Despues reemplazo @ por multiplicacion matricial
    U = Ac
                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op
    
def restri(L, b):

    # directamente trabajo con b
    # y la L me sirve para hacer las divisiones
    # guardo las x en un vector 
    return

def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    
