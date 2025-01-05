import numpy as np

def calcular_proyeccion(M):
    """
    Calcula la matriz P de la ecuación:
    P = I - M^T * ((M * M^T)^-1) * M
    
    Parameters:
    M (numpy.ndarray): Matriz de entrada.
    
    Returns:
    numpy.ndarray: Matriz P resultante.
    """
    # Verificar que la matriz no sea vacía
    if M.size == 0:
        raise ValueError("La matriz M no puede estar vacía.")
    
    # Calcular dimensiones
    m, n = M.shape

    # Matriz identidad de tamaño n x n
    I = np.eye(n)

    # Transpuesta de M
    M_T = M.T

    # Producto M * M^T
    M_M_T = M @ M_T

    # Inversa de (M * M^T)
    try:
        M_M_T_inv = np.linalg.inv(M_M_T)
    except np.linalg.LinAlgError:
        raise ValueError("La matriz (M * M^T) no es invertible.")

    # Calcular P
    P = I - M_T @ M_M_T_inv @ M

    return P

def comprobar_multiplicacion(M, P):
    """
    Comprueba la multiplicación M * P para verificar si resulta en la matriz cero.
    
    Parameters:
    M (numpy.ndarray): Matriz de entrada.
    P (numpy.ndarray): Matriz de proyección.
    
    Returns:
    bool: True si M * P es aproximadamente la matriz cero, False en caso contrario.
    """
    # Producto M * P
    MP = M @ P

    # Comprobar si el resultado es cercano a la matriz cero
    es_cero = np.allclose(MP, np.zeros_like(MP), atol=1e-10)

    return es_cero, MP


# Ejemplo de uso
M = np.array([[0.742, -0.837, -0.483],
              [0.612, 0.837,  0.483],
              [0,     0.958,  0.092]])

P = calcular_proyeccion(M)

# Comprobar la multiplicación
es_cero, MP = comprobar_multiplicacion(M, P)

print("Matriz M:")
print(M)
print("\nMatriz P:")
print(P)
print("\nMatriz M * P:")
print(MP)
print(f"\n¿Es M * P aproximadamente la matriz cero? {'Si' if es_cero else 'No'}")
