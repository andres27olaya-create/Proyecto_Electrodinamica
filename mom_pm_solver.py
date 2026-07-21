import numpy as np

EPSILON_0 = 8.8541878128e-12  # Permitividad del vacío [F/m]

def prim1(u, v, c):
    """
    Primitiva analítica exacta de la integral de 1/R sobre una superficie rectangular,
    según J. R. Mosig.
    """
    term1 = u * np.arcsinh(v / np.sqrt(u**2 + c**2))
    term2 = v * np.arcsinh(u / np.sqrt(v**2 + c**2))
    term3 = c * np.arctan((u * v) / (c * np.sqrt(u**2 + v**2 + c**2)))
    return term1 + term2 - term3

def discr_gf(xc, yc, zc, a, b):
    """
    Evalúa la Función de Green Discreta en el centro de una celda de prueba a partir
    de una celda fuente de dimensiones a x b.

    xc, yc, zc: Distancias relativas entre centro de prueba y centro fuente.
    a, b: Dimensiones de la celda fuente.
    """
    eps = 2.2204e-16
    zc = zc + eps  # Prevenir indeterminaciones 0/0

    # Límites de integración respecto al centro de la celda fuente
    u1, u2 = xc - a / 2.0, xc + a / 2.0
    v1, v2 = yc - b / 2.0, yc + b / 2.0

    # Combinación de la primitiva evaluada en los 4 vértices
    integral = (prim1(u2, v2, zc) 
              - prim1(u1, v2, zc) 
              - prim1(u2, v1, zc) 
              + prim1(u1, v1, zc))
              
    return integral / (a * b)

def stat_mom_element(xb, yb, zb, xt, yt, zt, a, b, tresh=10.0):
    """
    Calcula el elemento de matriz MoM entre una celda base y una de prueba.
    
    xb, yb, zb: Coordenadas del centro de la celda fuente (base).
    xt, yt, zt: Coordenadas del centro de la celda de prueba (test).
    a, b: Dimensiones de la celda fuente.
    tresh: Umbral para aproximación de campo lejano.
    """
    rh = np.sqrt((xt - xb)**2 + (yt - yb)**2 + (zt - zb)**2)
    max_dim = max(a, b)
    
    if rh > tresh * max_dim:
        # Aproximación rápida 1/r para campo lejano
        val = 1.0 / rh
    else:
        # Integración exacta
        xc, yc, zc = xt - xb, yt - yb, zt - zb
        val = discr_gf(xc, yc, zc, a, b)
        
    return (1.0 / (4.0 * np.pi * EPSILON_0)) * val

def build_mom_matrix(cuadrados, z_placa=0.0, tresh=10.0):
    """
    Construye la matriz de impedancia estática [C] para el Método de Momentos.
    
    cuadrados: Lista de celdas (cx, cy, lado).
    z_placa: Elevación z de la placa.
    tresh: Umbral para la aproximación de campo lejano.
    
    Retorna
    -------
    C : ndarray
        Matriz de impedancia de tamaño NxN.
    """
    N = len(cuadrados)
    C = np.zeros((N, N))
    
    for i in range(N):
        xt, yt, lt = cuadrados[i]
        zt = z_placa
        for j in range(N):
            xb, yb, lb = cuadrados[j]
            zb = z_placa
            # Como las celdas son cuadradas, a = b = lb
            C[i, j] = stat_mom_element(xb, yb, zb, xt, yt, zt, lb, lb, tresh)
            
    return C
