import numpy as np
from scipy.spatial.distance import cdist

# Constantes físicas
EPSILON_0 = 8.8541878128e-12  # Permitividad del vacío [F/m]

def calcular_potencial_excitacion(cuadrados, z_placa, Xs, Ys, Zs, Rs, dVs):
    """
    Calcula el potencial eléctrico generado por la nube de carga cuántica 
    sobre los centros de cada elemento de prueba en la placa.

    Parámetros
    ----------
    cuadrados : list of tuples
        Lista de celdas cuadradas, cada una definida como (cx, cy, lado).
    z_placa : float
        Coordenada z de la placa metálica.
    Xs, Ys, Zs : ndarray
        Coordenadas de los puntos de la nube de carga.
    Rs : ndarray
        Densidad volumétrica de carga en cada punto de la nube (rho_k).
    dVs : ndarray
        Diferencial de volumen asociado a cada punto de la nube.

    Retorna
    -------
    V_exc : ndarray
        Vector de potenciales evaluados en los centros de cada celda.
    """
    # Coordenadas de evaluación (centros de los cuadrados)
    r_eval = np.array([[cx, cy, z_placa] for cx, cy, _ in cuadrados])
    
    # Coordenadas de la nube de carga
    r_charge = np.column_stack((Xs, Ys, Zs))
    
    # Carga total de cada vóxel q_k = rho_k * dV_k
    q_k = Rs * dVs
    
    # Calculamos la matriz de distancias entre puntos de evaluación y cargas
    # dists tendrá forma (N_eval, N_charge)
    dists = cdist(r_eval, r_charge)
    
    # Prevenir divisiones por cero sumando un epsilon
    dists = np.maximum(dists, 1e-15)
    
    # Potencial V = 1 / (4*pi*eps0) * sum(q_k / r_k)
    # Sumamos a lo largo del eje de las cargas (axis=1)
    V_exc = (1.0 / (4.0 * np.pi * EPSILON_0)) * np.sum(q_k / dists, axis=1)
    
    return V_exc
