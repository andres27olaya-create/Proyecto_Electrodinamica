import numpy as np
from scipy.spatial.distance import cdist

from carga2 import EPSILON_0

def calcular_potencial_excitacion(cuadrados_3d, Xs, Ys, Zs, Rs, dVs, batch_size=500):
    """
    Calcula el potencial eléctrico generado por la nube de carga cuántica 
    sobre los centros de cada elemento de prueba en la malla 3D.
    
    Procesa en lotes para evitar problemas de memoria con matrices grandes.

    Parámetros
    ----------
    cuadrados_3d : list of tuples
        Lista de celdas cuadradas transformadas a 3D, cada una definida como (x, y, z, lado).
    Xs, Ys, Zs : ndarray
        Coordenadas de los puntos de la nube de carga.
    Rs : ndarray
        Densidad volumétrica de carga en cada punto de la nube (rho_k).
    dVs : ndarray
        Diferencial de volumen asociado a cada punto de la nube.
    batch_size : int
        Número de puntos de evaluación a procesar en cada lote.

    Retorna
    -------
    V_exc : ndarray
        Vector de potenciales evaluados en los centros de cada celda.
    """
    # Coordenadas de evaluación (centros de los cuadrados 3D)
    r_eval = np.array([[x, y, z] for x, y, z, _ in cuadrados_3d], dtype=np.float32)
    
    # Coordenadas de la nube de carga
    r_charge = np.column_stack((Xs, Ys, Zs)).astype(np.float32)
    
    # Carga total de cada vóxel q_k = rho_k * dV_k
    q_k = (Rs * dVs).astype(np.float32)
    
    n_eval = len(r_eval)
    V_exc = np.zeros(n_eval, dtype=np.float64)
    
    print(f"Calculando potencial para {n_eval} puntos de evaluación en lotes de {batch_size}...")
    
    # Procesar en lotes para evitar problemas de memoria
    for i in range(0, n_eval, batch_size):
        end_i = min(i + batch_size, n_eval)
        
        # Calcular distancias para este lote
        dists = cdist(r_eval[i:end_i], r_charge)
        
        # Prevenir divisiones por cero
        dists = np.maximum(dists, 1e-15)
        
        # Potencial V = 1 / (4*pi*eps0) * sum(q_k / r_k)
        V_exc[i:end_i] = (1.0 / (4.0 * np.pi * EPSILON_0)) * np.sum(q_k / dists, axis=1)
        
        # Progreso
        if (end_i - i) % batch_size == 0 or end_i == n_eval:
            print(f"  Procesados {end_i}/{n_eval} puntos...")
    
    return V_exc
