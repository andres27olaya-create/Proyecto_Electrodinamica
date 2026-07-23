from carga2 import generar_distribucion_carga, A0
from Ubi_plano import ubicar_cuadrados_en_plano, ubicar_vertices_en_plano
from potencial_cuantico import calcular_potencial_excitacion
from Mallado import generar_vertices_poligono_regular, teselar_quadtree
from mom_pm_solver import build_mom_matrix
import numpy as np

if __name__ == "__main__":

    print("=== Generando distribución de carga ===")
    Xs, Ys, Zs, Rs, dVs = generar_distribucion_carga(mask=[0, 0, 0, 1])
    print(f"Puntos de carga: {len(Xs)}")

    print("\n=== Generando malla de cuadrados ===")
    n_lados = 6
    radio_figura = 20 * A0
    profundidad_maxima = 8
    lado_minimo = A0 / 3
    h_min = 0.5
    h_max = 0.5
    alpha = 1.0

    vertices = generar_vertices_poligono_regular(n_lados, r=radio_figura)
    cuadrados = teselar_quadtree(
        vertices=vertices,
        profundidad=profundidad_maxima,
        min_lado=lado_minimo,
        h_min=h_min,
        h_max=h_max,
        alpha=alpha
    )
    print(f"Cuadrados generados: {len(cuadrados)}")

    print("\n=== Transformando a plano 3D ===")
    cuadrado_3d = ubicar_cuadrados_en_plano(cuadrados, plano="yz", offset=-20.0)
    vertices_3d = ubicar_vertices_en_plano(vertices, plano="yz", offset=-20.0)
    print(f"Cuadrados 3D: {len(cuadrado_3d)}")

    print("\n=== Calculando potencial de excitación (procesando en lotes) ===")
    V_exc = calcular_potencial_excitacion(cuadrado_3d, Xs, Ys, Zs, Rs, dVs, batch_size=500)

    print(f"\nPotencial de excitación:")
    print(f"  Mínimo: {np.min(V_exc):.6e} V")
    print(f"  Máximo: {np.max(V_exc):.6e} V")
    print(f"  Promedio: {np.mean(V_exc):.6e} V")
    
    print("\n=== Construyendo matriz de impedancia [C] ===")
    C = build_mom_matrix(cuadrado_3d, tresh=10.0)
    print(f"Matriz C de tamaño {C.shape}")
    print(f"Número de condición: {np.linalg.cond(C):.6e}")
    
    print("\n=== Resolviendo sistema MoM (C @ alpha = -V_exc) ===")
    try:
        alpha = np.linalg.solve(C, -V_exc)
        print(f"✓ Sistema resuelto exitosamente")
        
        print(f"\nDensidad superficial inducida (alpha):")
        print(f"  Mínimo: {np.min(alpha):.6e} C/m²")
        print(f"  Máximo: {np.max(alpha):.6e} C/m²")
        print(f"  Promedio: {np.mean(alpha):.6e} C/m²")
        
        print(f"\nCalculando potencial inducido (V_ind = C @ alpha) ...")
        V_ind = C @ alpha
        
        print(f"V_ind:")
        print(f"  Mínimo: {np.min(V_ind):.6e} V")
        print(f"  Máximo: {np.max(V_ind):.6e} V")
        print(f"  Promedio: {np.mean(V_ind):.6e} V")
        
        V_total = V_exc + V_ind
        print(f"\nPotencial total (V_total = V_exc + V_ind):")
        print(f"  Mínimo: {np.min(V_total):.6e} V")
        print(f"  Máximo: {np.max(V_total):.6e} V")
        print(f"  Promedio: {np.mean(V_total):.6e} V")
        
        print(f"\n✓ Método de Momentos completado exitosamente")
        
    except np.linalg.LinAlgError as e:
        print(f"✗ Error al resolver el sistema MoM: {e}")
