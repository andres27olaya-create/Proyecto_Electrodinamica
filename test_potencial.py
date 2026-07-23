from carga2 import generar_distribucion_carga, A0
from Ubi_plano import ubicar_cuadrados_en_plano, ubicar_vertices_en_plano
from potencial_cuantico import calcular_potencial_excitacion
from Mallado import generar_vertices_poligono_regular, teselar_quadtree
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
    V_exc = calcular_potencial_excitacion(cuadrado_3d, Xs, Ys, Zs, Rs, dVs, batch_size=1000)

    print(f"\nResultados del cálculo:")
    print(f"  Potencial mínimo: {np.min(V_exc):.6e} V")
    print(f"  Potencial máximo: {np.max(V_exc):.6e} V")
    print(f"  Potencial promedio: {np.mean(V_exc):.6e} V")
    print(f"  Desviación estándar: {np.std(V_exc):.6e} V")
    print("\n✓ Cálculo completado exitosamente")
