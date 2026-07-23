from carga2 import generar_distribucion_carga, A0
from Ubi_plano import ubicar_cuadrados_en_plano, ubicar_vertices_en_plano
from potencial_cuantico import calcular_potencial_excitacion
from Mallado import generar_vertices_poligono_regular, teselar_quadtree
import numpy as np

if __name__ == "__main__":

    print("=== PRUEBA DE OPTIMIZACIÓN ===\n")
    
    print("1. Generando distribución de carga optimizada...")
    Xs, Ys, Zs, Rs, dVs = generar_distribucion_carga(
        mask=[0, 0, 0, 1],
        n_x=50,
        n_y=50,
        n_z=50,
        max_r_factor=25,
        density_threshold=1e-8
    )
    print(f"   ✓ Vóxeles de carga generados: {len(Xs)}")
    print(f"   Reducción: 216000 → {len(Xs)} vóxeles")

    print("\n2. Generando malla de cuadrados...")
    n_lados = 6
    radio_figura = 20 * A0
    vertices = generar_vertices_poligono_regular(n_lados, r=radio_figura)
    cuadrados = teselar_quadtree(
        vertices=vertices,
        profundidad=8,
        min_lado=A0 / 3,
        h_min=0.5,
        h_max=0.5,
        alpha=1.0
    )
    print(f"   ✓ Cuadrados generados: {len(cuadrados)}")

    print("\n3. Transformando a plano 3D...")
    cuadrado_3d = ubicar_cuadrados_en_plano(cuadrados, plano="yz", offset=-20.0)
    vertices_3d = ubicar_vertices_en_plano(vertices, plano="yz", offset=-20.0)
    print(f"   ✓ Cuadrados 3D: {len(cuadrado_3d)}")

    print("\n4. Calculando potencial de excitación...")
    print(f"   Matriz de distancias aproximada: {len(cuadrado_3d)} × {len(Xs)}")
    print(f"   Tamaño en memoria (si sin lotes): ~{len(cuadrado_3d) * len(Xs) * 8 / 1e9:.1f} GB")
    
    V_exc = calcular_potencial_excitacion(cuadrado_3d, Xs, Ys, Zs, Rs, dVs, batch_size=1000)

    print(f"\n5. Resultados del potencial:")
    print(f"   Potencial mínimo: {np.min(V_exc):.6e} V")
    print(f"   Potencial máximo: {np.max(V_exc):.6e} V")
    print(f"   Potencial promedio: {np.mean(V_exc):.6e} V")
    print(f"   Desviación estándar: {np.std(V_exc):.6e} V")
    
    print("\n✓ Cálculo completado exitosamente sin errores de memoria!")
