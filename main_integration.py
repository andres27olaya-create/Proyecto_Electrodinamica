import numpy as np
import matplotlib.pyplot as plt

# Importar módulos desarrollados
from carga import generar_distribucion_carga, A0
from Mallado import generar_vertices_poligono_regular, teselar_quadtree
from potencial_cuantico import calcular_potencial_excitacion
from mom_pm_solver import build_mom_matrix

def main():
    print("Iniciando integración Cuántica - MoM (Point Matching)...")
    
    # 1. Parámetros Generales
    Z_PLACA = -10 * A0
    U_PLACA = 0.0  # Placa conectada a tierra
    
    print("\n[1/5] Generando nube de carga cuántica...")
    Xs, Ys, Zs, Rs, dVs = generar_distribucion_carga(
        mask=[0, 1, 1, 1], 
        n_r=40, n_theta=40, n_phi=40, 
        step_r=2, step_t=2, step_p=2
    )
    print(f"  -> Nube generada con {len(Xs)} vóxeles efectivos.")
    
    print("\n[2/5] Teselando placa metálica (Hexágono)...")
    vertices = generar_vertices_poligono_regular(6, r=15*A0)
    cuadrados = teselar_quadtree(
        vertices=vertices,
        profundidad=7,
        min_lado=A0/5,
        h_min=0.5*A0,
        h_max=3.0*A0,
        alpha=1.5
    )
    N_celdas = len(cuadrados)
    print(f"  -> Placa teselada con {N_celdas} celdas cuadradas.")
    
    print("\n[3/5] Calculando Potencial Excitador en la placa...")
    V_exc = calcular_potencial_excitacion(cuadrados, Z_PLACA, Xs, Ys, Zs, Rs, dVs)
    print(f"  -> Rango de V_exc: [{V_exc.min():.2e} V, {V_exc.max():.2e} V]")
    
    print("\n[4/5] Construyendo Matriz MoM y resolviendo sistema...")
    C_matrix = build_mom_matrix(cuadrados, z_placa=Z_PLACA, tresh=10.0)
    
    # Verificación rápida de propiedades de la matriz
    is_symmetric = np.allclose(C_matrix, C_matrix.T, atol=1e-12)
    print(f"  -> Matriz MoM construida. ¿Simétrica?: {is_symmetric}")
    
    # Condición de frontera: V_total = U_placa => V_inducido + V_exc = U_placa => V_inducido = U_placa - V_exc
    # V_inducido = [C] * [alpha]
    e_vector = U_PLACA - V_exc
    
    # Resolviendo para densidad de carga superficial inducida [alpha] (C/m^2)
    alpha_inducida = np.linalg.solve(C_matrix, e_vector)
    print(f"  -> Carga superficial hallada. Rango: [{alpha_inducida.min():.2e} C/m^2, {alpha_inducida.max():.2e} C/m^2]")
    
    print("\n[5/5] Graficando resultados...")
    graficar_resultados(vertices, cuadrados, alpha_inducida)
    print("\nProceso finalizado con éxito.")

def graficar_resultados(vertices, cuadrados, alpha_inducida):
    """
    Grafica la distribución 2D de la carga inducida sobre la placa.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extraer centros para scatter o para colormap
    cx_list = [c[0] for c in cuadrados]
    cy_list = [c[1] for c in cuadrados]
    lados = [c[2] for c in cuadrados]
    
    # Dibujar contorno del polígono
    vx = np.append(vertices[:, 0], vertices[0, 0])
    vy = np.append(vertices[:, 1], vertices[0, 1])
    ax.plot(vx, vy, color='black', linewidth=2, label="Borde Placa")
    
    # Dibujar celdas (opcional) y colorear
    sc = ax.scatter(cx_list, cy_list, c=alpha_inducida, cmap='coolwarm', s=[(l/2e-11)**2 for l in lados], marker='s')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Densidad de Carga Inducida $\\sigma$ ($C/m^2$)')
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Distribución de Carga Inducida sobre Placa Hexagonal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
