from carga2 import generar_distribucion_carga, A0
from Ubi_plano import ubicar_cuadrados_en_plano, ubicar_cuadrados_en_plano_arbitrario, ubicar_vertices_en_plano
from potencial_cuantico import calcular_potencial_excitacion
from Mallado import generar_vertices_poligono_regular, teselar_quadtree, dibujar_resultado
from mom_pm_solver import build_mom_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

if __name__ == "__main__":

    print("=== Calculo distribucion de carga cuántica y visualización 3D ===")
    Xs, Ys, Zs, Rs, dVs = generar_distribucion_carga(mask=[1,0,0,1])

    # --- transparencia basada en |rho| ---
    mag = np.abs(Rs)

    # normalización robusta para que no se aplaste por outliers
    vmin = np.percentile(mag, 5)
    vmax = np.percentile(mag, 95)
    norm = np.clip((mag - vmin) / (vmax - vmin + 1e-30), 0, 1)

    # alpha: bajo -> transparente, alto -> opaco
    alpha_min = 0.03
    alpha_max = 0.95
    alphas = alpha_min + (alpha_max - alpha_min) * norm

    # color por magnitud, transparencia por alpha
    cmap = plt.cm.viridis
    colors = cmap(norm)
    colors[:, 3] = alphas

    # --- gráfica ---
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # pyrefly: ignore [bad-keyword-argument]
    # ax.scatter(Xs, Ys, Zs, c=colors, s=6, marker='o')

    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('z (m)')
    # ax.set_title('Distribución 3D con transparencia según |ρ|')
    #plt.show()

   
    n_lados = 6                 # Número de lados del polígono regular
    radio_figura = 20 * A0      # Radio del polígono regular
    profundidad_maxima = 8     # Máxima profundidad de subdivisión del quadtree
    lado_minimo = A0 / 3
    h_min = 0.5                # Tamaño objetivo del cuadrado en el CENTRO de la figura                               
    h_max = 0.5               # Tamaño objetivo del cuadrado en el BORDE de la figura                                
    alpha = 1.0                 # Factor de transición entre el centro y el borde
  
    vertices = generar_vertices_poligono_regular(n_lados, r=radio_figura)
    cuadrados = teselar_quadtree(
        vertices=vertices,
        profundidad=profundidad_maxima,
        min_lado=lado_minimo,
        h_min=h_min,
        h_max=h_max,
        alpha=alpha
    )
    # dibujar_resultado(vertices, cuadrados)


    plan= "yz"
    offs=25
    cuadrado_3d = ubicar_cuadrados_en_plano(cuadrados, plano=plan, offset=offs)
    
    # # --- Transformar vértices al mismo plano que los cuadrados ---
    vertices_3d = ubicar_vertices_en_plano(vertices, plano=plan, offset=offs)
    
    # --- Gráfica combinada: distribución de carga + vértices en el mismo plano ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar la distribución de carga
    mag = np.abs(Rs)
    vmin = np.percentile(mag, 5)
    vmax = np.percentile(mag, 95)
    norm = np.clip((mag - vmin) / (vmax - vmin + 1e-30), 0, 1)
    
    alpha_min = 0.03
    alpha_max = 0.95
    alphas = alpha_min + (alpha_max - alpha_min) * norm
    
    cmap = plt.cm.viridis
    colors = cmap(norm)
    colors[:, 3] = alphas
    
    # pyrefly: ignore [bad-keyword-argument]
    ax.scatter(Xs, Ys, Zs, c=colors, s=6, marker='o', label='Distribución de carga')
    
    # Graficar los vértices transformados
    ax.plot(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2], 
            'r-o', linewidth=2, markersize=8, label='Vértices de la figura')
    # Cerrar la figura (conectar el último vértice con el primero)
    ax.plot([vertices_3d[-1, 0], vertices_3d[0, 0]], 
            [vertices_3d[-1, 1], vertices_3d[0, 1]], 
            [vertices_3d[-1, 2], vertices_3d[0, 2]], 'r-', linewidth=2)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Distribución 3D de carga con vértices de la figura en plano YZ')
    ax.legend()
    plt.show()
    
    # --- Calcular potencial de excitación en la malla 3D ---
    print("\n=== Calculando potencial de excitación en la malla ===")
    V_exc = calcular_potencial_excitacion(cuadrado_3d, Xs, Ys, Zs, Rs, dVs, batch_size=500)
    
    print(f"\nPotencial de excitación:")
    print(f"  Mínimo: {np.min(V_exc):.6e} V")
    print(f"  Máximo: {np.max(V_exc):.6e} V")
    print(f"  Promedio: {np.mean(V_exc):.6e} V")
    
    # --- Aplicar Método de Momentos (MoM) ---
    print("\n=== Construyendo matriz de impedancia [C] ===")
    C = build_mom_matrix(cuadrado_3d, tresh=10.0)
    print(f"Matriz C de tamaño {C.shape}")
    
    # Suponiendo placa a tierra (potencial = 0)
    # Resolvemos: C @ alpha = -V_exc
    print("\n=== Resolviendo sistema MoM para placa a tierra ===")
    try:
        alpha = np.linalg.solve(C, -V_exc)
        print(f"Densidad superficial inducida (alpha):")
        print(f"  Mínimo: {np.min(alpha):.6e} C/m²")
        print(f"  Máximo: {np.max(alpha):.6e} C/m²")
        print(f"  Promedio: {np.mean(alpha):.6e} C/m²")
        
        # Calcular potencial inducido: V_ind = C @ alpha
        V_ind = C @ alpha
        
        # Potencial total: V_total = V_exc + V_ind
        V_total = V_exc + V_ind
        
        print(f"\nPotencial inducido (V_ind):")
        print(f"  Mínimo: {np.min(V_ind):.6e} V")
        print(f"  Máximo: {np.max(V_ind):.6e} V")
        print(f"  Promedio: {np.mean(V_ind):.6e} V")
        
        print(f"\nPotencial total (V_total = V_exc + V_ind):")
        print(f"  Mínimo: {np.min(V_total):.6e} V")
        print(f"  Máximo: {np.max(V_total):.6e} V")
        print(f"  Promedio: {np.mean(V_total):.6e} V")
        
    except np.linalg.LinAlgError as e:
        print(f"Error al resolver el sistema MoM: {e}")
        alpha = None
        V_ind = None
        V_total = V_exc
    
    # --- Gráfica 2D del potencial de excitación ---
    print("\n=== Generando visualización 2D del potencial de excitación ===")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Extraer coordenadas 2D en el plano yz (u, v locales)
    centros_y_2d = np.array([y for x, y, z, _ in cuadrado_3d])
    centros_z_2d = np.array([z for x, y, z, _ in cuadrado_3d])
    
    # --- Subplot 1: Potencial de excitación ---
    V_norm_exc = np.abs(V_exc)
    V_min_exc = np.min(V_norm_exc)
    V_max_exc = np.max(V_norm_exc)
    
    scatter_exc = ax1.scatter(centros_y_2d, centros_z_2d, c=V_norm_exc, cmap='viridis', 
                              s=50, marker='s', edgecolors='black', linewidth=0.5)
    
    ax1.plot(vertices_3d[:, 1], vertices_3d[:, 2], 
             'r-o', linewidth=2, markersize=10, label='Vértices de la figura')
    ax1.plot([vertices_3d[-1, 1], vertices_3d[0, 1]], 
             [vertices_3d[-1, 2], vertices_3d[0, 2]], 'r-', linewidth=2)
    
    ax1.set_xlabel('y (m)', fontsize=12)
    ax1.set_ylabel('z (m)', fontsize=12)
    ax1.set_title('Potencial de Excitación V_exc (Plano YZ)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    cbar_exc = plt.colorbar(scatter_exc, ax=ax1)
    cbar_exc.set_label('Potencial (V)', rotation=270, labelpad=20)
    
    # --- Subplot 2: Potencial total ---
    if V_total is not None:
        V_norm_total = np.abs(V_total)
        V_min_total = np.min(V_norm_total)
        V_max_total = np.max(V_norm_total)
        
        scatter_total = ax2.scatter(centros_y_2d, centros_z_2d, c=V_norm_total, cmap='plasma', 
                                    s=50, marker='s', edgecolors='black', linewidth=0.5)
        
        ax2.plot(vertices_3d[:, 1], vertices_3d[:, 2], 
                 'r-o', linewidth=2, markersize=10, label='Vértices de la figura')
        ax2.plot([vertices_3d[-1, 1], vertices_3d[0, 1]], 
                 [vertices_3d[-1, 2], vertices_3d[0, 2]], 'r-', linewidth=2)
        
        ax2.set_xlabel('y (m)', fontsize=12)
        ax2.set_ylabel('z (m)', fontsize=12)
        ax2.set_title('Potencial Total V_total (Placa a Tierra)', fontsize=14, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        cbar_total = plt.colorbar(scatter_total, ax=ax2)
        cbar_total.set_label('Potencial (V)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
    
    # --- Gráfica de densidad de carga superficial inducida ---
    if alpha is not None:
        print("\n=== Generando visualización de densidad superficial inducida ===")
        fig, ax = plt.subplots(figsize=(12, 10))

        alpha = np.asarray(alpha, dtype=float)

        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)

        # Centro de color:
        # si hay cambio de signo, usa 0.
        # si no, usa la mediana para forzar contraste visual.
        if alpha_min < 0 < alpha_max:
            alpha_mid = 0.0
        else:
            alpha_mid = np.median(alpha)

        norm = TwoSlopeNorm(vmin=alpha_min, vcenter=alpha_mid, vmax=alpha_max)

        scatter_alpha = ax.scatter(
            centros_y_2d, centros_z_2d,
            c=alpha,
            cmap="seismic",
            norm=norm,
            s=120,
            marker='s',
            edgecolors='black',
            linewidth=0.5
        )

        ax.plot(vertices_3d[:, 1], vertices_3d[:, 2],
                'k-o', linewidth=2, markersize=10, label='Vértices de la figura')
        ax.plot([vertices_3d[-1, 1], vertices_3d[0, 1]],
                [vertices_3d[-1, 2], vertices_3d[0, 2]], 'k-', linewidth=2)

        ax.set_xlabel('y (m)', fontsize=12)
        ax.set_ylabel('z (m)', fontsize=12)
        ax.set_title('Densidad de Carga Superficial Inducida σ (C/m²)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        cbar_alpha = plt.colorbar(scatter_alpha, ax=ax)
        cbar_alpha.set_label('σ (C/m²)', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.show()