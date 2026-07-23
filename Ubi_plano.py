import numpy as np
import matplotlib.pyplot as plt


# =========================
# UBICAR LA MALLA EN 3D
# =========================

def ubicar_cuadrados_en_plano(cuadrados, plano="xy", offset=0.0):
    """
    Convierte una lista de cuadrados 2D (cx, cy, lado) -- generados con tu
    quadtree tal como está, en coordenadas locales (u, v) -- a coordenadas
    3D (x, y, z), insertándolos en el plano cartesiano elegido.

    Parámetros
    ----------
    cuadrados : list of (cx, cy, lado)
        Salida de teselar_quadtree, en el sistema local (u, v).
    plano : str
        'xy', 'xz' o 'yz'. Define qué par de ejes ocupan (u, v)
        y cuál queda fijo en `offset`.
    offset : float
        Valor constante del eje que queda fijo (ej: z_placa si plano='xy').

    Retorna
    -------
    cuadrados_3d : list of (x, y, z, lado)
        Centros de los cuadrados ya en 3D.
    """
    mapeos = {
        # (u,v) -> (x,y,z); el eje que no aparece queda fijo en `offset`
        "xy": lambda u, v: (u, v, offset),
        "xz": lambda u, v: (u, offset, v),
        "yz": lambda u, v: (offset, u, v),
    }
    if plano not in mapeos:
        raise ValueError("plano debe ser 'xy', 'xz' o 'yz'")

    f = mapeos[plano]
    cuadrados_3d = [(*f(cx, cy), lado) for cx, cy, lado in cuadrados]
    return cuadrados_3d


def base_ortonormal_desde_normal(normal):
    """
    Dado un vector normal (no necesariamente unitario), construye una base
    ortonormal (u_hat, v_hat, n_hat) para poder ubicar una malla 2D en un
    plano arbitrario (no solo xy/xz/yz).
    """
    n_hat = np.asarray(normal, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    # vector auxiliar no paralelo a n_hat, para construir u_hat con Gram-Schmidt
    aux = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(aux, n_hat)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])

    u_hat = aux - np.dot(aux, n_hat) * n_hat
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(n_hat, u_hat)

    return u_hat, v_hat, n_hat


def ubicar_cuadrados_en_plano_arbitrario(cuadrados, punto_origen, normal):
    """
    Igual que ubicar_cuadrados_en_plano, pero para un plano arbitrario en el
    espacio, definido por un punto de paso (punto_origen) y un vector normal.

    Cada centro local (cx, cy) se mapea a:
        r_3d = punto_origen + cx * u_hat + cy * v_hat

    Parámetros
    ----------
    cuadrados : list of (cx, cy, lado)
    punto_origen : array_like, shape (3,)
        Punto del plano que actúa como origen del sistema (u, v).
    normal : array_like, shape (3,)
        Vector normal al plano (define su orientación).

    Retorna
    -------
    cuadrados_3d : list of (x, y, z, lado)
    """
    u_hat, v_hat, n_hat = base_ortonormal_desde_normal(normal)
    p0 = np.asarray(punto_origen, dtype=float)

    cuadrados_3d = []
    for cx, cy, lado in cuadrados:
        r = p0 + cx * u_hat + cy * v_hat
        cuadrados_3d.append((r[0], r[1], r[2], lado))
    return cuadrados_3d


# =========================
# UBICAR VÉRTICES EN 3D
# =========================

def ubicar_vertices_en_plano(vertices, plano="xy", offset=0.0):
    """
    Convierte los vértices 2D (u, v) de un polígono a coordenadas 3D (x, y, z),
    insertándolos en el plano cartesiano elegido.

    Parámetros
    ----------
    vertices : array_like, shape (n, 2)
        Coordenadas 2D de los vértices en el sistema local (u, v).
    plano : str
        'xy', 'xz' o 'yz'. Define qué par de ejes ocupan (u, v)
        y cuál queda fijo en `offset`.
    offset : float
        Valor constante del eje que queda fijo.

    Retorna
    -------
    vertices_3d : ndarray, shape (n, 3)
        Coordenadas 3D de los vértices.
    """
    mapeos = {
        "xy": lambda u, v: (u, v, offset),
        "xz": lambda u, v: (u, offset, v),
        "yz": lambda u, v: (offset, u, v),
    }
    if plano not in mapeos:
        raise ValueError("plano debe ser 'xy', 'xz' o 'yz'")

    f = mapeos[plano]
    vertices = np.asarray(vertices, dtype=float)
    vertices_3d = np.array([f(u, v) for u, v in vertices])
    return vertices_3d


def ubicar_vertices_en_plano_arbitrario(vertices, punto_origen, normal):
    """
    Convierte los vértices 2D de un polígono a coordenadas 3D, ubicándolos
    en un plano arbitrario definido por un punto de paso y un vector normal.

    Cada vértice local (u, v) se mapea a:
        r_3d = punto_origen + u * u_hat + v * v_hat

    Parámetros
    ----------
    vertices : array_like, shape (n, 2)
        Coordenadas 2D de los vértices.
    punto_origen : array_like, shape (3,)
        Punto del plano que actúa como origen del sistema (u, v).
    normal : array_like, shape (3,)
        Vector normal al plano (define su orientación).

    Retorna
    -------
    vertices_3d : ndarray, shape (n, 3)
        Coordenadas 3D de los vértices.
    """
    u_hat, v_hat, n_hat = base_ortonormal_desde_normal(normal)
    p0 = np.asarray(punto_origen, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    vertices_3d = np.array([p0 + u * u_hat + v * v_hat for u, v in vertices])
    return vertices_3d


# =========================
# DEMO: colocar la misma malla en xy, xz, yz y un plano inclinado
# =========================

def malla_cuadrada_simple(L=5.0, n=8):
    """Genera una malla 2D uniforme de cuadrados (cx, cy, lado), solo para el demo."""
    ejes = np.linspace(-L, L, n)
    lado = ejes[1] - ejes[0]
    return [(cx, cy, lado) for cx in ejes for cy in ejes]


if __name__ == "__main__":
    cuadrados_2d = malla_cuadrada_simple(L=5.0, n=6)

    c_xy = ubicar_cuadrados_en_plano(cuadrados_2d, plano="xy", offset=0.0)
    c_xz = ubicar_cuadrados_en_plano(cuadrados_2d, plano="xz", offset=3.0)
    c_yz = ubicar_cuadrados_en_plano(cuadrados_2d, plano="yz", offset=-3.0)

    # plano inclinado: pasa por (0,0,6) con normal (1,1,1)
    c_inclinado = ubicar_cuadrados_en_plano_arbitrario(
        cuadrados_2d, punto_origen=(0, 0, 6), normal=(1, 1, 1)
    )

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for cuadrados_3d, color, etiqueta in [
        (c_xy, "tab:blue", "plano xy (z=0)"),
        (c_xz, "tab:red", "plano xz (y=3)"),
        (c_yz, "tab:green", "plano yz (x=-3)"),
        (c_inclinado, "tab:purple", "plano inclinado (normal 1,1,1)"),
    ]:
        xs = [c[0] for c in cuadrados_3d]
        ys = [c[1] for c in cuadrados_3d]
        zs = [c[2] for c in cuadrados_3d]
        ax.scatter(xs, ys, zs, s=8, color=color, label=etiqueta)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Misma malla ubicada en distintos planos")
    ax.legend()
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/mallas_en_planos.png", dpi=150)
    print("Listo.")