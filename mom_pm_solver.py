import numpy as np
from carga2 import EPSILON_0

K_COULOMB = 1.0 / (4.0 * np.pi * EPSILON_0)


def prim1(u, v, c):
    """
    Primitiva analítica exacta de la integral de 1/R sobre una superficie rectangular,
    según J. R. Mosig.

    Soporta escalares o arrays.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    c = np.asarray(c, dtype=float)

    term1 = u * np.arcsinh(v / np.sqrt(u*u + c*c))
    term2 = v * np.arcsinh(u / np.sqrt(v*v + c*c))
    term3 = c * np.arctan((u * v) / (c * np.sqrt(u*u + v*v + c*c)))

    return term1 + term2 - term3


def discr_gf(xc, yc, zc, a, b):
    """
    Función de Green discreta promediada sobre una celda rectangular fuente.
    Soporta escalares o arrays.
    """
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    zc = np.asarray(zc, dtype=float)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # evita 0/0 en el caso singular
    zc = zc + np.finfo(float).eps

    u1, u2 = xc - 0.5 * a, xc + 0.5 * a
    v1, v2 = yc - 0.5 * b, yc + 0.5 * b

    integral = (
        prim1(u2, v2, zc)
        - prim1(u1, v2, zc)
        - prim1(u2, v1, zc)
        + prim1(u1, v1, zc)
    )

    return integral / (a * b)


def build_mom_matrix(cuadrados_3d, tresh=10.0, assume_symmetric=True):
    """
    Construye la matriz MoM estática [C].

    cuadrados_3d: lista/array de celdas (x, y, z, lado)
    tresh: umbral para usar aproximación 1/r en campo lejano
    assume_symmetric:
        True  -> calcula solo la mitad superior y refleja (más rápido)
        False -> calcula toda la matriz sin imponer simetría
    """
    coords = np.asarray(cuadrados_3d, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("cuadrados_3d debe tener forma (N, 4) con columnas (x, y, z, lado).")

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    lado = coords[:, 3]

    N = len(coords)
    C = np.zeros((N, N), dtype=float)

    if assume_symmetric:
        for i in range(N):
            # Distancias del punto de prueba i a todas las fuentes j
            dx = np.abs(x[i] - x)
            dy = np.abs(y[i] - y)
            dz = np.abs(z[i] - z)

            rh = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Aproximación de campo lejano
            row = np.divide(
                K_COULOMB,
                rh,
                out=np.zeros_like(rh),
                where=rh > 0.0
            )

            # Celdas cercanas: integral exacta
            near = rh <= (tresh * lado)
            if np.any(near):
                row[near] = K_COULOMB * discr_gf(
                    dx[near], dy[near], dz[near],
                    lado[near], lado[near]
                )

            # imponer simetría
            C[i, :] = row
            C[:, i] = row

    else:
        for i in range(N):
            xt, yt, zt = x[i], y[i], z[i]

            dx = np.abs(xt - x)
            dy = np.abs(yt - y)
            dz = np.abs(zt - z)

            rh = np.sqrt(dx*dx + dy*dy + dz*dz)

            row = np.divide(
                K_COULOMB,
                rh,
                out=np.zeros_like(rh),
                where=rh > 0.0
            )

            near = rh <= (tresh * lado)
            if np.any(near):
                row[near] = K_COULOMB * discr_gf(
                    dx[near], dy[near], dz[near],
                    lado[near], lado[near]
                )

            C[i, :] = row

    return C