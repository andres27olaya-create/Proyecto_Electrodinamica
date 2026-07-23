#entonces la idea general es poder modelar una distribucion de carga que tenga, monopolo, dipolo, cuadrupolo y octupolo.
# esto para poder hacer un analisis de la distribucion de carga y ver como se comporta el campo electrico en cada caso.
# aunque la idea sea unicamente el octupolo, se hace un analisis de los otros casos para poder ver como se comporta el campo electrico en cada caso.
#creo que la mejor forma es hacerlo con armonicos esfericos y analizar como se comporta el campo electrico en cada caso, para poder ver como se comporta el campo electrico en cada caso.
#cuando hay varios casos simultaneos o otros no
# usando la expansion multipolar para el potencial

import numpy as np
from math import factorial
from scipy.special import eval_genlaguerre, eval_legendre
import matplotlib.pyplot as plt


# Constantes físicas
A0 = 1     # radio de Bohr [m]
E_CHARGE = 1  # carga elemental [C]
EPSILON_0 = 1/(4*np.pi)  # Permitividad del vacío [F/m]


def R_nl(n, l, r, a0=A0):
    """
    Función radial hidrogenoide normalizada R_nl(r).

    Parámetros
    ----------
    n : int
        Número cuántico principal
    l : int
        Número cuántico orbital
    r : array_like
        Radio(s)
    a0 : float
        Radio de Bohr

    Retorna
    -------
    R : ndarray
        Valor de R_nl(r)
    """
    if not (0 <= l < n):
        raise ValueError(f"Debe cumplirse 0 <= l < n. Recibido: n={n}, l={l}")

    r = np.asarray(r, dtype=float)
    rho = 2.0 * r / (n * a0)

    prefactor = (2.0 / (n * a0))**1.5 * np.sqrt(
        factorial(n - l - 1) / (2.0 * n * factorial(n + l))
    )

    return prefactor * np.exp(-rho / 2.0) * (rho**l) * eval_genlaguerre(n - l - 1, 2*l + 1, rho)


def psi_m0_superposition(n, r, theta, active_mask, coeffs=None, a0=A0, use_Yl0_norm=True):
    """
    Construye psi(r,theta) = sum_l c_l R_nl(r) P_l(cos(theta))
    para m=0, seleccionando los l activos con un arreglo tipo [1,0,0,1].

    Parámetros
    ----------
    n : int
        Número cuántico principal fijo
    r : array_like
        Radio(s)
    theta : array_like
        Ángulo(s) polar(es)
    active_mask : list o array
        Ejemplo: [1,0,0,1] -> activa l=0 y l=3
    coeffs : list o array, opcional
        Coeficientes complejos c_l para cada l.
        Si no se da, se toma c_l = 1 para todos los l activos.
    a0 : float
        Radio de Bohr
    use_Yl0_norm : bool
        Si True usa el factor sqrt((2l+1)/(4pi)) correcto.
        Si False usa solo P_l(cos theta).

    Retorna
    -------
    psi : ndarray complejo
        Función de onda superpuesta
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    active_mask = np.asarray(active_mask, dtype=int)

    # Se asegura de que r y theta puedan "broadcast"
    shape = np.broadcast(r, theta).shape
    psi = np.zeros(shape, dtype=complex)

    if coeffs is None:
        coeffs = np.ones(len(active_mask), dtype=complex)
    else:
        coeffs = np.asarray(coeffs, dtype=complex)
        if len(coeffs) != len(active_mask):
            raise ValueError("coeffs debe tener la misma longitud que active_mask")

    cos_theta = np.cos(theta)

    for l, active in enumerate(active_mask):
        if not active:
            continue

        if l >= n:
            raise ValueError(f"l={l} no es permitido para n={n}. Debe cumplirse l < n.")

        radial = R_nl(n, l, r, a0=a0)
        angular = eval_legendre(l, cos_theta)

        if use_Yl0_norm:
            angular *= np.sqrt((2*l + 1) / (4.0 * np.pi))

        psi += coeffs[l] * radial * angular

    return psi


def electron_density(psi):
    """
    Densidad de carga electrónica:
        rho = -e |psi|^2
    """
    return -E_CHARGE * np.abs(psi)**2

def generar_distribucion_carga(
    mask,
    n_x: int = 50,
    n_y: int = 50,
    n_z: int = 50,
    max_r_factor: float = 25,
    density_threshold: float = 1e-8,
) -> tuple:
    """
    Genera una distribución de carga discretizada en una malla cartesiana uniforme.
    Esto evita los huecos visuales y la no-uniformidad inherente de (r, theta, phi).

    Parámetros
    ----------
    mask : list
        Máscara de orbitales activos
    n_x, n_y, n_z : int
        Número de puntos en cada dimensión (reducido de 60 a 50 para menos vóxeles)
    max_r_factor : float
        Factor para el radio máximo (reducido de 40 a 25 A0 ya que la densidad tiende a 0)
    density_threshold : float
        Umbral mínimo de densidad de carga para incluir un vóxel.
        Los vóxeles con |rho| < threshold se descartan.

    Returns
    -------
    Xs, Ys, Zs : ndarray, shape (M,)
        Coordenadas cartesianas de los vóxeles.
    Rs : ndarray, shape (M,)
        Densidad de carga en cada vóxel.
    dVs : ndarray, shape (M,)
        Diferencial de volumen constante de cada vóxel.
    """
    n = len(mask)

    # Cubo que contiene a la distribución
    L = max_r_factor * A0

    x = np.linspace(-L, L, n_x)
    y = np.linspace(-L, L, n_y)
    z = np.linspace(-L, L, n_z)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Coordenadas esféricas derivadas
    R = np.sqrt(X**2 + Y**2 + Z**2)

    Theta = np.zeros_like(R)
    nz = R > 1e-300
    Theta[nz] = np.arccos(np.clip(Z[nz] / R[nz], -1.0, 1.0))

    # La densidad cuántica depende de r y theta; phi no entra si m=0
    psi = psi_m0_superposition(
        n=n,
        r=R,
        theta=Theta,
        active_mask=mask
    )
    rho = electron_density(psi)

    # Volumen uniforme por celda
    dx = x[1] - x[0] if n_x > 1 else 0.0
    dy = y[1] - y[0] if n_y > 1 else 0.0
    dz = z[1] - z[0] if n_z > 1 else 0.0
    dV = dx * dy * dz

    dVs = np.full(R.shape, dV, dtype=float)

    # Aplanar salida
    Xs = X.ravel()
    Ys = Y.ravel()
    Zs = Z.ravel()
    Rs = rho.ravel()
    dVs = dVs.ravel()

    # --- OPTIMIZACIÓN: Filtrar vóxeles con densidad muy baja ---
    mask_nonzero = np.abs(Rs) > density_threshold
    Xs = Xs[mask_nonzero]
    Ys = Ys[mask_nonzero]
    Zs = Zs[mask_nonzero]
    Rs = Rs[mask_nonzero]
    dVs = dVs[mask_nonzero]

    return Xs, Ys, Zs, Rs, dVs

if __name__ == "__main__":
    Xs, Ys, Zs, Rs, dVs = generar_distribucion_carga(mask=[0,0 , 0, 1])

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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # pyrefly: ignore [bad-keyword-argument]
    ax.scatter(Xs, Ys, Zs, c=colors, s=6, marker='o')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Distribución 3D con transparencia según |ρ|')

    plt.show()