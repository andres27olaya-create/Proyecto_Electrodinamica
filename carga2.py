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
A0 = 5.29177210903e-11      # radio de Bohr [m]
E_CHARGE = 1.602176634e-19  # carga elemental [C]


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

def _average_pool_3d(arr: np.ndarray, pool_r: int, pool_t: int, pool_p: int) -> np.ndarray:
    """
    Reduce una malla 3D por Average Pooling (promediado por bloques).

    Recorta el array para que sea divisible por los factores de pooling,
    luego reshape en bloques y promedia sobre las dimensiones del bloque.

    Parameters
    ----------
    arr : ndarray, shape (Nr, Nt, Np)
        Array 3D a reducir.
    pool_r, pool_t, pool_p : int
        Factores de pooling en cada eje.

    Returns
    -------
    ndarray
        Array reducido de forma (Nr//pool_r, Nt//pool_t, Np//pool_p).
    """
    # Recortar para que sea divisible
    nr, nt, np_ = arr.shape
    nr_cut = (nr // pool_r) * pool_r
    nt_cut = (nt // pool_t) * pool_t
    np_cut = (np_ // pool_p) * pool_p
    arr = arr[:nr_cut, :nt_cut, :np_cut]

    # Reshape en bloques y promediar
    return arr.reshape(
        nr_cut // pool_r, pool_r,
        nt_cut // pool_t, pool_t,
        np_cut // pool_p, pool_p,
    ).mean(axis=(1, 3, 5))


def _sum_pool_3d(arr: np.ndarray, pool_r: int, pool_t: int, pool_p: int) -> np.ndarray:
    """
    Reduce una malla 3D por Sum Pooling (suma por bloques).

    Idéntico al Average Pooling pero sumando en vez de promediando,
    lo cual es necesario para conservar la carga total cuando se reduce dV.

    Parameters
    ----------
    arr : ndarray, shape (Nr, Nt, Np)
        Array 3D a reducir.
    pool_r, pool_t, pool_p : int
        Factores de pooling en cada eje.

    Returns
    -------
    ndarray
        Array reducido de forma (Nr//pool_r, Nt//pool_t, Np//pool_p).
    """
    nr, nt, np_ = arr.shape
    nr_cut = (nr // pool_r) * pool_r
    nt_cut = (nt // pool_t) * pool_t
    np_cut = (np_ // pool_p) * pool_p
    arr = arr[:nr_cut, :nt_cut, :np_cut]

    return arr.reshape(
        nr_cut // pool_r, pool_r,
        nt_cut // pool_t, pool_t,
        np_cut // pool_p, pool_p,
    ).sum(axis=(1, 3, 5))


def generar_distribucion_carga(
    mask: list = [0, 1, 1, 1],
    n_r: int = 40,
    n_theta: int = 60,
    n_phi: int = 60,
    max_r_factor: float = 40,
    pool_r: int = 2,
    pool_t: int = 2,
    pool_p: int = 3,
) -> tuple:
    """
    Genera la distribución de carga discreta 3D y devuelve coordenadas,
    densidades y diferenciales de volumen.

    Utiliza Average Pooling para reducir la resolución de la malla 3D
    en lugar de submuestreo crudo por slicing. Esto promedia la densidad
    de carga dentro de cada bloque y acumula (sum-pool) los diferenciales
    de volumen, lo cual CONSERVA ESTRICTAMENTE la carga total integrada.

    Parameters
    ----------
    mask : list of int
        Máscara de activación de momentos multipolares.
    n_r, n_theta, n_phi : int
        Resolución de la malla en coordenadas esféricas.
    max_r_factor : float
        Factor multiplicativo de A0 para el radio máximo.
    pool_r, pool_t, pool_p : int
        Factores de pooling (reducción) en cada eje.
        pool=1 no reduce, pool=2 reduce a la mitad, etc.

    Returns
    -------
    Xs, Ys, Zs : ndarray, shape (M,)
        Coordenadas cartesianas de los M vóxeles reducidos.
    Rs : ndarray, shape (M,)
        Densidad de carga promediada en cada vóxel.
    dVs : ndarray, shape (M,)
        Diferencial de volumen acumulado por bloque.
    """
    n = len(mask)

    # --- malla 3D esférica ---
    r = np.linspace(0, max_r_factor * A0, n_r)
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    # Diferenciales uniformes
    dr = r[1] - r[0] if len(r) > 1 else 0.0
    dt = theta[1] - theta[0] if len(theta) > 1 else 0.0
    dp = phi[1] - phi[0] if len(phi) > 1 else 0.0

    R, T, P = np.meshgrid(r, theta, phi, indexing="ij")

    # psi depende de r y theta; phi no cambia nada porque m=0
    psi_rt = psi_m0_superposition(n=n, r=R[:, :, 0], theta=T[:, :, 0], active_mask=mask)
    rho_rt = electron_density(psi_rt)

    # Expandir por simetría axial
    rho_3d = rho_rt[:, :, None] * np.ones_like(P)

    # Coordenadas cartesianas
    X = R * np.sin(T) * np.cos(P)
    Y = R * np.sin(T) * np.sin(P)
    Z = R * np.cos(T)

    # Diferencial de volumen en esféricas: dV = r² sin(θ) dr dθ dφ
    dV_3d = (R**2) * np.sin(T) * dr * dt * dp

    # Carga por vóxel: q_k = rho_k * dV_k
    q_3d = rho_3d * dV_3d

    # Carga total antes de pooling (para verificación de conservación)
    Q_pre_pool = np.sum(q_3d)

    # --- Sum Pooling de la carga por vóxel (conserva Q exactamente) ---
    q_pooled = _sum_pool_3d(q_3d, pool_r, pool_t, pool_p)

    # --- Sum Pooling de los diferenciales de volumen ---
    dV_pooled = _sum_pool_3d(dV_3d, pool_r, pool_t, pool_p)

    # Densidad efectiva: rho_eff = Q_bloque / dV_bloque
    # Evitar div/0 en vóxeles con dV=0 (r=0 o sin(theta)=0)
    dV_safe = np.where(np.abs(dV_pooled) > 1e-300, dV_pooled, 1.0)
    rho_pooled = q_pooled / dV_safe
    # Donde dV=0, la carga también es 0, así que rho puede quedar en 0
    rho_pooled = np.where(np.abs(dV_pooled) > 1e-300, rho_pooled, 0.0)

    # --- Average Pooling de las coordenadas (centro geométrico del bloque) ---
    X_pooled = _average_pool_3d(X, pool_r, pool_t, pool_p)
    Y_pooled = _average_pool_3d(Y, pool_r, pool_t, pool_p)
    Z_pooled = _average_pool_3d(Z, pool_r, pool_t, pool_p)

    # Verificación de conservación de carga (debe ser exacta a nivel de fp)
    Q_post_pool = np.sum(q_pooled)
    rel_error = abs(Q_post_pool - Q_pre_pool) / (abs(Q_pre_pool) + 1e-300)
    assert rel_error < 1e-12, (
        f"Error de conservación de carga tras pooling: "
        f"Q_pre={Q_pre_pool:.6e}, Q_post={Q_post_pool:.6e}, err_rel={rel_error:.2e}"
    )

    # Aplanar para salida
    Xs = X_pooled.ravel()
    Ys = Y_pooled.ravel()
    Zs = Z_pooled.ravel()
    Rs = rho_pooled.ravel()
    dVs = dV_pooled.ravel()

    return Xs, Ys, Zs, Rs, dVs

if __name__ == "__main__":
    Xs, Ys, Zs, Rs, dVs = generar_distribucion_carga()

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