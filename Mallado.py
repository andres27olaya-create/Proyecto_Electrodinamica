# La idea por ahora es generar una tesela puede ser de cubos o tetraetros con los cuales telesar el espacio, creo que la opcion mas facil sera con cubos
# dado que tiene mas similitud  a las coordenadas cartesianas, y se puede generar un cubo a partir de un punto y una longitud de lado, ademas de que es mas facil de visualizar y manipular.
#comienzo con definir un cubo, que es un objeto que tiene 8 vertices y 6 caras, cada cara es un cuadrado, y cada vertice es un punto en el espacio tridimensional.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

class Cubo:
    def __init__(self, punto, longitud_lado):
        self.punto = punto
        self.longitud_lado = longitud_lado
        self.vertices = self.generar_vertices()
        self.caras = self.generar_caras()

    def generar_vertices(self):
        x, y, z = self.punto
        l = self.longitud_lado
        return [
            (x, y, z),
            (x + l, y, z),
            (x + l, y + l, z),
            (x, y + l, z),
            (x, y, z + l),
            (x + l, y, z + l),
            (x + l, y + l, z + l),
            (x, y + l, z + l)
        ]

    def generar_caras(self):
        return [
            [self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]],  # Cara inferior
            [self.vertices[4], self.vertices[5], self.vertices[6], self.vertices[7]],  # Cara superior
            [self.vertices[0], self.vertices[1], self.vertices[5], self.vertices[4]],  # Cara frontal
            [self.vertices[2], self.vertices[3], self.vertices[7], self.vertices[6]],  # Cara trasera
            [self.vertices[1], self.vertices[2], self.vertices[6], self.vertices[5]],  # Cara derecha
            [self.vertices[0], self.vertices[3], self.vertices[7], self.vertices[4]]   # Cara izquierda
        ]
#listo ya esta creado el cubo, ahora puedo generar un cubo a partir de un punto y una longitud de lado, y puedo obtener sus vertices y caras, ahora puedo generar un cubo y graficarlo.
#ahora necesito una funcion o una clase para hacer figuras geometricas, pero por ahora planas, como un triangulo, un cuadrado, hexagono etc, con n lados posibles
#iniciamente graficare una figura de lados n


#se grafica
# plot = plt.figure().add_subplot()
# x = [vertice[0] for vertice in vertices]
# y = [vertice[1] for vertice in vertices]
# x.append(vertices[0][0])  # Cerrar la figura
# y.append(vertices[0][1])
# plot.plot(x, y, color='b')
# plt.axis('equal')
# plt.show()

#ahora necesito teselar cualquier figura con cuadrados hasta un cierto nivel de profundidad, para eso necesito una funcion que reciba una figura y un nivel de profundidad, y que genere los cuadrados necesarios para teselar la figura, y que devuelva una lista de cuadrados.
#hmm como lo hago?
#¿que indica esto? la profuncidad minima, osea que los cuadrados solo se generaran dentro de la figura intentanto ocuparla lo mas posible, y que no se generaran cuadrados fuera de la figura, y que si un cuadrado no puede ser generado dentro de la figura, entonces se generara un cuadrado mas pequeño dentro de la figura, hasta llegar a la profundidad minima.
#entonces mi idea es que del centro vayan unos rayos a 45° si uno de los rayos toca un vertice o lado de la figura, entonces se genera un cuadrado en ese vertice, y si no toca ningun vertice, entonces se genera un cuadrado en el centro de la figura, y luego se repite el proceso con los cuadrados generados, hasta llegar a la profundidad minima.

import numpy as np
import matplotlib.pyplot as plt

# =========================
# GEOMETRÍA BÁSICA
# =========================

def generar_vertices_poligono_regular(n, r=1.0):
    if n < 3:
        raise ValueError("El número de lados debe ser al menos 3.")
    angulo = 2 * np.pi / n
    vertices = [(r * np.cos(i * angulo), r * np.sin(i * angulo)) for i in range(n)]
    return np.array(vertices, dtype=float)

def cuadrado_desde_centro(cx, cy, lado):
    h = lado / 2.0
    return np.array([
        [cx - h, cy - h],
        [cx + h, cy - h],
        [cx + h, cy + h],
        [cx - h, cy + h],
    ], dtype=float)

def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def punto_en_segmento(p, a, b, eps=1e-12):
    if abs(cross2(p - a, b - a)) > eps:
        return False
    return np.dot(p - a, p - b) <= eps

def segmentos_intersectan(a, b, c, d, eps=1e-12):
    r = b - a
    s = d - c
    den = cross2(r, s)

    if abs(den) < eps:
        if abs(cross2(c - a, r)) > eps:
            return False
        return (
            punto_en_segmento(a, c, d, eps) or
            punto_en_segmento(b, c, d, eps) or
            punto_en_segmento(c, a, b, eps) or
            punto_en_segmento(d, a, b, eps)
        )

    t = cross2(c - a, s) / den
    u = cross2(c - a, r) / den
    return (-eps <= t <= 1 + eps) and (-eps <= u <= 1 + eps)

def punto_en_poligono(p, poly):
    # incluye borde
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        if punto_en_segmento(p, a, b):
            return True

    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            if x < xinters:
                inside = not inside
    return inside

def cuadrado_dentro_poligono(corners, poly):
    return all(punto_en_poligono(c, poly) for c in corners)

def cuadrado_intersecta_poligono(corners, poly):
    # alguna esquina del cuadrado dentro
    if any(punto_en_poligono(c, poly) for c in corners):
        return True

    # algún vértice del polígono dentro del cuadrado
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0)
    for v in poly:
        if xmin <= v[0] <= xmax and ymin <= v[1] <= ymax:
            return True

    # intersección entre lados
    sq_edges = list(zip(corners, np.roll(corners, -1, axis=0)))
    poly_edges = list(zip(poly, np.roll(poly, -1, axis=0)))

    for a, b in sq_edges:
        for c, d in poly_edges:
            if segmentos_intersectan(a, b, c, d):
                return True

    return False

# =========================
# TESSELADO SIMPLE
# =========================

def profundidad_desde_tamano(lado_inicial, lado_objetivo, profundidad_max):
    if lado_objetivo <= 0:
        return profundidad_max
    d = int(np.ceil(np.log2(lado_inicial / lado_objetivo)))
    return max(0, min(profundidad_max, d))


def crear_tamano_objetivo(centro_ref, radio_ref, h_min=0.04, h_max=0.25, alpha=2.0):
    def tamano_objetivo(cx, cy):
        p = np.array([cx, cy], dtype=float)
        r = np.linalg.norm(p - centro_ref)
        t = 0.0 if radio_ref == 0 else np.clip(r / radio_ref, 0.0, 1.0)
        return h_min + (h_max - h_min) * (t ** alpha)
    return tamano_objetivo


def teselar_quadtree(vertices, profundidad, min_lado=1e-5, h_min=0.04, h_max=0.25, alpha=2.0):
    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)

    lado_inicial = max(xmax - xmin, ymax - ymin)
    cx0 = (xmin + xmax) / 2.0
    cy0 = (ymin + ymax) / 2.0

    centro_ref = np.array([cx0, cy0], dtype=float)
    radio_ref = np.max(np.linalg.norm(vertices - centro_ref, axis=1))
    tamano_objetivo = crear_tamano_objetivo(
        centro_ref=centro_ref,
        radio_ref=radio_ref,
        h_min=h_min,
        h_max=h_max,
        alpha=alpha
    )

    cuadrados = []

    def rec(cx, cy, lado, nivel_actual):
        corners = cuadrado_desde_centro(cx, cy, lado)

        if not cuadrado_intersecta_poligono(corners, vertices):
            return

        lado_objetivo = tamano_objetivo(cx, cy)
        nivel_objetivo = profundidad_desde_tamano(lado_inicial, lado_objetivo, profundidad)

        if cuadrado_dentro_poligono(corners, vertices) and nivel_actual >= nivel_objetivo:
            cuadrados.append((cx, cy, lado))
            return

        if nivel_actual >= profundidad or lado <= min_lado:
            if cuadrado_dentro_poligono(corners, vertices):
                cuadrados.append((cx, cy, lado))
            return

        h = lado / 4.0
        lado_hijo = lado / 2.0

        rec(cx - h, cy - h, lado_hijo, nivel_actual + 1)
        rec(cx + h, cy - h, lado_hijo, nivel_actual + 1)
        rec(cx - h, cy + h, lado_hijo, nivel_actual + 1)
        rec(cx + h, cy + h, lado_hijo, nivel_actual + 1)

    rec(cx0, cy0, lado_inicial, 0)
    return cuadrados

# =========================
# DIBUJO
# =========================

def dibujar_resultado(vertices, cuadrados):
    fig, ax = plt.subplots(figsize=(8, 8))

    vx = np.append(vertices[:, 0], vertices[0, 0])
    vy = np.append(vertices[:, 1], vertices[0, 1])
    ax.plot(vx, vy, linewidth=2)

    for cx, cy, lado in cuadrados:
        pts = cuadrado_desde_centro(cx, cy, lado)
        px = np.append(pts[:, 0], pts[0, 0])
        py = np.append(pts[:, 1], pts[0, 1])
        ax.plot(px, py, linewidth=1)

    ax.set_aspect('equal', adjustable='box')
    plt.show()

# =========================
# EJEMPLO
# =========================

# ---- PARÁMETROS DE LA FIGURA ----
n_lados = 5                 # Número de lados del polígono regular
radio_figura = 1.0          # Radio del polígono regular

# ---- PARÁMETROS DEL MALLADO ----
profundidad_maxima = 8      # Máxima profundidad de subdivisión del quadtree
                            # (mayor = mallado más fino, pero más lento)

# Tamaño mínimo permitido para los cuadrados (para evitar cálculos infinitos)
lado_minimo = 1e-5

# ---- PARÁMETROS DE REFINAMIENTO ADAPTATIVO ----
h_min = 0.04                # Tamaño objetivo del cuadrado en el CENTRO de la figura
                            # (menor = mallado más fino en el centro)

h_max = 0.10                # Tamaño objetivo del cuadrado en el BORDE de la figura
                            # (mayor = cuadrados más grandes en el borde)

alpha = 1.0                 # Factor de transición entre el centro y el borde
                            # alpha = 1.0 → transición lineal (uniforme)
                            # alpha > 1.0 → transición suave (más refinamiento en el borde)
                            # alpha < 1.0 → transición agresiva (más refinamiento en el centro)

# ---- GENERAR Y VISUALIZAR ----
vertices = generar_vertices_poligono_regular(n_lados, r=radio_figura)

cuadrados = teselar_quadtree(
    vertices=vertices,
    profundidad=profundidad_maxima,
    min_lado=lado_minimo,
    h_min=h_min,
    h_max=h_max,
    alpha=alpha
)

# Modificar también la función de tamaño objetivo dentro de teselar_quadtree
# si necesitas ajustes más avanzados en los parámetros h_min, h_max, alpha

print("Cantidad de cuadrados generados:", len(cuadrados))
dibujar_resultado(vertices, cuadrados)


# Cubo1 = Cubo((0, 0, 0), 1)

# plot = plt.figure().add_subplot(projection='3d')
# for cara in Cubo1.caras:
#     x = [vertice[0] for vertice in cara]
#     y = [vertice[1] for vertice in cara]
#     z = [vertice[2] for vertice in cara]
#     x.append(cara[0][0])  # Cerrar la cara
#     y.append(cara[0][1])
#     z.append(cara[0][2])
#     plot.plot(x, y, z, color='b')
# plt.show()
