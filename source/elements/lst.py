import numpy

from source.elements.element2d import Element2D


class LST(Element2D):
    """Triangulo de deformacion lineal (6 nodos, interpolacion cuadratica).

    Numeracion de nodos (antihoraria): vertices 1-3, luego puntos medios.

        3
        | \\
        6   5
        |     \\
        1--4---2

    Nodo 4: medio de 1-2, nodo 5: medio de 2-3, nodo 6: medio de 3-1.
    Coordenadas naturales de los vertices: (0,0), (1,0), (0,1).
    """

    number_of_nodes = 6

    natural_coordinates = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
    ]

    # Regla de 3 puntos (exacta hasta grado 2) sobre el triangulo de referencia
    gauss_points = [
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    ]

    def shape_functions(self, xi, eta):
        # Coordenadas de area L1, L2, L3 y sus derivadas respecto a (xi, eta)
        L = numpy.array([1.0 - xi - eta, xi, eta])
        dL = numpy.array([
            [-1.0, 1.0, 0.0],   # dL/dxi
            [-1.0, 0.0, 1.0],   # dL/deta
        ])

        # Nodos de vertice: N_i = L_i (2 L_i - 1)
        # Nodos de lado:    N_4 = 4 L1 L2, N_5 = 4 L2 L3, N_6 = 4 L3 L1
        N = numpy.zeros(6)
        dN = numpy.zeros((2, 6))

        for i in range(3):
            N[i] = L[i] * (2.0 * L[i] - 1.0)
            dN[:, i] = (4.0 * L[i] - 1.0) * dL[:, i]

        sides = [(0, 1), (1, 2), (2, 0)]

        for k in range(3):
            a, b = sides[k]
            N[3 + k] = 4.0 * L[a] * L[b]
            dN[:, 3 + k] = 4.0 * (L[a] * dL[:, b] + L[b] * dL[:, a])

        return N, dN
