import math

import numpy

from source.elements.element2d import Element2D

_G = math.sqrt(3.0 / 5.0)
_POINTS = [-_G, 0.0, _G]
_WEIGHTS = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
_GAUSS_3X3 = [
    (_POINTS[i], _POINTS[j], _WEIGHTS[i] * _WEIGHTS[j])
    for j in range(3) for i in range(3)
]


class Q8(Element2D):
    """Cuadrilatero cuadratico serendipity (8 nodos).

    Numeracion de nodos (antihoraria): vertices 1-4, luego puntos medios.

        4----7----3
        |         |
        8         6
        |         |
        1----5----2

    Nodo 5: medio de 1-2, nodo 6: medio de 2-3,
    nodo 7: medio de 3-4, nodo 8: medio de 4-1.
    """

    number_of_nodes = 8

    natural_coordinates = [
        (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
        (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
    ]

    # Cuadratura de Gauss 3x3 (puntos en 0 y +-sqrt(3/5))
    gauss_points = _GAUSS_3X3

    def shape_functions(self, xi, eta):
        N = numpy.zeros(8)
        dN = numpy.zeros((2, 8))

        for i in range(8):
            xi_i, eta_i = self.natural_coordinates[i]

            if i < 4:
                # Nodos de vertice:
                # N_i = 1/4 (1 + xi_i xi)(1 + eta_i eta)(xi_i xi + eta_i eta - 1)
                a = 1.0 + xi_i * xi
                b = 1.0 + eta_i * eta
                c = xi_i * xi + eta_i * eta - 1.0

                N[i] = 0.25 * a * b * c
                dN[0, i] = 0.25 * xi_i * b * (c + a)
                dN[1, i] = 0.25 * eta_i * a * (c + b)

            elif xi_i == 0.0:
                # Nodos de lado en eta = +-1: N_i = 1/2 (1 - xi^2)(1 + eta_i eta)
                N[i] = 0.5 * (1.0 - xi * xi) * (1.0 + eta_i * eta)
                dN[0, i] = -xi * (1.0 + eta_i * eta)
                dN[1, i] = 0.5 * (1.0 - xi * xi) * eta_i

            else:
                # Nodos de lado en xi = +-1: N_i = 1/2 (1 + xi_i xi)(1 - eta^2)
                N[i] = 0.5 * (1.0 + xi_i * xi) * (1.0 - eta * eta)
                dN[0, i] = 0.5 * xi_i * (1.0 - eta * eta)
                dN[1, i] = -eta * (1.0 + xi_i * xi)

        return N, dN
