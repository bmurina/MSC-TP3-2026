import math

import numpy

from source.elements.element2d import Element2D


class Q4(Element2D):
    """Cuadrilatero bilineal isoparametrico (4 nodos).

    Numeracion de nodos (antihoraria):

        4--------3
        |        |
        1--------2

    Coordenadas naturales (xi, eta) en [-1, 1].
    """

    number_of_nodes = 4

    # Coordenadas naturales de los nodos
    natural_coordinates = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    # Cuadratura de Gauss 2x2 (puntos en +-1/sqrt(3), pesos 1)
    _g = 1.0 / math.sqrt(3.0)
    gauss_points = [
        (-_g, -_g, 1.0), (_g, -_g, 1.0),
        (_g, _g, 1.0), (-_g, _g, 1.0),
    ]

    def shape_functions(self, xi, eta):
        N = numpy.zeros(4)
        dN = numpy.zeros((2, 4))

        for i in range(4):
            xi_i, eta_i = self.natural_coordinates[i]

            # N_i = 1/4 (1 + xi_i xi)(1 + eta_i eta)
            N[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta)
            dN[0, i] = 0.25 * xi_i * (1.0 + eta_i * eta)
            dN[1, i] = 0.25 * eta_i * (1.0 + xi_i * xi)

        return N, dN
