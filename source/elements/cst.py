import numpy

from source.elements.element2d import Element2D


class CST(Element2D):
    """Triangulo de deformacion constante (3 nodos, interpolacion lineal).

    Numeracion de nodos (antihoraria):

        3
        | \\
        1--2

    Coordenadas naturales: nodo1 (0,0), nodo2 (1,0), nodo3 (0,1).
    """

    number_of_nodes = 3

    # Coordenadas naturales (xi, eta) de cada nodo
    natural_coordinates = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]

    # B es constante: un solo punto de Gauss integra exactamente.
    # El peso 1/2 es el area del triangulo de referencia.
    gauss_points = [(1.0 / 3.0, 1.0 / 3.0, 0.5)]

    def shape_functions(self, xi, eta):
        # Funciones de forma N_i
        N = numpy.array([
            1.0 - xi - eta,
            xi,
            eta,
        ])

        # Derivadas: fila 0 = dN/dxi, fila 1 = dN/deta
        dN = numpy.array([
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ])

        return N, dN
