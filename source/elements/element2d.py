import numpy


class Element2D:
    """Clase base para elementos planos (estado plano de tension).

    Mantiene la misma interfaz que Bar2D (global_dofs, stiffness_matrix, ...)
    para poder usar DenseMatrix.scatter_local_to_global sin cambios.

    Cada elemento concreto (CST, LST, Q4, Q8) solo define:
      - number_of_nodes
      - natural_coordinates: coordenadas (xi, eta) de sus nodos
      - gauss_points: lista de (xi, eta, peso) para integrar
      - shape_functions(xi, eta): devuelve N (n,) y dN (2, n) respecto a (xi, eta)
    """

    dofs_per_node = 2
    number_of_nodes = 0
    natural_coordinates = []
    gauss_points = []

    def __init__(self, global_id, node_global_ids, mesh, material, thickness=1.0):
        if len(node_global_ids) != self.number_of_nodes:
            raise ValueError(
                "Expected %d nodes, got %d"
                % (self.number_of_nodes, len(node_global_ids))
            )

        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        # Coordenadas nodales del elemento: matriz (n, 2)
        self.coords = numpy.zeros((self.number_of_nodes, 2))

        for i in range(self.number_of_nodes):
            node = mesh.node(node_global_ids[i])
            self.coords[i, 0] = node.x
            self.coords[i, 1] = node.y

        # Matriz constitutiva de tension plana
        self.D = material.plane_stress_matrix()

        # Matriz de rigidez y area del elemento (integracion de Gauss)
        self.area = 0.0
        self.Ke = self._compute_stiffness()

    # ------------------------------------------------------------------
    # Cinematica
    # ------------------------------------------------------------------
    def shape_functions(self, xi, eta):
        raise NotImplementedError

    def jacobian(self, dN):
        """Jacobiano J = dN @ coords (2x2) y su determinante."""
        J = dN @ self.coords
        detJ = numpy.linalg.det(J)

        if detJ <= 0.0:
            raise ValueError(
                "Non positive jacobian in element %d (check node ordering)"
                % self.global_id
            )

        return J, detJ

    def B_matrix(self, xi, eta):
        """Matriz deformacion-desplazamiento B (3 x 2n) en el punto (xi, eta).

        Devuelve tambien N y detJ para reutilizarlos en la integracion.
        """
        N, dN = self.shape_functions(xi, eta)
        J, detJ = self.jacobian(dN)

        # Derivadas de las funciones de forma respecto a (x, y)
        dNdx = numpy.linalg.solve(J, dN)

        B = numpy.zeros((3, 2 * self.number_of_nodes))

        for i in range(self.number_of_nodes):
            B[0, 2 * i] = dNdx[0, i]        # eps_xx = du/dx
            B[1, 2 * i + 1] = dNdx[1, i]    # eps_yy = dv/dy
            B[2, 2 * i] = dNdx[1, i]        # gamma_xy = du/dy + dv/dx
            B[2, 2 * i + 1] = dNdx[0, i]

        return B, N, detJ

    # ------------------------------------------------------------------
    # Rigidez y cargas
    # ------------------------------------------------------------------
    def _compute_stiffness(self):
        """Ke = integral(B^T D B t dA) por cuadratura de Gauss."""
        n_dofs = self.dofs_per_node * self.number_of_nodes
        Ke = numpy.zeros((n_dofs, n_dofs))

        for xi, eta, weight in self.gauss_points:
            B, N, detJ = self.B_matrix(xi, eta)
            Ke += B.T @ self.D @ B * self.thickness * detJ * weight
            self.area += detJ * weight

        return Ke

    def stiffness_matrix(self):
        return self.Ke

    def global_dofs(self):
        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    def body_force_vector(self, bx, by):
        """Fuerzas nodales equivalentes de una fuerza de volumen (bx, by)
        [N/m3] constante: fe = integral(N^T b t dA).

        Para peso propio usar bx = 0, by = -rho * g.
        """
        fe = numpy.zeros(self.dofs_per_node * self.number_of_nodes)

        for xi, eta, weight in self.gauss_points:
            N, dN = self.shape_functions(xi, eta)
            J, detJ = self.jacobian(dN)

            for i in range(self.number_of_nodes):
                fe[2 * i] += N[i] * bx * self.thickness * detJ * weight
                fe[2 * i + 1] += N[i] * by * self.thickness * detJ * weight

        return fe

    # ------------------------------------------------------------------
    # Post-proceso
    # ------------------------------------------------------------------
    def element_displacements(self, u):
        """Extrae del vector global u los desplazamientos del elemento."""
        dofs = self.global_dofs()
        ue = numpy.zeros(len(dofs))

        for i in range(len(dofs)):
            ue[i] = u[dofs[i]]

        return ue

    def strain(self, u, xi, eta):
        """Deformacion [eps_xx, eps_yy, gamma_xy] en el punto (xi, eta)."""
        B, N, detJ = self.B_matrix(xi, eta)
        return B @ self.element_displacements(u)

    def stress(self, u, xi, eta):
        """Tension [sigma_xx, sigma_yy, tau_xy] en el punto (xi, eta)."""
        return self.D @ self.strain(u, xi, eta)

    def stress_at_nodes(self, u):
        """Tensiones evaluadas en los nodos del elemento: matriz (n, 3)."""
        sigma = numpy.zeros((self.number_of_nodes, 3))

        for i in range(self.number_of_nodes):
            xi, eta = self.natural_coordinates[i]
            sigma[i] = self.stress(u, xi, eta)

        return sigma

    def stress_at_centroid(self, u):
        """Tension promedio del elemento (promedio ponderado en puntos de Gauss)."""
        sigma = numpy.zeros(3)
        total = 0.0

        for xi, eta, weight in self.gauss_points:
            B, N, detJ = self.B_matrix(xi, eta)
            sigma += self.stress(u, xi, eta) * detJ * weight
            total += detJ * weight

        return sigma / total

    @staticmethod
    def principal_stresses(sigma):
        """Tensiones principales (sigma_1 >= sigma_2) de [sxx, syy, txy]."""
        mean = 0.5 * (sigma[0] + sigma[1])
        radius = numpy.sqrt((0.5 * (sigma[0] - sigma[1])) ** 2 + sigma[2] ** 2)
        return mean + radius, mean - radius
