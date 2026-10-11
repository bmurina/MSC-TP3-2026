import numpy

from source.systems.matrix import DenseMatrix
from source.systems.boundary_conditions import BoundaryConditions
from source.systems.loads import compute_hydrostatic_forces, compute_body_forces


class System:
    """Sistema global de ecuaciones por Elementos Finitos (K u = F + R).

    Administra el ensamblaje de la matriz de rigidez global, la aplicacion
    de cargas (hidroestatica, peso propio, nodales), la imposicion de
    condiciones de borde esenciales y el calculo de desplazamientos y reacciones.
    """

    def __init__(self, mesh, elements):
        self.mesh = mesh
        self.elements = elements
        self.number_of_dofs = mesh.number_of_dofs()

        self.K = DenseMatrix(self.number_of_dofs)
        self.F = numpy.zeros(self.number_of_dofs, dtype=float)
        self.bc = BoundaryConditions(self.number_of_dofs)

        self.u = None
        self.reactions = None
        self.hydrostatic_info = None

    def assemble_stiffness(self):
        """Ensambla la matriz de rigidez global K a partir de los elementos."""
        for element in self.elements:
            ke = element.stiffness_matrix()
            dofs = element.global_dofs()
            self.K.scatter_local_to_global(ke, dofs)
        return self.K

    def apply_hydrostatic_pressure(self, water_height=28.0, rho_water=1000.0, gravity=9.81, tol=1e-5):
        """Aplica la distribucion de presion hidroestatica sobre los nodos en x=0, 0 <= y <= water_height."""
        f_hydro, info = compute_hydrostatic_forces(
            self.mesh,
            water_height=water_height,
            rho_water=rho_water,
            gravity=gravity,
            tol=tol,
        )
        self.F += f_hydro
        self.hydrostatic_info = info
        return info

    def apply_gravity(self, gravity=9.81):
        """Aplica la fuerza volumetrica de peso propio a todos los elementos."""
        f_body = compute_body_forces(self.mesh, self.elements, gravity=gravity)
        self.F += f_body
        return f_body

    def apply_nodal_force(self, node_id, fx=0.0, fy=0.0):
        """Aplica una carga puntual externa a un nodo especifico."""
        dofs = self.mesh.node_global_dofs(node_id)
        self.F[dofs[0]] += fx
        self.F[dofs[1]] += fy

    def fix_base(self, y_val=0.0, tol=1e-5):
        """Impone condicion de empotramiento / desplazamiento nulo (ux = 0, uy = 0) en la base y=y_val."""
        return self.bc.fix_base(self.mesh, y_val=y_val, tol=tol)

    def solve(self, solver_func=None):
        """Resuelve el sistema estatico lineal K u = F con condiciones de Dirichlet.

        Calcula el vector global de desplazamientos u y el vector de reacciones R.

        Parametros
        ----------
        solver_func : funcion opcional (K_ff, F_eff) -> u_f. Si es None usa numpy.linalg.solve.

        Devuelve
        --------
        u         : ndarray (number_of_dofs,) con los desplazamientos nodales
        reactions : ndarray (number_of_dofs,) con las fuerzas de reaccion nodales
        """
        if solver_func is None:
            solver_func = numpy.linalg.solve

        free_dofs = self.bc.get_free_dofs()
        fixed_dofs = self.bc.get_fixed_dofs()

        if not fixed_dofs:
            raise RuntimeError("El sistema no tiene condiciones de borde prescriptas (matriz singular).")

        u = numpy.zeros(self.number_of_dofs, dtype=float)
        u_p = self.bc.get_prescribed_values()
        u[fixed_dofs] = u_p

        K_mat = self.K.to_numpy()

        # Particion: K_ff * u_f = F_f - K_fp * u_p
        K_ff = K_mat[numpy.ix_(free_dofs, free_dofs)]
        F_f = self.F[free_dofs]

        if len(fixed_dofs) > 0 and numpy.any(u_p != 0.0):
            K_fp = K_mat[numpy.ix_(free_dofs, fixed_dofs)]
            F_eff = F_f - K_fp @ u_p
        else:
            F_eff = F_f

        u_f = solver_func(K_ff, F_eff)
        u[free_dofs] = u_f

        # Calculo exacto de reacciones nodales: R = K @ u - F
        # (En los DOFs libres, R vale exactamente 0 dentro del error numerico)
        reactions = K_mat @ u - self.F

        self.u = u
        self.reactions = reactions
        return u, reactions

    def get_base_reactions(self, y_val=0.0, tol=1e-5):
        """Devuelve las reacciones de vinculo en los nodos de la base (y = y_val).

        Retorna un diccionario con:
          - 'nodes': lista de dicts [{'node_id', 'x', 'Rx', 'Ry'}]
          - 'total_Rx': resultante horizontal total en la base [N]
          - 'total_Ry': resultante vertical total en la base [N]
          - 'moment_origin': momento resultante respecto a (0, 0)
        """
        if self.reactions is None:
            raise RuntimeError("Debe ejecutarse solve() antes de consultar las reacciones.")

        base_nodes_data = []
        sum_rx = 0.0
        sum_ry = 0.0
        moment_origin = 0.0

        for node in self.mesh.nodes:
            if abs(node.y - y_val) <= tol:
                dofs = self.mesh.node_global_dofs(node.global_id)
                rx = self.reactions[dofs[0]]
                ry = self.reactions[dofs[1]]
                base_nodes_data.append({
                    "node_id": node.global_id,
                    "x": node.x,
                    "y": node.y,
                    "Rx": rx,
                    "Ry": ry,
                })
                sum_rx += rx
                sum_ry += ry
                # Momento respecto al origen: Rx * y - Ry * x (en y=0, solo -Ry * x)
                moment_origin += rx * node.y - ry * node.x

        return {
            "nodes": base_nodes_data,
            "total_Rx": sum_rx,
            "total_Ry": sum_ry,
            "moment_origin": moment_origin,
        }
