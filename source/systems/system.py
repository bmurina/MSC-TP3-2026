import math

import numpy

from source.systems.matrix import DenseMatrix
from source.systems.boundary_conditions import BoundaryConditions
from source.systems.loads import compute_hydrostatic_forces, compute_body_forces


class System:
    """Sistema global de ecuaciones por Elementos Finitos (K u = F + R).

    Administra el ensamblaje de la matriz de rigidez global, la aplicacion
    de cargas (hidroestatica, peso propio, nodales), la imposicion de
    condiciones de borde esenciales (Dirichlet) o de Robin (resortes elasticos)
    y el calculo de desplazamientos y reacciones.
    """

    def __init__(self, mesh, elements):
        self.mesh = mesh
        self.elements = elements
        self.number_of_dofs = mesh.number_of_dofs()

        self.K = DenseMatrix(self.number_of_dofs)
        self.Ks = DenseMatrix(self.number_of_dofs)   # rigidez de resortes (Robin)
        self.has_springs = False
        self.released_nodes = set()   # nodos de la base liberados por traccion
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

    def apply_robin_base(self, kn, kt, y_val=0.0, tol=1e-5):
        """Condicion de Robin en la base y = y_val: fundacion elastica de Winkler.

        La fundacion ejerce una traccion t = -(kt u_t t_dir + kn u_n n_dir)
        [N/m2], proporcional al desplazamiento tangencial u_t y normal u_n.
        Su contribucion a la rigidez es

            Ks = integral_Gb N^T S N t ds,   S = kt t_dir t_dir^T + kn n_dir n_dir^T

        integrada con Gauss de 3 puntos sobre cada arista de la base
        (lineal de 2 nodos o cuadratica de 3 nodos).

        Parametros
        ----------
        kn : float, modulo de balasto normal [N/m3]
        kt : float, modulo de balasto tangencial [N/m3]

        Devuelve
        --------
        dict con 'nodes' (ids de nodos de la base) y 'length' (longitud
        total integrada, debe coincidir con el ancho de la base).
        """
        gp = [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]
        gw = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

        base_nodes = set()
        total_length = 0.0

        for element in self.elements:
            for local_edge in element.edges:
                nodes = [element.node_global_ids[i] for i in local_edge]

                if any(abs(self.mesh.node(n).y - y_val) > tol for n in nodes):
                    continue

                coords = numpy.array([[self.mesh.node(n).x, self.mesh.node(n).y] for n in nodes])
                m = len(nodes)
                ks_e = numpy.zeros((2 * m, 2 * m))

                for s, w in zip(gp, gw):
                    if m == 2:
                        N = numpy.array([0.5 * (1.0 - s), 0.5 * (1.0 + s)])
                        dN = numpy.array([-0.5, 0.5])
                    else:
                        # Orden (vertice, medio, vertice)
                        N = numpy.array([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)])
                        dN = numpy.array([s - 0.5, -2.0 * s, s + 0.5])

                    dxds = dN @ coords
                    jac = numpy.linalg.norm(dxds)
                    t_dir = dxds / jac
                    n_dir = numpy.array([-t_dir[1], t_dir[0]])
                    S = kt * numpy.outer(t_dir, t_dir) + kn * numpy.outer(n_dir, n_dir)

                    Nmat = numpy.zeros((2, 2 * m))
                    Nmat[0, 0::2] = N
                    Nmat[1, 1::2] = N

                    ks_e += Nmat.T @ S @ Nmat * element.thickness * jac * w
                    total_length += jac * w

                dofs = []
                for n in nodes:
                    dofs.extend(self.mesh.node_global_dofs(n))
                self.Ks.scatter_local_to_global(ks_e, dofs)

                base_nodes.update(nodes)

        if not base_nodes:
            raise ValueError("No se encontraron aristas en la base (y=%f)" % y_val)

        self.has_springs = True
        return {"nodes": sorted(base_nodes), "length": total_length}

    def release_base_node(self, node_id):
        """Libera un nodo de la base: quita sus DOFs prescriptos (empotramiento)
        y sus resortes de Robin (filas y columnas de Ks), de modo que queda
        libre de desplazarse."""
        self.bc.release_node(self.mesh, node_id)
        dofs = self.mesh.node_global_dofs(node_id)
        self.Ks[dofs, :] = 0.0
        self.Ks[:, dofs] = 0.0
        self.released_nodes.add(node_id)

    def solve_without_base_tension(self, solver_func=None, y_val=0.0, tol=1e-5, max_iter=50):
        """Resuelve el sistema sin admitir traccion en la base.

        La presa se apoya por peso propio: el suelo solo puede comprimirla.
        Tras cada resolucion se liberan los nodos de la base cuya reaccion
        vertical es de traccion (Ry < 0, el suelo tira de la presa hacia
        abajo) y se vuelve a resolver, hasta que ningun nodo vinculado quede
        traccionado (al liberar nodos la traccion puede migrar a vecinos).

        Devuelve
        --------
        u, reactions : como solve()
        info : dict con 'released' (ids de nodos liberados, en orden),
               'iterations' y 'penetrating' (nodos liberados con uy < 0).
        """
        released = []
        for iteration in range(1, max_iter + 1):
            u, reactions = self.solve(solver_func=solver_func)
            base = self.get_base_reactions(y_val=y_val, tol=tol)
            force_tol = 1e-9 * max(abs(base["total_Ry"]), 1.0)

            tensile = [n["node_id"] for n in base["nodes"]
                       if n["node_id"] not in self.released_nodes and n["Ry"] < -force_tol]
            if not tensile:
                break
            for node_id in tensile:
                self.release_base_node(node_id)
            released.extend(tensile)
        else:
            raise RuntimeError("No se elimino la traccion en la base en %d iteraciones" % max_iter)

        if not self.bc.get_fixed_dofs() and not numpy.any(self.Ks.to_numpy()):
            raise RuntimeError("Se liberaron todos los vinculos de la base (presa sin apoyo).")

        # Un nodo liberado con uy < 0 penetraria el suelo (deberia volver a apoyarse)
        penetrating = [n for n in released if u[self.mesh.node_global_dofs(n)[1]] < 0.0]

        return u, reactions, {"released": released, "iterations": iteration, "penetrating": penetrating}

    def solve(self, solver_func=None):
        """Resuelve el sistema estatico lineal (K + Ks) u = F con condiciones de
        Dirichlet y/o resortes de Robin.

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

        if not fixed_dofs and not self.has_springs:
            raise RuntimeError("El sistema no tiene condiciones de borde prescriptas (matriz singular).")

        u = numpy.zeros(self.number_of_dofs, dtype=float)
        u_p = self.bc.get_prescribed_values()
        u[fixed_dofs] = u_p

        K_mat = self.K.to_numpy()
        K_tot = K_mat + self.Ks.to_numpy()

        # Particion: K_ff * u_f = F_f - K_fp * u_p
        K_ff = K_tot[numpy.ix_(free_dofs, free_dofs)]
        F_f = self.F[free_dofs]

        if len(fixed_dofs) > 0 and numpy.any(u_p != 0.0):
            K_fp = K_tot[numpy.ix_(free_dofs, fixed_dofs)]
            F_eff = F_f - K_fp @ u_p
        else:
            F_eff = F_f

        u_f = solver_func(K_ff, F_eff)
        u[free_dofs] = u_f

        # Calculo exacto de reacciones nodales: R = K @ u - F (K sin resortes)
        # En los DOFs libres sin resortes R vale 0 dentro del error numerico;
        # en los nodos con resortes R = -Ks @ u es la fuerza de la fundacion.
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
