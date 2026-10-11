import math
import numpy


def compute_hydrostatic_forces(mesh, water_height=28.0, rho_water=1000.0, gravity=9.81, x_upstream=0.0, tol=1e-5):
    """Calcula las fuerzas nodales equivalentes debidas a la presion hidroestatica del agua.

    El agua se ubica en la cara aguas arriba (x = x_upstream), entre y = 0 e y = water_height.
    La presion varia linealmente: p(y) = rho_water * gravity * (water_height - y) para 0 <= y <= water_height.
    La direccion de la carga es +x (empuje normal hacia la presa).

    Parametros
    ----------
    mesh         : Mesh
    water_height : float, altura de la superficie libre de agua [m] (por defecto 28 m)
    rho_water    : float, densidad del agua [kg/m3] (1000 kg/m3)
    gravity      : float, aceleracion de la gravedad [m/s2] (9.81 m/s2)
    x_upstream   : float, coordenada x de la cara aguas arriba [m] (0.0 m)
    tol          : float, tolerancia geometrica para seleccionar nodos

    Devuelve
    --------
    F_hydro : ndarray de tamano (number_of_dofs,) con las fuerzas nodales aplicadas en los DOFs u_x.
    info    : dict con metadatos (nodos afectados, fuerza total resultante, momento respecto al origen)
    """
    number_of_dofs = mesh.number_of_dofs()
    F_hydro = numpy.zeros(number_of_dofs, dtype=float)

    # Filtrar nodos sobre la cara aguas arriba y dentro de la cota de agua
    upstream_nodes = [
        node for node in mesh.nodes
        if abs(node.x - x_upstream) <= tol and -tol <= node.y <= water_height + tol
    ]

    if not upstream_nodes:
        raise ValueError("No se encontraron nodos en la cara aguas arriba (x=%f, y in [0, %f])" % (x_upstream, water_height))

    # Ordenar nodos por cota y creciente
    upstream_nodes.sort(key=lambda n: n.y)

    # Detectar el orden de interpolacion predominante en la malla
    order = mesh.element_orders[0] if mesh.element_orders else 1

    # Espesor de la presa (obtenido del primer material de la malla)
    first_mat_id = list(mesh.thicknesses.keys())[0] if mesh.thicknesses else None
    thickness = mesh.thicknesses[first_mat_id] if first_mat_id in mesh.thicknesses else 1.0

    def pressure(y):
        return rho_water * gravity * max(0.0, water_height - y)

    nodal_loads = {node.global_id: 0.0 for node in upstream_nodes}

    if order == 1:
        # Segmentos lineales de 2 nodos consecutivos
        for k in range(len(upstream_nodes) - 1):
            n1 = upstream_nodes[k]
            n2 = upstream_nodes[k + 1]
            length = n2.y - n1.y
            if length <= 0.0:
                continue

            p1 = pressure(n1.y)
            p2 = pressure(n2.y)

            # Integracion consistente lineal:
            # f1 = integral N1(s) p(s) t ds = t * L * (2*p1 + p2) / 6
            # f2 = integral N2(s) p(s) t ds = t * L * (p1 + 2*p2) / 6
            f1 = thickness * length * (2.0 * p1 + p2) / 6.0
            f2 = thickness * length * (p1 + 2.0 * p2) / 6.0

            nodal_loads[n1.global_id] += f1
            nodal_loads[n2.global_id] += f2

    elif order == 2:
        # Segmentos cuadraticos de 3 nodos (dos vertices y un nodo medio)
        # Cuadratura de Gauss de 3 puntos en [-1, 1]
        gp = [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]
        gw = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

        step = 2
        for k in range(0, len(upstream_nodes) - 1, step):
            if k + 2 >= len(upstream_nodes):
                # Caso borde si quedase un segmento impar (fallback a lineal)
                n1 = upstream_nodes[k]
                n2 = upstream_nodes[k + 1]
                length = n2.y - n1.y
                p1 = pressure(n1.y)
                p2 = pressure(n2.y)
                nodal_loads[n1.global_id] += thickness * length * (2.0 * p1 + p2) / 6.0
                nodal_loads[n2.global_id] += thickness * length * (p1 + 2.0 * p2) / 6.0
                continue

            n1 = upstream_nodes[k]
            n2 = upstream_nodes[k + 1]  # nodo intermedio
            n3 = upstream_nodes[k + 2]  # vertice superior
            length = n3.y - n1.y
            y_mid = 0.5 * (n1.y + n3.y)

            for xi, w in zip(gp, gw):
                y_val = y_mid + xi * (0.5 * length)
                p_val = pressure(y_val)

                # Funciones de forma 1D cuadraticas
                N1 = 0.5 * xi * (xi - 1.0)
                N2 = 1.0 - xi * xi
                N3 = 0.5 * xi * (xi + 1.0)

                det_jac = 0.5 * length
                coeff = p_val * thickness * det_jac * w

                nodal_loads[n1.global_id] += N1 * coeff
                nodal_loads[n2.global_id] += N2 * coeff
                nodal_loads[n3.global_id] += N3 * coeff
    else:
        raise ValueError("Unsupported interpolation order: %d" % order)

    # Ensamblar en el vector de fuerzas global (en el DOF u_x correspondiente)
    total_force_x = 0.0
    overturning_moment_origin = 0.0

    for node in upstream_nodes:
        nid = node.global_id
        fx = nodal_loads[nid]
        # DOF u_x del nodo: global_dofs()[0]
        dof_x = mesh.node_global_dofs(nid)[0]
        F_hydro[dof_x] += fx

        total_force_x += fx
        overturning_moment_origin += fx * node.y

    # Centro de presiones equivalente medido desde y=0
    center_of_pressure_y = overturning_moment_origin / total_force_x if total_force_x > 0 else 0.0

    info = {
        "upstream_nodes": [n.global_id for n in upstream_nodes],
        "nodal_forces": nodal_loads,
        "total_force_x": total_force_x,
        "overturning_moment_origin": overturning_moment_origin,
        "center_of_pressure_y": center_of_pressure_y,
    }

    return F_hydro, info


def compute_body_forces(mesh, elements, gravity=9.81):
    """Calcula las fuerzas de volumen debidas al peso propio (bx = 0, by = -rho * g).

    Devuelve un vector de fuerzas global de tamano (number_of_dofs,).
    """
    F_body = numpy.zeros(mesh.number_of_dofs(), dtype=float)

    for element in elements:
        rho = element.material.rho
        by = -rho * gravity
        fe = element.body_force_vector(0.0, by)
        dofs = element.global_dofs()

        for local_idx, dof in enumerate(dofs):
            F_body[dof] += fe[local_idx]

    return F_body
