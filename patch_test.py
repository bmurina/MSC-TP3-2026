"""patch_test.py
--------------
Patch test lineal para elementos 2D de estado plano de tension.

El patch test verifica que el elemento reproduce exactamente un campo de
deformaciones constante. Se impone el campo:
    u(x, y) = a0 + a1 * x + a2 * y
    v(x, y) = b0 + b1 * x + b2 * y

con deformaciones exactas constantes:
    eps_xx = a1
    eps_yy = b2
    gamma_xy = a2 + b1

Se aplican los desplazamientos exactos como condicion de Dirichlet en TODOS
los nodos del contorno exterior. Se verifica que:
  1. Los desplazamientos interiores coinciden con los exactos (error < tol).
  2. Las tensiones en TODOS los elementos son constantes e iguales a las exactas.

Elementos disponibles:
    CST  -> meshes/tri_cst.mesh
    LST  -> meshes/tri_lst.mesh
    Q4   -> meshes/quad_q4.mesh
    Q8   -> meshes/quad_q8.mesh

Uso:
    python patch_test.py          # pide el tipo de elemento interactivamente
    python patch_test.py CST      # ejecuta directamente el CST
    python patch_test.py all      # ejecuta los cuatro elementos
"""

import sys
import numpy as np

from source.mesh.mesh import Mesh
from source.elements.factory import build_elements
from source.systems.matrix import DenseMatrix
from source.systems.boundary_conditions import BoundaryConditions
from source.solvers.linear_solver import LinearSolver


# ---------------------------------------------------------------------------
# Configuracion del patch test
# ---------------------------------------------------------------------------
# Campo de desplazamientos lineales impuesto
PATCH_PARAMS = {
    "a0": 0.0,   # u = a0 + a1 x + a2 y
    "a1": 1.0e-3,
    "a2": 0.5e-3,
    "b0": 0.0,   # v = b0 + b1 x + b2 y
    "b1": 0.5e-3,
    "b2": -0.3e-3,
}

MESH_MAP = {
    "CST": "meshes/tri_cst.mesh",
    "LST": "meshes/tri_lst.mesh",
    "Q4":  "meshes/quad_q4.mesh",
    "Q8":  "meshes/quad_q8.mesh",
}

TOL_DISP = 1e-10   # tolerancia en desplazamientos [m]
TOL_STRESS = 1e-2  # tolerancia en tensiones [Pa] (numerico)


# ---------------------------------------------------------------------------
# Funciones del patch test
# ---------------------------------------------------------------------------
def exact_displacement(x, y, params=PATCH_PARAMS):
    """Desplazamiento exacto (u, v) del campo lineal."""
    u = params["a0"] + params["a1"] * x + params["a2"] * y
    v = params["b0"] + params["b1"] * x + params["b2"] * y
    return u, v


def exact_strains(params=PATCH_PARAMS):
    """Deformaciones exactas constantes del campo lineal."""
    eps_xx  = params["a1"]
    eps_yy  = params["b2"]
    gamma_xy = params["a2"] + params["b1"]
    return np.array([eps_xx, eps_yy, gamma_xy])


def exact_stresses(material, params=PATCH_PARAMS):
    """Tensiones exactas constantes: sigma = D @ eps_exact."""
    eps = exact_strains(params)
    return material.plane_stress_matrix() @ eps


def get_boundary_nodes(mesh, tol=1e-8):
    """Devuelve el conjunto de IDs de nodos sobre el contorno exterior de la malla.

    Para el patch test se usan mallas rectangulares alineadas con los ejes.
    Se detecta el borde como los nodos que tienen coordenada minima o maxima en x o y.
    """
    xs = [n.x for n in mesh.nodes]
    ys = [n.y for n in mesh.nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    boundary = set()
    for n in mesh.nodes:
        on_boundary = (
            abs(n.x - x_min) < tol or
            abs(n.x - x_max) < tol or
            abs(n.y - y_min) < tol or
            abs(n.y - y_max) < tol
        )
        if on_boundary:
            boundary.add(n.global_id)
    return boundary


def run_patch_test(element_type, verbose=True):
    """Ejecuta el patch test para el tipo de elemento indicado.

    Parametros
    ----------
    element_type : str, uno de: 'CST', 'LST', 'Q4', 'Q8'
    verbose      : bool, si True imprime resultados detallados

    Devuelve
    --------
    passed : bool
    errors : dict con metricas de error
    """
    if element_type not in MESH_MAP:
        raise ValueError("Tipo de elemento desconocido: %s. Opciones: %s" % (element_type, list(MESH_MAP.keys())))

    mesh_file = MESH_MAP[element_type]

    if verbose:
        print("\n" + "=" * 65)
        print("  PATCH TEST  -  Elemento: %s" % element_type)
        print("  Malla: %s" % mesh_file)
        print("=" * 65)

    # 1. Cargar la malla y construir los elementos
    mesh = Mesh()
    mesh.read(mesh_file)
    elements = build_elements(mesh)
    n_dofs = mesh.number_of_dofs()

    if verbose:
        print("  Nodos: %d | Elementos: %d | DOFs: %d" % (mesh.number_of_nodes(), len(elements), n_dofs))
        print("\n  Campo impuesto:")
        print("    u(x,y) = %.2e + %.2e*x + %.2e*y" % (PATCH_PARAMS["a0"], PATCH_PARAMS["a1"], PATCH_PARAMS["a2"]))
        print("    v(x,y) = %.2e + %.2e*x + %.2e*y" % (PATCH_PARAMS["b0"], PATCH_PARAMS["b1"], PATCH_PARAMS["b2"]))
        eps_ex = exact_strains()
        print("  Deformaciones exactas: eps_xx=%.3e  eps_yy=%.3e  gamma_xy=%.3e" % tuple(eps_ex))

    # 2. Ensamblar la rigidez global
    K = DenseMatrix(n_dofs)
    for elem in elements:
        K.scatter_local_to_global(elem.stiffness_matrix(), elem.global_dofs())

    # 3. Condiciones de borde de Dirichlet en todos los nodos del contorno
    bc = BoundaryConditions(n_dofs)
    boundary_ids = get_boundary_nodes(mesh)
    interior_ids = {n.global_id for n in mesh.nodes} - boundary_ids

    for nid in boundary_ids:
        node = mesh.nodes[nid]
        u_ex, v_ex = exact_displacement(node.x, node.y)
        dofs = mesh.node_global_dofs(nid)
        bc.fix_dof(dofs[0], u_ex)
        bc.fix_dof(dofs[1], v_ex)

    if verbose:
        print("\n  Nodos contorno: %d (Dirichlet)" % len(boundary_ids))
        print("  Nodos interiores: %d" % len(interior_ids))

    # 4. Resolver el sistema particionado
    F = np.zeros(n_dofs)
    free_dofs = bc.get_free_dofs()
    fixed_dofs = bc.get_fixed_dofs()

    u_sol = np.zeros(n_dofs)
    u_sol[fixed_dofs] = bc.get_prescribed_values()

    K_mat = K.to_numpy()

    if free_dofs:
        K_ff = K_mat[np.ix_(free_dofs, free_dofs)]
        K_fp = K_mat[np.ix_(free_dofs, fixed_dofs)]
        F_eff = F[free_dofs] - K_fp @ u_sol[fixed_dofs]
        u_sol[free_dofs] = LinearSolver.solve(K_ff, F_eff)

    # 5. Verificacion de desplazamientos en nodos interiores
    max_disp_error = 0.0
    if verbose and interior_ids:
        print("\n  --- Verificacion de desplazamientos (nodos interiores) ---")
        print("  %6s  %8s  %8s  %12s  %12s  %12s  %12s" % ("Nodo", "x", "y", "u_FEM", "u_exac", "v_FEM", "v_exac"))

    for nid in sorted(interior_ids):
        node = mesh.nodes[nid]
        dofs = mesh.node_global_dofs(nid)
        u_fem = u_sol[dofs[0]]
        v_fem = u_sol[dofs[1]]
        u_ex, v_ex = exact_displacement(node.x, node.y)

        err = max(abs(u_fem - u_ex), abs(v_fem - v_ex))
        max_disp_error = max(max_disp_error, err)

        if verbose:
            print("  %6d  %8.4f  %8.4f  %12.6e  %12.6e  %12.6e  %12.6e" % (
                nid + 1, node.x, node.y, u_fem, u_ex, v_fem, v_ex))

    # 6. Verificacion de tensiones en cada elemento
    if verbose:
        print("\n  --- Verificacion de tensiones en cada elemento ---")

    mat = list(mesh.materials.values())[0]
    sigma_exact = exact_stresses(mat)
    max_stress_error = 0.0

    if verbose:
        print("  Tensiones exactas: sigma_xx=%.4f Pa | sigma_yy=%.4f Pa | tau_xy=%.4f Pa" % tuple(sigma_exact))
        print()
        print("  %6s  %14s  %14s  %14s  %10s  %10s  %10s" % (
            "Elem", "sigma_xx", "sigma_yy", "tau_xy", "err_xx", "err_yy", "err_xy"))

    for elem in elements:
        sigma_fem = elem.stress_at_centroid(u_sol)
        err = np.abs(sigma_fem - sigma_exact)
        max_err = np.max(err)
        max_stress_error = max(max_stress_error, max_err)

        if verbose:
            print("  %6d  %14.4f  %14.4f  %14.4f  %10.2e  %10.2e  %10.2e" % (
                elem.global_id + 1,
                sigma_fem[0], sigma_fem[1], sigma_fem[2],
                err[0], err[1], err[2],
            ))

    # 7. Veredicto
    passed = (max_disp_error <= TOL_DISP) and (max_stress_error <= TOL_STRESS)

    if verbose:
        print()
        print("  --- Resumen ---")
        print("  Error maximo desplazamientos: %.4e m  (tol = %.1e)" % (max_disp_error, TOL_DISP))
        print("  Error maximo tensiones:       %.4e Pa (tol = %.1e)" % (max_stress_error, TOL_STRESS))
        status = "[PASSED]" if passed else "[FAILED]"
        print("\n  Resultado: %s" % status)
        print("=" * 65)

    return passed, {
        "element_type": element_type,
        "max_disp_error": max_disp_error,
        "max_stress_error": max_stress_error,
        "passed": passed,
    }


def ask_element_type():
    """Pide interactivamente el tipo de elemento a evaluar."""
    options = list(MESH_MAP.keys()) + ["ALL"]
    print("\n" + "=" * 65)
    print("  PATCH TEST - Verificacion de elementos 2D FEM")
    print("=" * 65)
    print("  Elementos disponibles:")
    for i, name in enumerate(options, 1):
        desc = {
            "CST": "Triangulo de Deformacion Constante (3 nodos)",
            "LST": "Triangulo de Deformacion Lineal   (6 nodos)",
            "Q4":  "Cuadrilatero Bilineal Isoparametrico (4 nodos)",
            "Q8":  "Cuadrilatero Serendipity Cuadratico  (8 nodos)",
            "ALL": "Ejecutar todos los elementos",
        }[name]
        print("    [%d] %s - %s" % (i, name, desc))
    print()

    while True:
        raw = input("  Seleccione una opcion (1-%d) o nombre del elemento: " % len(options)).strip().upper()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        elif raw in options:
            return raw
        print("  Opcion invalida. Intente nuevamente.")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        choice = sys.argv[1].upper()
    else:
        choice = ask_element_type()

    if choice == "ALL":
        results = []
        for etype in MESH_MAP.keys():
            passed, errs = run_patch_test(etype)
            results.append(errs)

        print("\n" + "=" * 65)
        print("  RESUMEN GLOBAL DEL PATCH TEST")
        print("=" * 65)
        print("  %-6s  %20s  %20s  %10s" % ("Tipo", "Err. Desplaz. [m]", "Err. Tension [Pa]", "Resultado"))
        for r in results:
            status = "PASSED" if r["passed"] else "FAILED"
            print("  %-6s  %20.4e  %20.4e  %10s" % (
                r["element_type"], r["max_disp_error"], r["max_stress_error"], status))
        all_passed = all(r["passed"] for r in results)
        print()
        print("  >> Resultado global: %s" % ("TODOS PASARON" if all_passed else "ALGUNO FALLO"))
        print("=" * 65)
    elif choice in MESH_MAP:
        run_patch_test(choice)
    else:
        print("ERROR: Tipo de elemento '%s' desconocido. Opciones: %s o ALL" % (choice, list(MESH_MAP.keys())))
        sys.exit(1)
