"""main.py
-------
Script principal de ejecucion del analisis estatico de la presa de gravedad.
Modela el comportamiento mecanico 2D (deformacion plana) bajo peso propio y
empuje hidroestatico. La base (y=0) puede estar empotrada (Dirichlet) o
apoyada sobre una fundacion elastica de Winkler (condicion de Robin). Como la
presa se sostiene por peso propio, el suelo no puede traccionarla: los nodos
de la base con reaccion de traccion se liberan y se vuelve a resolver.

Permite seleccionar interactivamente el tipo de elemento (CST, LST, Q4, Q8),
resolver el sistema y visualizar la configuracion deformada amplificada y la
tension principal maxima sigma_1 sobre la deformada.

Uso:
    python main.py                       # interactivo
    python main.py Q8 5000               # base empotrada
    python main.py Q8 5000 robin 1e9     # base sobre resortes, kn = 1e9 N/m3
    python main.py ALL 2000              # los cuatro elementos, sin mostrar figuras
"""

import sys
import os
import numpy as np

from source.mesh.mesh import Mesh
from source.elements.factory import build_elements
from source.systems.system import System
from source.solvers.linear_solver import LinearSolver
from source.output.plotter import plot_deformed_mesh, plot_principal_stress
from source.output.vtk_writer import write_vtu
from source.output.stability import stability_report, print_stability_table


ELEMENT_MESHES = {
    "CST": "meshes/presa_cst_strain.mesh",
    "LST": "meshes/presa_lst_strain.mesh",
    "Q4":  "meshes/presa_q4_strain.mesh",
    "Q8":  "meshes/presa_q8_strain.mesh",
}

# Condicion de Robin: modulo de balasto tangencial kt = KT_RATIO * kn
DEFAULT_KN = 1.0e10     # [N/m3]
KT_RATIO = 0.4


def ask_element_choice():
    """Pregunta interactivamente al usuario que tipo de elemento desea utilizar."""
    options = list(ELEMENT_MESHES.keys())
    print("\n" + "=" * 70)
    print(" SELECCION DE TIPO DE ELEMENTO PARA LA PRESA DE GRAVEDAD")
    print("=" * 70)
    for i, name in enumerate(options, 1):
        desc = {
            "CST": "Triangular lineal (3 nodos por elemento)",
            "LST": "Triangular cuadratico (6 nodos por elemento)",
            "Q4":  "Cuadrangular bilineal (4 nodos por elemento)",
            "Q8":  "Cuadrangular serendipity cuadratico (8 nodos por elemento)",
        }[name]
        print("  [%d] %s : %s" % (i, name, desc))
    print()

    while True:
        raw = input("Seleccione una opcion (1-%d) o nombre del elemento [default: Q8]: " % len(options)).strip().upper()
        if raw == "":
            return "Q8"
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        elif raw in options:
            return raw
        print("Opcion invalida. Intente de nuevo.")


def ask_scale_factor():
    """Pregunta interactivamente el factor de escala para la deformada."""
    default_scale = 5000.0
    while True:
        raw = input("\nFactor de escala para amplificar deformaciones [default: %g]: " % default_scale).strip()
        if raw == "":
            return default_scale
        try:
            val = float(raw)
            if val > 0:
                return val
            print("El factor debe ser un numero positivo.")
        except ValueError:
            print("Entrada invalida. Ingrese un numero decimal o entero.")


def ask_base_condition():
    """Pregunta la condicion de borde de la base: empotrada o Robin (kn)."""
    print("\nCondicion de borde en la base (y=0):")
    print("  [1] Empotrada (ux = uy = 0)")
    print("  [2] Robin: fundacion elastica de Winkler (kt = %g kn)" % KT_RATIO)

    while True:
        raw = input("Seleccione una opcion [default: 1]: ").strip()
        if raw in ("", "1"):
            return "fixed", DEFAULT_KN
        if raw == "2":
            break
        print("Opcion invalida. Intente de nuevo.")

    while True:
        raw = input("Modulo de balasto normal kn [N/m3] [default: %g]: " % DEFAULT_KN).strip()
        if raw == "":
            return "robin", DEFAULT_KN
        try:
            val = float(raw)
            if val > 0:
                return "robin", val
            print("kn debe ser positivo.")
        except ValueError:
            print("Entrada invalida.")


def print_released_nodes(system, release_info):
    """Informa los nodos de la base liberados por estar traccionados."""
    released = release_info["released"]
    if not released:
        print("Base sin traccion: no se libero ningun nodo.")
        return
    print("Nodos de la base liberados por traccion (%d, en %d iteraciones):" % (
        len(released), release_info["iterations"]))
    for node_id in released:
        node = system.mesh.node(node_id)
        print("  nodo %d (x=%.3f, y=%.3f)" % (node_id + 1, node.x, node.y))
    for node_id in release_info["penetrating"]:
        print("  AVISO: el nodo liberado %d penetra el suelo (uy < 0)" % (node_id + 1))


def run_dam_analysis(
    mesh_file=None,
    water_height=28.0,
    gravity=9.81,
    scale_factor=5000.0,
    plot=True,
    show=True,
    cmap="viridis",
    base="fixed",
    kn=DEFAULT_KN,
    kt_ratio=KT_RATIO,
    sigma1_vlim=None,
    export_vtu=True,
    mu=0.7,
    release_tension=True,
):
    """Analisis estatico lineal de la presa.

    base : "fixed" (empotrada) o "robin" (resortes de Winkler con modulos
           kn [N/m3] normal y kt = kt_ratio * kn tangencial).
    release_tension : si es True, los nodos de la base cuya reaccion es de
           traccion quedan sin vinculo (libres) y se vuelve a resolver.
    """
    if base not in ("fixed", "robin"):
        raise ValueError("base debe ser 'fixed' o 'robin'")

    if mesh_file is None:
        mesh_file = "meshes/presa_q8_strain.mesh"

    print("=" * 70)
    print(" ANALISIS ESTATICO LINEAL - PRESA DE GRAVEDAD")
    print("=" * 70)
    print("Archivo de malla: %s" % mesh_file)

    # 1. Lectura de la malla y construccion de elementos
    mesh = Mesh()
    mesh.read(mesh_file)
    elements = build_elements(mesh)
    print("Nodos: %d | Elementos: %d (%s orden %d)" % (
        mesh.number_of_nodes(),
        len(elements),
        mesh.element_types[0],
        mesh.element_orders[0],
    ))

    # 2. Inicializacion del sistema y ensamblaje de rigidez K
    system = System(mesh, elements)
    system.assemble_stiffness()
    print("Matriz de rigidez global K ensamblada: %d x %d DOFs" % (
        system.number_of_dofs, system.number_of_dofs
    ))

    # 3. Aplicacion de cargas
    # 3a. Carga hidroestatica en x=0, 0 <= y <= water_height
    hydro_info = system.apply_hydrostatic_pressure(
        water_height=water_height,
        rho_water=1000.0,
        gravity=gravity,
    )
    # 3b. Peso propio volumetrico
    system.apply_gravity(gravity=gravity)

    print("\n--- CARGAS EXTERNAS APLICADAS ---")
    print("Empuje hidroestatico total Fx : %14.2f N" % hydro_info["total_force_x"])
    print("Centro de empuje hidroestatico: %14.4f m (teorico Hw/3 = %.4f m)" % (
        hydro_info["center_of_pressure_y"], water_height / 3.0
    ))
    print("Momento de vuelco (origen)    : %14.2f N*m" % hydro_info["overturning_moment_origin"])
    print("Peso propio total Fy          : %14.2f N" % np.sum(system.F[1::2]))

    # 4. Condiciones de borde en la base (y=0)
    print("\n--- CONDICIONES DE BORDE ---")
    if base == "fixed":
        # Empotramiento: ux = 0, uy = 0
        fixed_nodes = system.fix_base(y_val=0.0)
        print("Base empotrada (y=0): %d nodos (%d DOFs restringidos)" % (
            len(fixed_nodes), len(fixed_nodes) * 2
        ))
    else:
        # Robin: t = -diag(kt, kn) u sobre la base
        kt = kt_ratio * kn
        robin_info = system.apply_robin_base(kn=kn, kt=kt, y_val=0.0)
        print("Base sobre fundacion elastica (Robin): %d nodos, longitud %.4f m" % (
            len(robin_info["nodes"]), robin_info["length"]
        ))
        print("  kn = %.3e N/m3 | kt = %.3e N/m3" % (kn, kt))

    # 5. Resolucion del sistema de ecuaciones (K u = F + R)
    # La presa se sostiene por peso propio: el suelo no puede traccionarla,
    # asi que se liberan los nodos de la base con reaccion de traccion.
    print("\n--- RESOLUCION DEL SISTEMA ---")
    if release_tension:
        u, reactions, release_info = system.solve_without_base_tension(solver_func=LinearSolver.solve)
        print_released_nodes(system, release_info)
    else:
        u, reactions = system.solve(solver_func=LinearSolver.solve)
    print("Sistema resuelto exitosamente.")

    # 6. Reacciones de vinculo y equilibrio estatico
    base_reactions = system.get_base_reactions(y_val=0.0)
    sum_fx = np.sum(system.F[0::2])
    sum_fy = np.sum(system.F[1::2])
    sum_rx = base_reactions["total_Rx"]
    sum_ry = base_reactions["total_Ry"]

    print("\n--- REACCIONES EN LA BASE Y EQUILIBRIO GLOBAL ---")
    if base == "robin":
        print("(fuerzas de los resortes de la fundacion)")
    print("Reaccion horizontal total Rx  : %14.2f N" % sum_rx)
    print("Reaccion vertical total Ry    : %14.2f N" % sum_ry)
    print("Equilibrio horizontal sum(Fx)+Rx: %12.4e N" % (sum_fx + sum_rx))
    print("Equilibrio vertical   sum(Fy)+Ry: %12.4e N" % (sum_fy + sum_ry))

    # 6b. Tabla de estabilidad: momentos analiticos vs FEM y factores de seguridad
    stability = stability_report(
        system, water_height=water_height, gravity=gravity, mu=mu,
    )
    print()
    print_stability_table(stability)

    # 7. Desplazamientos nodales maximos
    u_x = u[0::2]
    u_y = u[1::2]
    u_mag = np.sqrt(u_x**2 + u_y**2)
    max_idx = np.argmax(u_mag)
    max_node = mesh.nodes[max_idx]

    print("\n--- DESPLAZAMIENTOS ---")
    print("Desplazamiento maximo: %.4f mm en nodo %d (x=%.2f, y=%.2f)" % (
        u_mag[max_idx] * 1000.0, max_node.global_id + 1, max_node.x, max_node.y
    ))
    print("  ux = %.4f mm, uy = %.4f mm" % (u_x[max_idx] * 1000.0, u_y[max_idx] * 1000.0))

    # 8. Tensiones por elemento (promedio en puntos de Gauss)
    # sigma_zz != 0 en deformacion plana (laderas impiden el desplazamiento en z)
    sigma = np.array([element.stress_at_centroid(u) for element in elements])
    sigma_1 = np.array([elements[0].principal_stresses_3d(s)[0] for s in sigma])

    print("\n--- TENSIONES POR ELEMENTO [MPa] (deformacion plana) ---")
    for label, values in (
        ("sigma_xx", sigma[:, 0]),
        ("sigma_yy", sigma[:, 1]),
        ("tau_xy", sigma[:, 2]),
        ("sigma_zz", sigma[:, 3]),
        ("sigma_1 (3D)", sigma_1),
    ):
        print("%-13s min = %9.4f   max = %9.4f" % (label, values.min() / 1e6, values.max() / 1e6))
    print("=" * 70)

    base_name = os.path.splitext(os.path.basename(mesh_file))[0]
    if base == "fixed":
        base_label = "base empotrada"
        suffix = ""
    else:
        kn_str = ("%.0e" % kn).replace("+0", "").replace("+", "")   # 1e+09 -> 1e9
        base_label = "base Robin $k_n$=%s N/m$^3$" % kn_str
        suffix = "_robin_kn%s" % kn_str

    # 9. Exportacion a VTU para ParaView (original + deformada amplificada)
    if export_vtu:
        vtu_path = os.path.join("results", "vtu", "%s%s.vtu" % (base_name, suffix))
        vtu_def_path = os.path.join("results", "vtu", "%s%s_deformada.vtu" % (base_name, suffix))
        write_vtu(vtu_path, mesh, elements, u, scale_factor=0.0)
        write_vtu(vtu_def_path, mesh, elements, u, scale_factor=scale_factor)
        print("VTU guardado en: %s" % vtu_path)
        print("VTU deformada (x%g) guardado en: %s" % (scale_factor, vtu_def_path))

    # 10. Visualizacion: deformada y sigma_1 sobre la deformada
    if plot:
        element_label = "%s orden %d" % (mesh.element_types[0], mesh.element_orders[0])

        title = "Configuracion Deformada - Presa de Gravedad (%s, %s)\nEscala: %g$\\times$ | Disp. max: %.4f mm" % (
            element_label, base_label, scale_factor, u_mag[max_idx] * 1000.0
        )
        plot_deformed_mesh(
            mesh,
            u,
            scale_factor=scale_factor,
            title=title,
            save_path=os.path.join("results", "deformada_%s%s.png" % (base_name, suffix)),
            show=show,
            cmap=cmap,
        )

        title = "$\\sigma_1$ sobre la deformada (%s, %s)\nEscala: %g$\\times$ | $\\sigma_1$ promedio por elemento en [%.3f, %.3f] MPa" % (
            element_label, base_label, scale_factor, sigma_1.min() / 1e6, sigma_1.max() / 1e6
        )
        plot_principal_stress(
            mesh,
            elements,
            u,
            scale_factor=scale_factor,
            vlim=sigma1_vlim,
            title=title,
            save_path=os.path.join("results", "sigma1_%s%s.png" % (base_name, suffix)),
            show=show,
        )

    return system


if __name__ == "__main__":
    # Permite pasar los parametros por consola:
    #   python main.py <CST|LST|Q4|Q8|ALL|archivo.mesh> [escala] [fixed|robin] [kn]
    # O bien correr interactivamente si no se pasa ningun argumento.
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg == "ALL":
            selected_meshes = list(ELEMENT_MESHES.values())
        elif arg in ELEMENT_MESHES:
            selected_meshes = [ELEMENT_MESHES[arg]]
        elif os.path.isfile(sys.argv[1]):
            selected_meshes = [sys.argv[1]]
        else:
            print("Argumento no reconocido: %s. Use CST, LST, Q4, Q8, ALL o ruta al archivo .mesh" % sys.argv[1])
            sys.exit(1)

        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 5000.0
        base = sys.argv[3].lower() if len(sys.argv) > 3 else "fixed"
        if base not in ("fixed", "robin"):
            print("Condicion de base no reconocida: %s. Use fixed o robin" % sys.argv[3])
            sys.exit(1)
        kn = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_KN
    else:
        element_choice = ask_element_choice()
        selected_meshes = [ELEMENT_MESHES[element_choice]]
        scale = ask_scale_factor()
        base, kn = ask_base_condition()

    for selected_mesh in selected_meshes:
        run_dam_analysis(
            mesh_file=selected_mesh,
            scale_factor=scale,
            plot=True,
            show=len(selected_meshes) == 1,
            base=base,
            kn=kn,
        )
