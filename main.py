"""main.py
-------
Script principal de ejecucion del analisis estatico de la presa de gravedad.
Modela el comportamiento mecanico 2D bajo peso propio y empuje hidroestatico
con condiciones de apoyo fijo en la base (y=0).

Permite seleccionar interactivamente el tipo de elemento (CST, LST, Q4, Q8),
resolver el sistema y visualizar la configuracion deformada amplificada.
"""

import sys
import os
import numpy as np

from source.mesh.mesh import Mesh
from source.elements.factory import build_elements
from source.systems.system import System
from source.solvers.linear_solver import LinearSolver
from source.output.plotter import plot_deformed_mesh


ELEMENT_MESHES = {
    "CST": "meshes/presa_cst_strain.mesh",
    "LST": "meshes/presa_lst_strain.mesh",
    "Q4":  "meshes/presa_q4_strain.mesh",
    "Q8":  "meshes/presa_q8_strain.mesh",
}


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


def run_dam_analysis(
    mesh_file=None,
    water_height=28.0,
    gravity=9.81,
    scale_factor=5000.0,
    plot=True,
    cmap="viridis",
):
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

    # 4. Condiciones de borde: base fija (y=0 -> ux=0, uy=0)
    fixed_nodes = system.fix_base(y_val=0.0)
    print("\n--- CONDICIONES DE BORDE ---")
    print("Nodos fijados en la base (y=0): %d nodos (%d DOFs restringidos)" % (
        len(fixed_nodes), len(fixed_nodes) * 2
    ))

    # 5. Resolucion del sistema de ecuaciones (K u = F + R)
    u, reactions = system.solve(solver_func=LinearSolver.solve)
    print("\n--- RESOLUCION DEL SISTEMA ---")
    print("Sistema resuelto exitosamente.")

    # 6. Reacciones de vinculo y equilibrio estatico
    base_reactions = system.get_base_reactions(y_val=0.0)
    sum_fx = np.sum(system.F[0::2])
    sum_fy = np.sum(system.F[1::2])
    sum_rx = base_reactions["total_Rx"]
    sum_ry = base_reactions["total_Ry"]

    print("\n--- REACCIONES EN LA BASE Y EQUILIBRIO GLOBAL ---")
    print("Reaccion horizontal total Rx  : %14.2f N" % sum_rx)
    print("Reaccion vertical total Ry    : %14.2f N" % sum_ry)
    print("Equilibrio horizontal sum(Fx)+Rx: %12.4e N" % (sum_fx + sum_rx))
    print("Equilibrio vertical   sum(Fy)+Ry: %12.4e N" % (sum_fy + sum_ry))

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
    print("=" * 70)

    # 8. Visualizacion de configuracion deformada
    if plot:
        base_name = os.path.splitext(os.path.basename(mesh_file))[0]
        save_file = os.path.join("results", "deformada_%s.png" % base_name)
        title = "Configuracion Deformada - Presa de Gravedad (%s orden %d)\nEscala: %g$\\times$ | Disp. max: %.4f mm" % (
            mesh.element_types[0], mesh.element_orders[0], scale_factor, u_mag[max_idx] * 1000.0
        )
        plot_deformed_mesh(
            mesh,
            u,
            scale_factor=scale_factor,
            title=title,
            save_path=save_file,
            show=True,
            cmap=cmap,
        )

    return system


if __name__ == "__main__":
    # Permite pasar el tipo de elemento por argumento de consola:
    # Ejemplo: python main.py CST 5000
    # O bien correr interactivamente si no se pasa ningun argumento.
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in ELEMENT_MESHES:
            selected_mesh = ELEMENT_MESHES[arg]
        elif os.path.isfile(sys.argv[1]):
            selected_mesh = sys.argv[1]
        else:
            print("Argumento no reconocido: %s. Use CST, LST, Q4, Q8 o ruta al archivo .mesh" % sys.argv[1])
            sys.exit(1)

        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 5000.0
    else:
        element_choice = ask_element_choice()
        selected_mesh = ELEMENT_MESHES[element_choice]
        scale = ask_scale_factor()

    run_dam_analysis(mesh_file=selected_mesh, scale_factor=scale, plot=True)
