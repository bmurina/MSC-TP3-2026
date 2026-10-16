import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from source.mesh.mesh import Mesh
from source.elements.factory import build_elements
from source.systems.system import System
from source.solvers.linear_solver import LinearSolver


MESHES = [
    "meshes/presa_cst_strain.mesh",
    "meshes/presa_lst_strain.mesh",
    "meshes/presa_q4_strain.mesh",
    "meshes/presa_q8_strain.mesh",
]


def test_presa_equilibrium():
    for mesh_path in MESHES:
        mesh = Mesh()
        mesh.read(mesh_path)
        elems = build_elements(mesh)

        fem_sys = System(mesh, elems)
        fem_sys.assemble_stiffness()

        # Aplicar cargas: hidrostatica en x=0 (0 <= y <= 28) y peso propio
        hydro_info = fem_sys.apply_hydrostatic_pressure(water_height=28.0, rho_water=1000.0, gravity=9.81)
        f_body = fem_sys.apply_gravity(gravity=9.81)

        # Condiciones de borde: nodos en y=0 fijos (ux = 0, uy = 0)
        fixed_nodes = fem_sys.fix_base(y_val=0.0)

        # Resolver
        u, R = fem_sys.solve(solver_func=LinearSolver.solve)

        # Reacciones
        base_info = fem_sys.get_base_reactions(y_val=0.0)

        total_Fx = hydro_info["total_force_x"]
        total_Fy = float(np.sum(f_body))
        sum_Rx = base_info["total_Rx"]
        sum_Ry = base_info["total_Ry"]

        # Verificacion de equilibrio estatico: sum(F) + sum(R) == 0
        eq_x = abs(total_Fx + sum_Rx)
        eq_y = abs(total_Fy + sum_Ry)

        # Desplazamiento maximo
        u_mag = np.sqrt(u[0::2] ** 2 + u[1::2] ** 2)
        max_u_mm = np.max(u_mag) * 1000.0

        print("=" * 65)
        print("Malla: %s" % mesh_path)
        print("  Nodos: %d | Elementos: %d | Nodos en base: %d" % (mesh.number_of_nodes(), len(elems), len(fixed_nodes)))
        print("  Fuerza hidroestatica Fx:  %12.2f N (y_cp = %.4f m)" % (total_Fx, hydro_info["center_of_pressure_y"]))
        print("  Peso propio Fy:           %12.2f N" % total_Fy)
        print("  Reaccion base Rx:         %12.2f N" % sum_Rx)
        print("  Reaccion base Ry:         %12.2f N" % sum_Ry)
        print("  Residual equilibrio X:    %12.4e N" % eq_x)
        print("  Residual equilibrio Y:    %12.4e N" % eq_y)
        print("  Desplazamiento maximo:    %12.4f mm" % max_u_mm)

        assert eq_x < 1e-4, "Error de equilibrio en X"
        assert eq_y < 1e-4, "Error de equilibrio en Y"



def _solve_presa(mesh_path, base, kn=None, kt_ratio=0.4):
    mesh = Mesh()
    mesh.read(mesh_path)
    fem_sys = System(mesh, build_elements(mesh))
    fem_sys.assemble_stiffness()
    fem_sys.apply_hydrostatic_pressure(water_height=28.0, rho_water=1000.0, gravity=9.81)
    fem_sys.apply_gravity(gravity=9.81)

    if base == "fixed":
        fem_sys.fix_base(y_val=0.0)
    else:
        info = fem_sys.apply_robin_base(kn=kn, kt=kt_ratio * kn, y_val=0.0)
        assert abs(info["length"] - 18.0) < 1e-9, "Longitud de base integrada incorrecta"

    fem_sys.solve(solver_func=LinearSolver.solve)
    return fem_sys


def test_presa_robin_equilibrium():
    """Con resortes en la base, las fuerzas de la fundacion equilibran las cargas."""
    for mesh_path in MESHES:
        fem_sys = _solve_presa(mesh_path, "robin", kn=1.0e9)
        base_info = fem_sys.get_base_reactions(y_val=0.0)

        eq_x = abs(np.sum(fem_sys.F[0::2]) + base_info["total_Rx"])
        eq_y = abs(np.sum(fem_sys.F[1::2]) + base_info["total_Ry"])

        print("%s | Robin kn=1e9 | residual X: %.3e N | residual Y: %.3e N" % (mesh_path, eq_x, eq_y))
        assert eq_x < 1e-4, "Error de equilibrio en X (Robin)"
        assert eq_y < 1e-4, "Error de equilibrio en Y (Robin)"


def test_robin_stiff_limit():
    """Con kn muy grande la solucion de Robin tiende a la empotrada."""
    for mesh_path in MESHES:
        u_fixed = _solve_presa(mesh_path, "fixed").u
        u_robin = _solve_presa(mesh_path, "robin", kn=1.0e14).u

        rel_err = np.linalg.norm(u_robin - u_fixed) / np.linalg.norm(u_fixed)
        print("%s | Robin kn=1e14 vs empotrada: error relativo %.3e" % (mesh_path, rel_err))
        assert rel_err < 1e-2, "Robin rigido no converge al empotramiento"


def test_base_without_tension():
    """Al liberar los nodos traccionados la base queda solo comprimida y se
    mantiene el equilibrio (empotrada y Robin)."""
    for mesh_path in MESHES:
        for base, kn in (("fixed", None), ("robin", 1.0e10)):
            mesh = Mesh()
            mesh.read(mesh_path)
            fem_sys = System(mesh, build_elements(mesh))
            fem_sys.assemble_stiffness()
            fem_sys.apply_hydrostatic_pressure(water_height=28.0, rho_water=1000.0, gravity=9.81)
            fem_sys.apply_gravity(gravity=9.81)
            if base == "fixed":
                fem_sys.fix_base(y_val=0.0)
            else:
                fem_sys.apply_robin_base(kn=kn, kt=0.4 * kn, y_val=0.0)

            _, _, info = fem_sys.solve_without_base_tension(solver_func=LinearSolver.solve)
            base_info = fem_sys.get_base_reactions(y_val=0.0)

            min_ry = min(n["Ry"] for n in base_info["nodes"])
            eq_x = abs(np.sum(fem_sys.F[0::2]) + base_info["total_Rx"])
            eq_y = abs(np.sum(fem_sys.F[1::2]) + base_info["total_Ry"])

            print("%s | %s | liberados: %s | min Ry: %.3e N" % (mesh_path, base, info["released"], min_ry))
            assert min_ry > -1e-3, "Quedan nodos traccionados en la base"
            assert not info["penetrating"], "Un nodo liberado penetra el suelo"
            assert eq_x < 1e-4 and eq_y < 1e-4, "Error de equilibrio sin traccion"


if __name__ == "__main__":
    test_presa_equilibrium()
    test_presa_robin_equilibrium()
    test_robin_stiff_limit()
    test_base_without_tension()
