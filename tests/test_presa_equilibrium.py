import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from source.mesh.mesh import Mesh
from source.elements.factory import build_elements
from source.systems.system import System
from source.solvers.linear_solver import LinearSolver


def test_presa_equilibrium():
    meshes = [
        "meshes/presa_cst_strain.mesh",
        "meshes/presa_lst_strain.mesh",
        "meshes/presa_q4_strain.mesh",
        "meshes/presa_q8_strain.mesh",
    ]

    for mesh_path in meshes:
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


if __name__ == "__main__":
    test_presa_equilibrium()
