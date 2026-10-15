"""stability_table.py
-------------------
Paso 8: tabla formal de estabilidad de la presa para los cuatro elementos.

Compara el momento estabilizante (peso propio), el desestabilizante (empuje
hidrostatico) y el momento de las reacciones de vinculo del FEM en la base,
respecto del pie aguas abajo, y calcula los factores de seguridad al vuelco y
al deslizamiento.

Salidas:
  - Tabla detallada por elemento en consola.
  - Resumen comparativo en consola.
  - results/tabla_estabilidad.md (tabla en Markdown).

Uso:
    python stability_table.py            # base empotrada, mu = 0.7
    python stability_table.py 0.6        # otro coeficiente de friccion mu
"""

import io
import os
import sys
import contextlib

from main import ELEMENT_MESHES, run_dam_analysis
from source.output.stability import stability_report, print_stability_table


def main(mu=0.7):
    rows = []

    for name, mesh_file in ELEMENT_MESHES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            system = run_dam_analysis(
                mesh_file=mesh_file, plot=False, show=False, export_vtu=False, mu=mu,
            )
        rep = stability_report(system, mu=mu)
        print_stability_table(rep, label="[%s]" % name)
        print()
        rows.append((name, rep))

    # Resumen comparativo
    print("=" * 100)
    print(" RESUMEN (mu = %.2f)" % mu)
    print("=" * 100)
    header = "%-5s %14s %14s %16s %9s %9s %9s %9s" % (
        "Elem", "M_est [MN m]", "M_des [MN m]", "|M_R| FEM [MN m]", "x_R [m]", "FS vuelco", "FS desl.", "Err M_R")
    print(header)
    for name, r in rows:
        print("%-5s %14.3f %14.3f %16.3f %9.3f %9.3f %9.3f %9.1e" % (
            name, r["m_est"] / 1e6, r["m_des"] / 1e6, -r["m_react"] / 1e6,
            r["x_resultant"], r["fs_overturning"], r["fs_sliding"], r["err_m_react"]))
    print("=" * 100)

    # Markdown
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "tabla_estabilidad.md")
    r0 = rows[0][1]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Estabilidad de la presa (base empotrada, mu = %.2f)\n\n" % mu)
        f.write("Momentos respecto del pie aguas abajo (x = %.0f m, y = 0), por metro de espesor.\n\n" % r0["base"])
        f.write("## Solucion analitica\n\n")
        f.write("| Magnitud | Valor |\n|---|---|\n")
        f.write("| Area del perfil | %.2f m2 |\n" % r0["area"])
        f.write("| Peso propio W | %.1f N/m |\n" % r0["weight"])
        f.write("| Centroide xc | %.4f m (brazo B - xc = %.4f m) |\n" % (r0["xc"], r0["arm_weight"]))
        f.write("| M_est = W (B - xc) | %.1f N m/m |\n" % r0["m_est"])
        f.write("| Empuje Fh = rho_w g Hw^2 / 2 | %.1f N/m |\n" % r0["f_hydro"])
        f.write("| M_des = Fh Hw / 3 | %.1f N m/m |\n" % r0["m_des"])
        f.write("| FS vuelco = M_est / M_des | %.3f |\n" % r0["fs_overturning"])
        f.write("| FS deslizamiento = mu W / Fh | %.3f |\n\n" % r0["fs_sliding"])
        f.write("## Comparacion con el FEM\n\n")
        f.write("| Elemento | M_est carga FEM [N m/m] | M_des carga FEM [N m/m] | M_R reacciones [N m/m] | Err. M_R | Rx [N/m] | Ry [N/m] | x_R [m] | FS vuelco | FS desl. |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for name, r in rows:
            f.write("| %s | %.1f | %.1f | %.1f | %.1e | %.1f | %.1f | %.4f | %.3f | %.3f |\n" % (
                name, r["fem_m_est"], r["fem_m_des"], -r["m_react"], r["err_m_react"],
                r["sum_rx"], r["sum_ry"], r["x_resultant"], r["fem_fs_overturning"], r["fem_fs_sliding"]))
    print("Tabla guardada en: %s" % path)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.7)
