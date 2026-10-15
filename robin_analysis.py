"""robin_analysis.py
------------------
Punto bonus: analisis de la presa con condicion de borde de Robin en la base.

En lugar del empotramiento (ux = uy = 0 en y = 0) la base se apoya sobre una
fundacion elastica de Winkler, t = -diag(kt, kn) u, con kt = 0.4 kn. Se barre
el modulo de balasto normal kn = 1e8 ... 1e11 N/m3 (de roca blanda al limite
rigido) y se compara con el caso empotrado y con la solucion de cuerpo rigido
de Navier para sigma_yy en la base:

    sigma_yy(x) = -N/(B t) - M_c (x - B/2) / (t B^3 / 12)

con N y M_c (respecto del centro de la base) obtenidos de las reacciones.

Salidas:
  - Tabla comparativa en consola.
  - results/robin_base_syy_<malla>.png: sigma_yy a lo largo de la base.
  - results/deformada_*_robin_kn*.png y results/sigma1_*_robin_kn*.png
    (sigma_1 sobre la deformada para cada kn).

Uso:
    python robin_analysis.py            # Q8
    python robin_analysis.py LST 2000   # elemento y factor de escala
    python robin_analysis.py ALL
"""

import io
import os
import sys
import contextlib

import numpy as np
import matplotlib.pyplot as plt

from main import ELEMENT_MESHES, KT_RATIO, run_dam_analysis


KN_VALUES = [1.0e8, 1.0e9, 1.0e10, 1.0e11]   # [N/m3]


def kn_label(kn):
    """1e9 -> '1e9' (para leyendas y tablas)."""
    return ("%.0e" % kn).replace("+0", "").replace("+", "")


def base_sigma_yy(system, y_val=0.0, tol=1e-5):
    """sigma_yy en los nodos de la base, promediando los valores nodales de
    los elementos que comparten cada nodo. Devuelve (x, sigma_yy) ordenados."""
    sums = {}
    counts = {}

    for element in system.elements:
        sigma = element.stress_at_nodes(system.u)
        for i, nid in enumerate(element.node_global_ids):
            if abs(system.mesh.node(nid).y - y_val) <= tol:
                sums[nid] = sums.get(nid, 0.0) + sigma[i, 1]
                counts[nid] = counts.get(nid, 0) + 1

    nodes = sorted(sums.keys(), key=lambda n: system.mesh.node(n).x)
    x = np.array([system.mesh.node(n).x for n in nodes])
    syy = np.array([sums[n] / counts[n] for n in nodes])
    return x, syy


def summarize(system, label):
    """Magnitudes de comparacion de un caso resuelto."""
    u = system.u
    u_mag = np.sqrt(u[0::2] ** 2 + u[1::2] ** 2)
    i_max = int(np.argmax(u_mag))

    sigma = np.array([e.stress_at_centroid(u) for e in system.elements])
    principal = np.array([system.elements[0].principal_stresses_3d(s) for s in sigma])

    reactions = system.get_base_reactions(y_val=0.0)
    xs = np.array([n["x"] for n in reactions["nodes"]])
    ry = np.array([n["Ry"] for n in reactions["nodes"]])
    x_heel, x_toe = xs.min(), xs.max()
    width = x_toe - x_heel
    x_center = 0.5 * (x_heel + x_toe)

    thickness = system.elements[0].thickness
    N = reactions["total_Ry"]
    M_center = np.sum(ry * (xs - x_center))
    M_toe = np.sum(ry * (xs - x_toe))

    x_base, syy_base = base_sigma_yy(system)

    return {
        "label": label,
        "u_max": u_mag[i_max],
        "ux_max": u[2 * i_max],
        "uy_max": u[2 * i_max + 1],
        "s1_max": principal[:, 0].max(),
        "s3_min": principal[:, 2].min(),
        "syy_heel": syy_base[0],
        "syy_toe": syy_base[-1],
        "M_toe": M_toe,
        "x_resultant": np.sum(ry * xs) / N,
        "x_base": x_base,
        "syy_base": syy_base,
        "navier": (N, M_center, x_center, width, thickness),
    }


def navier_sigma_yy(x, navier):
    """Tension de contacto de cuerpo rigido (flexion compuesta) en la base."""
    N, M_center, x_center, width, thickness = navier
    inertia = thickness * width ** 3 / 12.0
    return -N / (width * thickness) - M_center * (x - x_center) / inertia


def run_robin_study(element, scale_factor=2000.0):
    mesh_file = ELEMENT_MESHES[element]
    base_name = os.path.splitext(os.path.basename(mesh_file))[0]

    print("=" * 96)
    print(" CONDICION DE ROBIN EN LA BASE - %s (%s), kt = %g kn" % (element, mesh_file, KT_RATIO))
    print("=" * 96)

    cases = [("fixed", None)] + [("robin", kn) for kn in KN_VALUES]
    results = []

    for base, kn in cases:
        label = "Empotrada" if base == "fixed" else "kn = %s" % kn_label(kn)

        # La salida detallada de cada corrida se omite; se resume en la tabla
        with contextlib.redirect_stdout(io.StringIO()):
            system = run_dam_analysis(
                mesh_file=mesh_file,
                scale_factor=scale_factor,
                plot=(base == "robin"),
                show=False,
                base=base,
                kn=kn if kn is not None else 1.0,
            )

        results.append(summarize(system, label))

    # Tabla comparativa
    print("%-12s %9s %9s %9s %10s %10s %11s %11s %13s %9s" % (
        "Caso", "u_max", "ux", "uy", "sigma1_max", "sigma3_min",
        "syy_talon", "syy_pie", "M_R pie B", "x_R"))
    print("%-12s %9s %9s %9s %10s %10s %11s %11s %13s %9s" % (
        "", "[mm]", "[mm]", "[mm]", "[MPa]", "[MPa]", "[MPa]", "[MPa]", "[MN.m/m]", "[m]"))
    print("-" * 96)
    for r in results:
        print("%-12s %9.4f %9.4f %9.4f %10.4f %10.4f %11.4f %11.4f %13.4f %9.4f" % (
            r["label"], r["u_max"] * 1e3, r["ux_max"] * 1e3, r["uy_max"] * 1e3,
            r["s1_max"] / 1e6, r["s3_min"] / 1e6,
            r["syy_heel"] / 1e6, r["syy_toe"] / 1e6,
            r["M_toe"] / 1e6, r["x_resultant"],
        ))

    x_nav = np.linspace(results[0]["x_base"][0], results[0]["x_base"][-1], 50)
    syy_nav = navier_sigma_yy(x_nav, results[0]["navier"])
    print("-" * 96)
    print("%-12s %83s %11.4f %11.4f" % ("Navier", "", syy_nav[0] / 1e6, syy_nav[-1] / 1e6))
    print("(sigma_* promedio por elemento; syy en la base promediado en nodos; momento de reacciones")
    print(" respecto del pie B, antihorario positivo: igual en todos los casos por equilibrio)")

    # Grafico de sigma_yy en la base
    fig, ax = plt.subplots(figsize=(8, 5))
    blues = plt.get_cmap("Blues")(np.linspace(0.4, 0.95, len(KN_VALUES)))

    ax.plot(x_nav, syy_nav / 1e6, color="gray", linestyle=":", linewidth=2.0, label="Navier (cuerpo rigido)")
    ax.plot(results[0]["x_base"], results[0]["syy_base"] / 1e6, color="black", linestyle="--",
            linewidth=2.0, marker="s", markersize=5, label="Empotrada")
    for r, color in zip(results[1:], blues):
        ax.plot(r["x_base"], r["syy_base"] / 1e6, color=color, linewidth=2.0, marker="o",
                markersize=5, label="Robin, %s N/m$^3$" % r["label"].replace("kn", "$k_n$"))

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("x [m] (talon A en x = 0, pie B en x = %.0f m)" % results[0]["x_base"][-1], fontsize=11)
    ax.set_ylabel("$\\sigma_{yy}$ en la base [MPa]", fontsize=11)
    ax.set_title("$\\sigma_{yy}$ en la base: empotrada vs. Robin ($k_t$ = %g $k_n$) - %s" % (KT_RATIO, element),
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(framealpha=0.9, fontsize=9)
    plt.tight_layout()

    save_path = os.path.join("results", "robin_base_syy_%s.png" % base_name)
    os.makedirs("results", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nFigura guardada en: %s" % save_path)
    print("Figuras de deformada y sigma_1 por kn en: results/*_%s_robin_kn*.png\n" % base_name)

    return results


if __name__ == "__main__":
    choice = sys.argv[1].upper() if len(sys.argv) > 1 else "Q8"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 2000.0

    if choice == "ALL":
        selected = list(ELEMENT_MESHES.keys())
    elif choice in ELEMENT_MESHES:
        selected = [choice]
    else:
        print("Elemento no reconocido: %s. Use CST, LST, Q4, Q8 o ALL" % choice)
        sys.exit(1)

    for element in selected:
        run_robin_study(element, scale_factor=scale)
