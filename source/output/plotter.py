import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# Orden perimetral de nodos locales para dibujar el poligono cerrado de cada elemento
PERIMETER_INDICES = {
    ("TRI", 1): [0, 1, 2],                      # CST (3 vertices)
    ("TRI", 2): [0, 3, 1, 4, 2, 5],             # LST (vertices 0,1,2; medios 3,4,5)
    ("QUAD", 1): [0, 1, 2, 3],                  # Q4 (4 vertices)
    ("QUAD", 2): [0, 4, 1, 5, 2, 6, 3, 7],      # Q8 (vertices 0,1,2,3; medios 4,5,6,7)
}


def plot_deformed_mesh(
    mesh,
    u,
    scale_factor=5000.0,
    show_undeformed=True,
    title=None,
    save_path=None,
    show=True,
    cmap="viridis",
):
    """Grafica la configuracion deformada de la estructura comparada con la indeformada.

    Parametros
    ----------
    mesh            : Mesh, objeto de malla
    u               : ndarray (n_dofs,), vector global de desplazamientos [m]
    scale_factor    : float, factor de amplificacion de los desplazamientos
    show_undeformed : bool, si True grafica el contorno original en linea punteada
    title           : str opcional, titulo del grafico
    save_path       : str opcional, ruta para guardar la figura en disco (.png)
    show            : bool, si True ejecuta plt.show()
    """
    n_nodes = mesh.number_of_nodes()
    x_orig = np.zeros(n_nodes)
    y_orig = np.zeros(n_nodes)
    ux = np.zeros(n_nodes)
    uy = np.zeros(n_nodes)

    for node in mesh.nodes:
        nid = node.global_id
        x_orig[nid] = node.x
        y_orig[nid] = node.y
        dofs = mesh.node_global_dofs(nid)
        ux[nid] = u[dofs[0]]
        uy[nid] = u[dofs[1]]

    # Coordenadas deformadas amplificadas
    x_def = x_orig + scale_factor * ux
    y_def = y_orig + scale_factor * uy

    # Magnitud de desplazamiento nodales en mm para mapa de color
    u_mag_mm = np.sqrt(ux**2 + uy**2) * 1000.0

    fig, ax = plt.subplots(figsize=(9, 10))

    elem_type = mesh.element_types[0] if mesh.element_types else "TRI"
    elem_order = mesh.element_orders[0] if mesh.element_orders else 1
    perim_idx = PERIMETER_INDICES.get((elem_type, elem_order), list(range(len(mesh.connectivity[0]))))

    # 1. Malla indeformada (si se solicita)
    if show_undeformed:
        undeformed_patches = []
        for conn in mesh.connectivity:
            pts = np.array([[x_orig[conn[i]], y_orig[conn[i]]] for i in perim_idx])
            poly = Polygon(pts, closed=True)
            undeformed_patches.append(poly)

        p_undef = PatchCollection(
            undeformed_patches,
            facecolor="lightgray",
            edgecolor="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.4,
            label="Original (indeformada)",
        )
        ax.add_collection(p_undef)

    # 2. Malla deformada coloreada por magnitud de desplazamiento
    deformed_patches = []
    elem_colors = []
    for conn in mesh.connectivity:
        pts = np.array([[x_def[conn[i]], y_def[conn[i]]] for i in perim_idx])
        poly = Polygon(pts, closed=True)
        deformed_patches.append(poly)
        # Promedio del desplazamiento en los nodos del elemento
        avg_disp = np.mean([u_mag_mm[nid] for nid in conn])
        elem_colors.append(avg_disp)

    p_def = PatchCollection(
        deformed_patches,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    p_def.set_array(np.array(elem_colors))
    ax.add_collection(p_def)

    cbar = fig.colorbar(p_def, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
    cbar.set_label("Magnitud de desplazamiento $|u|$ [mm]", fontsize=11, fontweight="bold")

    # 3. Representacion esquematica del agua (pelo de agua en y = 28m)
    ax.axhline(28.0, xmin=0.0, xmax=0.35, color="dodgerblue", linestyle="-.", linewidth=1.5)
    ax.fill_between([-2.5, 0], [0, 0], [28, 28], color="deepskyblue", alpha=0.15)
    ax.annotate(
        "Nivel de agua\n$H_w = 28\\,$m",
        xy=(0, 28),
        xytext=(-3.5, 28.5),
        arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=1.2),
        fontsize=9,
        color="darkblue",
        fontweight="bold",
    )

    # 4. Ajuste de limites y aspecto
    all_x = np.concatenate([x_orig, x_def])
    all_y = np.concatenate([y_orig, y_def])
    margin_x = 2.0
    margin_y = 2.0
    ax.set_xlim(min(all_x) - margin_x - 3.0, max(all_x) + margin_x)
    ax.set_ylim(-margin_y, max(all_y) + margin_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)

    if title is None:
        title = "Configuracion Deformada - Presa de Gravedad\n(Factor de escala: %g$\\times$ | Disp. max: %.4f mm)" % (
            scale_factor, np.max(u_mag_mm)
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)

    if show_undeformed:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], linestyle="--", color="gray", lw=1.2, label="Original (indeformada)"),
            Line2D([0], [0], linestyle="-", color="black", lw=1.2, label="Deformada (x%g)" % scale_factor),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Figura guardada en: %s" % save_path)

    if show:
        plt.show()

    return fig, ax
