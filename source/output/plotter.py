import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# Orden perimetral de nodos locales para dibujar el poligono cerrado de cada elemento
PERIMETER_INDICES = {
    ("TRI", 1): [0, 1, 2],                      # CST (3 vertices)
    ("TRI", 2): [0, 3, 1, 4, 2, 5],             # LST (vertices 0,1,2; medios 3,4,5)
    ("QUAD", 1): [0, 1, 2, 3],                  # Q4 (4 vertices)
    ("QUAD", 2): [0, 4, 1, 5, 2, 6, 3, 7],      # Q8 (vertices 0,1,2,3; medios 4,5,6,7)
}


# ----------------------------------------------------------------------
# Auxiliares comunes
# ----------------------------------------------------------------------
def _nodal_coordinates(mesh, u, scale_factor):
    """Coordenadas nodales originales, deformadas amplificadas y |u| [m]."""
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

    x_def = x_orig + scale_factor * ux
    y_def = y_orig + scale_factor * uy
    u_mag = np.sqrt(ux**2 + uy**2)

    return x_orig, y_orig, x_def, y_def, u_mag


def _perimeter_indices(mesh):
    elem_type = mesh.element_types[0] if mesh.element_types else "TRI"
    elem_order = mesh.element_orders[0] if mesh.element_orders else 1
    return PERIMETER_INDICES.get((elem_type, elem_order), list(range(len(mesh.connectivity[0]))))


def _element_polygons(mesh, x, y):
    """Lista de poligonos (uno por elemento) con las coordenadas dadas."""
    perim_idx = _perimeter_indices(mesh)
    return [
        Polygon(np.array([[x[conn[i]], y[conn[i]]] for i in perim_idx]), closed=True)
        for conn in mesh.connectivity
    ]


def _draw_undeformed(ax, mesh, x_orig, y_orig):
    ax.add_collection(PatchCollection(
        _element_polygons(mesh, x_orig, y_orig),
        facecolor="lightgray",
        edgecolor="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.4,
    ))


def _draw_water(ax, water_height=28.0):
    """Representacion esquematica del agua aguas arriba (x < 0)."""
    ax.axhline(water_height, xmin=0.0, xmax=0.35, color="dodgerblue", linestyle="-.", linewidth=1.5)
    ax.fill_between([-2.5, 0], [0, 0], [water_height, water_height], color="deepskyblue", alpha=0.15)
    ax.annotate(
        "Nivel de agua\n$H_w = %g\\,$m" % water_height,
        xy=(0, water_height),
        xytext=(-3.5, water_height + 0.5),
        arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=1.2),
        fontsize=9,
        color="darkblue",
        fontweight="bold",
    )


def _finish_axes(fig, ax, x_orig, y_orig, x_def, y_def, title, scale_factor, show_undeformed, save_path, show,
                 extra_legend=None):
    """Limites, aspecto, titulo, leyenda, guardado y visualizacion."""
    all_x = np.concatenate([x_orig, x_def])
    all_y = np.concatenate([y_orig, y_def])
    margin = 2.0
    ax.set_xlim(min(all_x) - margin - 3.0, max(all_x) + margin)
    ax.set_ylim(min(0.0, min(all_y)) - margin, max(all_y) + margin)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)

    legend_elements = []
    if show_undeformed:
        legend_elements += [
            Line2D([0], [0], linestyle="--", color="gray", lw=1.2, label="Original (indeformada)"),
            Line2D([0], [0], linestyle="-", color="black", lw=1.2, label="Deformada (x%g)" % scale_factor),
        ]
    legend_elements += extra_legend or []
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Figura guardada en: %s" % save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------
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
    x_orig, y_orig, x_def, y_def, u_mag = _nodal_coordinates(mesh, u, scale_factor)
    u_mag_mm = u_mag * 1000.0

    fig, ax = plt.subplots(figsize=(9, 10))

    # 1. Malla indeformada (si se solicita)
    if show_undeformed:
        _draw_undeformed(ax, mesh, x_orig, y_orig)

    # 2. Malla deformada coloreada por el promedio de |u| en los nodos del elemento
    p_def = PatchCollection(
        _element_polygons(mesh, x_def, y_def),
        cmap=cmap,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    p_def.set_array(np.array([np.mean(u_mag_mm[conn]) for conn in mesh.connectivity]))
    ax.add_collection(p_def)

    cbar = fig.colorbar(p_def, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
    cbar.set_label("Magnitud de desplazamiento $|u|$ [mm]", fontsize=11, fontweight="bold")

    # 3. Agua
    _draw_water(ax)

    if title is None:
        title = "Configuracion Deformada - Presa de Gravedad\n(Factor de escala: %g$\\times$ | Disp. max: %.4f mm)" % (
            scale_factor, np.max(u_mag_mm)
        )

    _finish_axes(fig, ax, x_orig, y_orig, x_def, y_def, title, scale_factor, show_undeformed, save_path, show)

    return fig, ax


def plot_principal_stress(
    mesh,
    elements,
    u,
    scale_factor=5000.0,
    n_sub=4,
    vlim=None,
    show_undeformed=True,
    title=None,
    save_path=None,
    show=True,
    cmap="RdBu_r",
):
    """Grafica la tension principal maxima sigma_1 (3D, incluye sigma_zz)
    sobre la configuracion deformada.

    Cada elemento se subdivide en n_sub divisiones por lado en su dominio de
    referencia; sigma_1 se evalua en esos puntos con las funciones de forma
    del propio elemento (sin promediado nodal, por lo que se ven los saltos
    entre elementos) y se interpola linealmente en cada sub-triangulo.
    Las posiciones se mapean isoparametricamente, por lo que los lados de
    LST y Q8 se dibujan curvos.

    Parametros
    ----------
    mesh         : Mesh
    elements     : lista de Element2D (CST, LST, Q4 o Q8)
    u            : ndarray (n_dofs,), vector global de desplazamientos [m]
    scale_factor : float, factor de amplificacion de los desplazamientos
    n_sub        : int, divisiones por lado de cada elemento
    vlim         : float opcional [MPa]; satura la escala en +-vlim (util por
                   la singularidad del empotramiento en el talon)
    cmap         : colormap divergente centrado en sigma_1 = 0
                   (rojo = traccion, azul = compresion)
    """
    x_orig, y_orig, x_def, y_def, u_mag = _nodal_coordinates(mesh, u, scale_factor)

    # 1. Triangulacion global de puntos de muestreo
    xs, ys, values, triangles = [], [], [], []
    offset = 0

    for element in elements:
        points, local_tris = element.sample_grid(n_sub)

        for xi, eta in points:
            x, y = element.deformed_position(u, xi, eta, scale_factor)
            xs.append(x)
            ys.append(y)
            values.append(element.max_principal_stress(u, xi, eta) / 1e6)

        triangles.append(local_tris + offset)
        offset += len(points)

    xs = np.array(xs)
    ys = np.array(ys)
    values = np.array(values)
    triangulation = mtri.Triangulation(xs, ys, np.vstack(triangles))

    # 2. Escala de colores divergente centrada en 0
    s_max = float(values.max())
    s_min = float(values.min())
    if vlim is not None:
        vmin, vmax = -vlim, vlim
    else:
        vmin, vmax = min(s_min, -1e-9), max(s_max, 1e-9)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 10))

    if show_undeformed:
        _draw_undeformed(ax, mesh, x_orig, y_orig)

    field = ax.tripcolor(triangulation, np.clip(values, vmin, vmax), shading="gouraud", cmap=cmap, norm=norm)

    # 3. Limite de la zona traccionada. En los paramentos libres comprimidos
    # sigma_1 = 0 exactamente, por lo que la isolinea 0 queda ruidosa sobre
    # ellos; se dibuja el umbral de 1% del rango para marcar traccion efectiva.
    level = 0.01 * max(abs(s_min), abs(s_max))
    if s_max > level:
        ax.tricontour(triangulation, values, levels=[level], colors="black", linewidths=1.4)

    # 4. Contorno de los elementos deformados
    ax.add_collection(PatchCollection(
        _element_polygons(mesh, x_def, y_def),
        facecolor="none",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.6,
    ))

    # 5. Ubicacion del sigma_1 maximo
    i_max = int(np.argmax(values))
    ax.plot(xs[i_max], ys[i_max], marker="*", color="gold", markeredgecolor="black", markersize=14, zorder=5)
    ax.annotate(
        "$\\sigma_{1,max}$ = %.3f MPa" % s_max,
        xy=(xs[i_max], ys[i_max]),
        xytext=(xs[i_max] + 4.0, ys[i_max] + 3.0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    cbar = fig.colorbar(field, ax=ax, orientation="vertical", shrink=0.7, pad=0.03,
                        extend="both" if vlim is not None else "neither")
    # Ticks equiespaciados en cada mitad (la norma es asimetrica respecto de 0)
    cbar.set_ticks(np.concatenate([np.linspace(vmin, 0.0, 5), np.linspace(0.0, vmax, 5)[1:]]))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cbar.set_label("Tension principal maxima $\\sigma_1$ [MPa]", fontsize=11, fontweight="bold")

    _draw_water(ax)

    if title is None:
        title = "Tension principal maxima $\\sigma_1$ sobre la deformada\n(Escala: %g$\\times$ | $\\sigma_1$ en [%.3f, %.3f] MPa)" % (
            scale_factor, s_min, s_max
        )

    extra_legend = [Line2D([0], [0], color="black", lw=1.4, label="Limite zona traccionada ($\sigma_1$ = %.3f MPa)" % level)]
    _finish_axes(fig, ax, x_orig, y_orig, x_def, y_def, title, scale_factor, show_undeformed, save_path, show,
                 extra_legend=extra_legend if s_max > level else None)

    return fig, ax
