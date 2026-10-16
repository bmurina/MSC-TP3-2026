import os

import numpy

# Tipos de celda de VTK segun (familia, orden). El orden de nodos del
# .mesh (vertices primero y luego nodos medios) coincide con el de VTK.
VTK_CELL_TYPES = {
    ("TRI", 1): 5,     # VTK_TRIANGLE
    ("TRI", 2): 22,    # VTK_QUADRATIC_TRIANGLE
    ("QUAD", 1): 9,    # VTK_QUAD
    ("QUAD", 2): 23,   # VTK_QUADRATIC_QUAD
}


def nodal_stresses(mesh, elements, u):
    """Tensiones [sxx, syy, txy, szz] por nodo, promediando la contribucion
    de todos los elementos que comparten cada nodo. Devuelve (n_nodos, 4)."""
    n_nodes = mesh.number_of_nodes()
    total = numpy.zeros((n_nodes, 4))
    count = numpy.zeros(n_nodes)

    for element in elements:
        sigma_e = element.stress_at_nodes(u)
        for local, node_id in enumerate(element.node_global_ids):
            total[node_id] += sigma_e[local]
            count[node_id] += 1.0

    return total / count[:, None]


def von_mises_plane_strain(sigma):
    """Von Mises 3D de filas [sxx, syy, txy, szz] (tau_xz = tau_yz = 0)."""
    sxx, syy, txy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2], sigma[:, 3]
    return numpy.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * txy ** 2
    )


def _principal_3d(elements, sigma):
    """sigma_1, sigma_2, sigma_3 de filas [sxx, syy, txy, szz]."""
    principal = numpy.array([elements[0].principal_stresses_3d(s) for s in sigma])
    return principal[:, 0], principal[:, 1], principal[:, 2]


def _write_array(file, name, data, n_components=1):
    file.write(
        '        <DataArray type="Float64" Name="%s" NumberOfComponents="%d" format="ascii">\n'
        % (name, n_components)
    )
    flat = numpy.asarray(data, dtype=float).reshape(-1, n_components)
    for row in flat:
        file.write("          " + " ".join("%.10e" % v for v in row) + "\n")
    file.write("        </DataArray>\n")


def write_vtu(filename, mesh, elements, u, scale_factor=0.0):
    """Exporta la solucion a un archivo VTU (VTK XML, ASCII) para ParaView.

    Datos nodales (PointData):
      Desplazamiento [m] (vector), Desplazamiento_mag [m],
      Tension_xx/yy/xy/zz [Pa], Sigma_1/2/3 [Pa], VonMises [Pa].
    Datos por elemento (CellData): promedio de Gauss de las mismas tensiones
      (sin suavizar entre elementos).

    scale_factor = 0: los puntos quedan en la configuracion original (en
    ParaView usar el filtro 'Warp By Vector' con Desplazamiento).
    scale_factor > 0: los puntos se escriben en la configuracion deformada
    amplificada x + scale_factor * u.
    """
    n_nodes = mesh.number_of_nodes()
    n_cells = len(elements)

    coords = numpy.array([[node.x, node.y] for node in mesh.nodes])
    u_nodes = numpy.zeros((n_nodes, 3))
    u_nodes[:, 0] = u[0::2]
    u_nodes[:, 1] = u[1::2]

    points = numpy.zeros((n_nodes, 3))
    points[:, :2] = coords + scale_factor * u_nodes[:, :2]

    sigma_n = nodal_stresses(mesh, elements, u)
    s1_n, s2_n, s3_n = _principal_3d(elements, sigma_n)
    vm_n = von_mises_plane_strain(sigma_n)

    sigma_c = numpy.array([element.stress_at_centroid(u) for element in elements])
    s1_c, s2_c, s3_c = _principal_3d(elements, sigma_c)
    vm_c = von_mises_plane_strain(sigma_c)

    connectivity = []
    offsets = []
    types = []
    offset = 0
    for element in elements:
        key = (mesh.element_types[element.global_id], mesh.element_orders[element.global_id])
        if key not in VTK_CELL_TYPES:
            raise ValueError("Elemento sin tipo VTK equivalente: %s orden %d" % key)
        connectivity.extend(element.node_global_ids)
        offset += len(element.node_global_ids)
        offsets.append(offset)
        types.append(VTK_CELL_TYPES[key])

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write('    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (n_nodes, n_cells))

        f.write("      <Points>\n")
        _write_array(f, "Points", points, 3)
        f.write("      </Points>\n")

        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(str(c) for c in connectivity) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(str(o) for o in offsets) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(str(t) for t in types) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")

        f.write('      <PointData Vectors="Desplazamiento" Scalars="Sigma_1">\n')
        _write_array(f, "Desplazamiento", u_nodes, 3)
        _write_array(f, "Desplazamiento_mag", numpy.linalg.norm(u_nodes, axis=1))
        _write_array(f, "Tension_xx", sigma_n[:, 0])
        _write_array(f, "Tension_yy", sigma_n[:, 1])
        _write_array(f, "Tension_xy", sigma_n[:, 2])
        _write_array(f, "Tension_zz", sigma_n[:, 3])
        _write_array(f, "Sigma_1", s1_n)
        _write_array(f, "Sigma_2", s2_n)
        _write_array(f, "Sigma_3", s3_n)
        _write_array(f, "VonMises", vm_n)
        f.write("      </PointData>\n")

        f.write('      <CellData Scalars="Sigma_1_elem">\n')
        _write_array(f, "Tension_xx_elem", sigma_c[:, 0])
        _write_array(f, "Tension_yy_elem", sigma_c[:, 1])
        _write_array(f, "Tension_xy_elem", sigma_c[:, 2])
        _write_array(f, "Tension_zz_elem", sigma_c[:, 3])
        _write_array(f, "Sigma_1_elem", s1_c)
        _write_array(f, "Sigma_2_elem", s2_c)
        _write_array(f, "Sigma_3_elem", s3_c)
        _write_array(f, "VonMises_elem", vm_c)
        f.write("      </CellData>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")

    return filename
