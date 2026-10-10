"""
dat_to_mesh.py
--------------
Convierte archivos .dat exportados por Simcenter/NX (formato BDF/Nastran)
al formato .mesh utilizado por el solver FEM del proyecto.

Uso:
    python dat_to_mesh.py

Mapeo de tipos de elemento Nastran -> familia/orden del proyecto:
    CPLSTN3 / CTRIA3   -> TRI  1  (CST, 3 nodos)
    CPLSTN6 / CTRIA6   -> TRI  2  (LST, 6 nodos)
    CPLSTN4 / CQUAD4   -> QUAD 1  (Q4,  4 nodos)
    CPLSTN8 / CQUAD8   -> QUAD 2  (Q8,  8 nodos)

Notas sobre unidades (archivos NX exportados en mm/milli-newton):
    Coordenadas: mm -> m   (/ 1000)
    E:  N/mm^2 (MPa) -> Pa  (* 1e6)
    rho: kg/mm^3 -> kg/m^3  (* 1e9)
    Espesor (PPLANE): mm -> m  (/ 1000)
"""

import os
import sys

from pyNastran.bdf.bdf import BDF

# ---------------------------------------------------------------------------
# Mapeo tipo Nastran -> (familia, orden)
# ---------------------------------------------------------------------------
ELEMENT_MAP = {
    "CTRIA3":  ("TRI",  1),
    "CPLSTN3": ("TRI",  1),
    "CTRIA6":  ("TRI",  2),
    "CPLSTN6": ("TRI",  2),
    "CQUAD4":  ("QUAD", 1),
    "CPLSTN4": ("QUAD", 1),
    "CQUAD8":  ("QUAD", 2),
    "CPLSTN8": ("QUAD", 2),
}


def fix_begin_bulk(bdf_file):
    """
    Comenta la linea 'BEGIN BULK' si no empieza con '$'.
    Necesario para que pyNastran pueda leer el archivo en modo punch=True.
    (Misma logica que el script original de la catedra.)
    """
    with open(bdf_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if not line.startswith("$"):
            if line.strip().upper() == "BEGIN BULK":
                lines[i] = "$" + line
            break

    with open(bdf_file, "w") as f:
        f.writelines(lines)


def read_pplane_thickness(bdf_file):
    """
    Extrae el espesor de la tarjeta PPLANE directamente del texto del .dat.
    pyNastran 1.4.x no soporta completamente PPLANE, por eso lo leemos a mano.
    Retorna el espesor en mm (sin conversion).
    """
    thickness_mm = 1000.0  # default: 1000 mm = 1 m
    with open(bdf_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("PPLANE"):
                # Formato: PPLANE  pid  mid  t  nsm
                parts = stripped.split()
                if len(parts) >= 4:
                    try:
                        thickness_mm = float(parts[3])
                    except ValueError:
                        pass
                break
    return thickness_mm


def convert_dat_to_mesh(dat_file, mesh_file, dim=2):
    """
    Convierte un archivo .dat (BDF/Nastran) a formato .mesh del proyecto.

    Parametros
    ----------
    dat_file  : str  - ruta al .dat de entrada
    mesh_file : str  - ruta al .mesh de salida
    dim       : int  - dimension del problema (2 para 2D)
    """
    print("Convirtiendo: %s" % dat_file)
    print("          ->  %s" % mesh_file)

    # 1. Leer el BDF
    fix_begin_bulk(dat_file)
    model = BDF(debug=False)
    model.read_bdf(dat_file, punch=True, xref=True, validate=False)

    # 2. Nodos ordenados: reindexar de 1 a N
    nids_sorted = sorted(model.nodes.keys())
    n_nodes = len(nids_sorted)
    nid_to_idx = {nid: (i + 1) for i, nid in enumerate(nids_sorted)}

    # 3. Filtrar elementos compatibles
    valid_elements = []
    for eid, elem in sorted(model.elements.items()):
        etype = elem.type
        if etype in ELEMENT_MAP:
            valid_elements.append((eid, elem))
        else:
            print("  [AVISO] Elemento '%s' (eid=%d) no soportado, omitido." % (etype, eid))

    n_elements = len(valid_elements)
    if n_elements == 0:
        raise RuntimeError("No se encontraron elementos compatibles en el archivo.")

    # 4. Material (tomamos el primero disponible)
    if not model.materials:
        raise RuntimeError("No se encontraron materiales en el archivo.")

    mid = sorted(model.materials.keys())[0]
    mat = model.materials[mid]

    # Conversion de unidades (sistema mm / milli-newton del .dat):
    #   E  : mN/mm^2  ->  Pa  : * (1e-3 N/mN) * (1e6 mm^2/m^2) = * 1e3
    #   rho: kg/mm^3  ->  kg/m^3 : * 1e9
    E_Pa = mat.e * 1e3
    nu = mat.nu
    if mat.rho is not None and mat.rho != 0.0:
        rho_kgm3 = mat.rho * 1e9
    else:
        rho_kgm3 = 2400.0  # Hormigon H21 por defecto

    # Espesor: PPLANE en mm -> m
    thickness_mm = read_pplane_thickness(dat_file)
    thickness_m = thickness_mm / 1000.0

    # 5. Escribir el .mesh
    with open(mesh_file, "w") as f:
        f.write("# Generado automaticamente desde: %s\n" % os.path.basename(dat_file))
        f.write("# n_nodes n_elements n_materials dim\n")
        f.write("%d %d 1 %d\n" % (n_nodes, n_elements, dim))
        f.write("\n")
        f.write("# node_id x[m] y[m]\n")
        for i, nid in enumerate(nids_sorted):
            node = model.nodes[nid]
            x_m = node.xyz[0] / 1000.0
            y_m = node.xyz[1] / 1000.0
            f.write("%d %.10f %.10f\n" % (i + 1, x_m, y_m))
        f.write("\n")
        f.write("# material_id type E[Pa] nu rho[kg/m3] thickness[m]\n")
        f.write("1 LINEAR_ELASTIC %.6e %.6f %.4f %.6f\n" % (E_Pa, nu, rho_kgm3, thickness_m))
        f.write("\n")
        f.write("# element_id family order material_id connectivity(1-indexed)\n")
        for e_idx, (eid, elem) in enumerate(valid_elements):
            family, order = ELEMENT_MAP[elem.type]
            nodes_proj = [nid_to_idx[n] for n in elem.nodes]
            conn_str = " ".join(str(n) for n in nodes_proj)
            f.write("%d %s %d 1 %s\n" % (e_idx + 1, family, order, conn_str))

    print("  -> Nodos: %d, Elementos: %d" % (n_nodes, n_elements))
    print("  -> Material: E=%.4e Pa, nu=%.3f, rho=%.1f kg/m3, t=%.3f m" % (
        E_Pa, nu, rho_kgm3, thickness_m))
    print("  -> OK\n")


# ---------------------------------------------------------------------------
# Lista de conversiones a realizar
# ---------------------------------------------------------------------------
CONVERSIONS = [
    (
        r"C:\Users\Juan\Desktop\Finitos\mallas_planestrain\trianglineal_strain.dat",
        r"meshes\presa_cst_strain.mesh",
    ),
    (
        r"C:\Users\Juan\Desktop\Finitos\mallas_planestrain\triangcuad_strain.dat",
        r"meshes\presa_lst_strain.mesh",
    ),
    (
        r"C:\Users\Juan\Desktop\Finitos\mallas_planestrain\Q4_strain.dat",
        r"meshes\presa_q4_strain.mesh",
    ),
    (
        r"C:\Users\Juan\Desktop\Finitos\mallas_planestrain\Q8_strain.dat",
        r"meshes\presa_q8_strain.mesh",
    ),
]


if __name__ == "__main__":
    print("=" * 60)
    print("  Conversor .dat (Nastran/NX) -> .mesh (solver FEM)")
    print("=" * 60)
    print()

    errors = 0
    for dat_file, mesh_file in CONVERSIONS:
        if not os.path.isfile(dat_file):
            print("[ERROR] No se encontro el archivo: %s\n" % dat_file)
            errors += 1
            continue
        try:
            convert_dat_to_mesh(dat_file, mesh_file)
        except Exception as exc:
            print("  [ERROR] Fallo la conversion: %s\n" % exc)
            errors += 1

    if errors == 0:
        print("=" * 60)
        print("  Todos los archivos convertidos exitosamente.")
        print("=" * 60)
    else:
        print("Conversion completada con %d error(es)." % errors)
        sys.exit(1)
