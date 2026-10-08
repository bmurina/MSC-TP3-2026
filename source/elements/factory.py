from source.elements.cst import CST
from source.elements.lst import LST
from source.elements.q4 import Q4
from source.elements.q8 import Q8


# (familia, orden) -> clase de elemento
ELEMENT_CLASSES = {
    ("TRI", 1): CST,
    ("TRI", 2): LST,
    ("QUAD", 1): Q4,
    ("QUAD", 2): Q8,
}


def build_elements(mesh):
    """Crea los elementos de la malla segun familia, orden y material."""
    elements = []

    for e in range(mesh.number_of_elements()):
        key = (mesh.element_types[e], mesh.element_orders[e])

        if key not in ELEMENT_CLASSES:
            raise ValueError("Unsupported element: %s order %d" % key)

        material_id = mesh.element_material_ids[e]

        element = ELEMENT_CLASSES[key](
            mesh.element_global_ids[e],
            mesh.connectivity[e],
            mesh,
            mesh.materials[material_id],
            mesh.thicknesses[material_id],
        )

        elements.append(element)

    return elements
