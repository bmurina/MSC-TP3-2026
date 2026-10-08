from source.mesh.node import Node
from source.materials.linear_elastic import LinearElastic


class Mesh:
    """Malla leida desde un archivo de texto plano .mesh.

    Formato (las lineas vacias y todo lo que sigue a '#' se ignoran):

        n_nodes n_elements n_materials dim
        id x y                                    (n_nodes lineas)
        id LINEAR_ELASTIC E nu rho thickness      (n_materials lineas)
        id family order material_id n1 n2 ...     (n_elements lineas)

    family: TRI o QUAD.  order: 1 (lineal) o 2 (cuadratico).
    Todos los ids del archivo empiezan en 1; internamente se usan desde 0.
    """

    def __init__(self):
        self.dim = 0
        self.dofs_per_node = 2
        self.nodes = []
        self.materials = {}         # material_id -> LinearElastic
        self.thicknesses = {}       # material_id -> espesor [m]
        self.element_global_ids = []
        self.element_types = []     # "TRI" o "QUAD"
        self.element_orders = []    # 1 o 2
        self.element_material_ids = []
        self.connectivity = []

    def read(self, filename):
        file = open(filename, "r")

        lines = []

        for raw_line in file:
            line = raw_line.split("#")[0]
            line = line.strip()

            if line != "":
                lines.append(line)

        file.close()

        header = lines[0].split()

        n_nodes = int(header[0])
        n_elements = int(header[1])
        n_materials = int(header[2])
        self.dim = int(header[3])

        line_number = 1

        # Nodos: id x y
        for i in range(n_nodes):
            values = lines[line_number].split()
            line_number = line_number + 1

            global_id = int(values[0]) - 1

            if global_id != i:
                raise ValueError("Unexpected node numbering")

            self.nodes.append(Node(global_id, float(values[1]), float(values[2])))

        # Materiales: id tipo E nu rho espesor
        for i in range(n_materials):
            values = lines[line_number].split()
            line_number = line_number + 1

            material_id = int(values[0])

            if values[1] != "LINEAR_ELASTIC":
                raise ValueError("Unknown material type: " + values[1])

            self.materials[material_id] = LinearElastic(
                float(values[2]), float(values[3]), float(values[4])
            )
            self.thicknesses[material_id] = float(values[5])

        # Elementos: id family order material_id conectividad...
        for i in range(n_elements):
            values = lines[line_number].split()
            line_number = line_number + 1

            element_conn = []

            for j in range(4, len(values)):
                node_id = int(values[j]) - 1

                if node_id < 0 or node_id >= n_nodes:
                    raise ValueError("Node out of range in element " + values[0])

                element_conn.append(node_id)

            material_id = int(values[3])

            if material_id not in self.materials:
                raise ValueError("Unknown material in element " + values[0])

            self.element_global_ids.append(int(values[0]) - 1)
            self.element_types.append(values[1])
            self.element_orders.append(int(values[2]))
            self.element_material_ids.append(material_id)
            self.connectivity.append(element_conn)

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_elements(self):
        return len(self.connectivity)

    def node(self, node_global_id):
        return self.nodes[node_global_id]

    def number_of_dofs(self):
        return self.number_of_nodes() * self.dofs_per_node

    def node_global_dofs(self, node_global_id):
        dofs = []

        for i in range(self.dofs_per_node):
            dofs.append(node_global_id * self.dofs_per_node + i)

        return dofs
