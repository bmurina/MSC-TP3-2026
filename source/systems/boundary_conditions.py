import numpy


class BoundaryConditions:
    """Gestiona condiciones de borde esenciales (Dirichlet) para el sistema FEM."""

    def __init__(self, number_of_dofs):
        self.number_of_dofs = number_of_dofs
        self.prescribed_dofs = {}  # dof_index -> valor prescripto

    def fix_dof(self, dof_index, value=0.0):
        """Fija un grado de libertad global a un valor determinado."""
        if dof_index < 0 or dof_index >= self.number_of_dofs:
            raise IndexError("DOF index %d out of range (total DOFs: %d)" % (dof_index, self.number_of_dofs))
        self.prescribed_dofs[dof_index] = float(value)

    def fix_node(self, mesh, node_id, dof_indices=(0, 1), values=(0.0, 0.0)):
        """Fija los grados de libertad indicados de un nodo.

        dof_indices: 0 para u_x, 1 para u_y (o tupla con ambos).
        """
        node_dofs = mesh.node_global_dofs(node_id)
        if isinstance(dof_indices, int):
            dof_indices = (dof_indices,)
            values = (values,)
        for local_dof, val in zip(dof_indices, values):
            self.fix_dof(node_dofs[local_dof], val)

    def fix_base(self, mesh, y_val=0.0, tol=1e-5):
        """Fija ux = 0 y uy = 0 en todos los nodos ubicados en la base (y = y_val)."""
        fixed_nodes = []
        for node in mesh.nodes:
            if abs(node.y - y_val) <= tol:
                self.fix_node(mesh, node.global_id, dof_indices=(0, 1), values=(0.0, 0.0))
                fixed_nodes.append(node.global_id)
        return fixed_nodes

    def get_fixed_dofs(self):
        """Devuelve una lista ordenada de los DOFs restringidos."""
        return sorted(self.prescribed_dofs.keys())

    def get_free_dofs(self):
        """Devuelve una lista ordenada de los DOFs libres (no restringidos)."""
        fixed_set = set(self.prescribed_dofs.keys())
        return [i for i in range(self.number_of_dofs) if i not in fixed_set]

    def get_prescribed_values(self):
        """Devuelve un arreglo numpy con los valores prescriptos para los fixed_dofs."""
        fixed = self.get_fixed_dofs()
        return numpy.array([self.prescribed_dofs[dof] for dof in fixed], dtype=float)
