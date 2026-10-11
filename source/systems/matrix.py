import numpy


class DenseMatrix:
    """Matriz densa global para ensamblaje FEM.

    Permite acumular la rigidez elemental de cada elemento mediante
    scatter_local_to_global, compatible con la interfaz usual de Element2D.
    """

    def __init__(self, n_rows, n_cols=None):
        if n_cols is None:
            n_cols = n_rows
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.data = numpy.zeros((n_rows, n_cols), dtype=float)

    def scatter_local_to_global(self, local_matrix, dofs):
        """Suma la matriz local (k x k) en los grados de libertad globales especificados.

        Parametros
        ----------
        local_matrix : ndarray de forma (k, k)
        dofs         : lista de enteros con los indices globales de los DOFs
        """
        dofs = numpy.asarray(dofs, dtype=int)
        self.data[numpy.ix_(dofs, dofs)] += local_matrix

    def to_numpy(self):
        """Devuelve el arreglo numpy subyacente."""
        return self.data

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, value):
        self.data[item] = value

    @property
    def shape(self):
        return self.data.shape
