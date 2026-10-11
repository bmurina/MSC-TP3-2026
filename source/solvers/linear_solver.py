import numpy
import scipy.linalg


class LinearSolver:
    """Solver directo para sistemas algebraicos lineales (K u = f)."""

    @staticmethod
    def solve(K, f):
        """Resuelve el sistema K u = f usando descomposicion directa de SciPy/NumPy."""
        return scipy.linalg.solve(K, f)
