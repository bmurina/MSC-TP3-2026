import numpy


class LinearElastic:
    """Material elastico lineal isotropo."""

    def __init__(self, E, nu, rho=0.0):
        self.E = E          # modulo de elasticidad [Pa]
        self.nu = nu        # coeficiente de Poisson [-]
        self.rho = rho      # densidad [kg/m3] (usada para peso propio)

    def plane_stress_matrix(self):
        """Matriz constitutiva D (3x3) para estado plano de tension.

        {sigma_xx, sigma_yy, tau_xy} = D @ {eps_xx, eps_yy, gamma_xy}
        """
        c = self.E / (1.0 - self.nu * self.nu)

        return c * numpy.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0],
        ])
