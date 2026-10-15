import numpy


PLANE_STRAIN = "PLANE_STRAIN"


class LinearElastic:
    """Material elastico lineal isotropo en estado plano de deformacion.

    Hipotesis en la direccion z (perpendicular al plano): eps_zz = 0, por lo
    que sigma_zz = nu (sigma_xx + sigma_yy) != 0. Corresponde a la seccion de
    presa confinada lateralmente (las laderas impiden el desplazamiento en z).
    """

    def __init__(self, E, nu, rho=0.0):
        self.E = E          # modulo de elasticidad [Pa]
        self.nu = nu        # coeficiente de Poisson [-]
        self.rho = rho      # densidad [kg/m3] (usada para peso propio)

    def constitutive_matrix(self):
        """Matriz constitutiva D (3x3) para estado plano de deformacion
        (eps_zz = gamma_xz = gamma_yz = 0).

        {sigma_xx, sigma_yy, tau_xy} = D @ {eps_xx, eps_yy, gamma_xy}
        """
        c = self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        return c * numpy.array([
            [1.0 - self.nu, self.nu, 0.0],
            [self.nu, 1.0 - self.nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * self.nu) / 2.0],
        ])

    def out_of_plane_stress(self, sigma):
        """Tension normal al plano sigma_zz a partir de [sxx, syy, txy].

        De eps_zz = 0 en la ley de Hooke 3D:
            sigma_zz = nu (sigma_xx + sigma_yy)
        """
        return self.nu * (sigma[0] + sigma[1])
