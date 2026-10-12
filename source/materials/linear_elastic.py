import numpy


PLANE_STRAIN = "PLANE_STRAIN"
PLANE_STRESS = "PLANE_STRESS"


class LinearElastic:
    """Material elastico lineal isotropo para problemas planos.

    plane_state indica la hipotesis en la direccion z (perpendicular al plano):
      - PLANE_STRAIN: eps_zz = 0, sigma_zz = nu (sigma_xx + sigma_yy) != 0.
        Seccion de presa confinada lateralmente (laderas impiden el
        desplazamiento en z).
      - PLANE_STRESS: sigma_zz = 0 (placas delgadas sin carga en z).
    """

    def __init__(self, E, nu, rho=0.0, plane_state=PLANE_STRAIN):
        if plane_state not in (PLANE_STRAIN, PLANE_STRESS):
            raise ValueError("Unknown plane state: " + plane_state)

        self.E = E          # modulo de elasticidad [Pa]
        self.nu = nu        # coeficiente de Poisson [-]
        self.rho = rho      # densidad [kg/m3] (usada para peso propio)
        self.plane_state = plane_state

    def constitutive_matrix(self):
        """Matriz D (3x3) segun la hipotesis plana del material.

        {sigma_xx, sigma_yy, tau_xy} = D @ {eps_xx, eps_yy, gamma_xy}
        """
        if self.plane_state == PLANE_STRAIN:
            return self.plane_strain_matrix()

        return self.plane_stress_matrix()

    def plane_strain_matrix(self):
        """Matriz constitutiva D (3x3) para estado plano de deformacion
        (eps_zz = gamma_xz = gamma_yz = 0).
        """
        c = self.E / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        return c * numpy.array([
            [1.0 - self.nu, self.nu, 0.0],
            [self.nu, 1.0 - self.nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * self.nu) / 2.0],
        ])

    def plane_stress_matrix(self):
        """Matriz constitutiva D (3x3) para estado plano de tension
        (sigma_zz = tau_xz = tau_yz = 0).
        """
        c = self.E / (1.0 - self.nu * self.nu)

        return c * numpy.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0],
        ])

    def out_of_plane_stress(self, sigma):
        """Tension normal al plano sigma_zz a partir de [sxx, syy, txy].

        En deformacion plana, de eps_zz = 0 en la ley de Hooke 3D:
            sigma_zz = nu (sigma_xx + sigma_yy)
        En tension plana es nula por hipotesis.
        """
        if self.plane_state == PLANE_STRAIN:
            return self.nu * (sigma[0] + sigma[1])

        return 0.0
