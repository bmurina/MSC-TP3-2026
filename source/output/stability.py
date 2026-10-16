"""Verificacion de estabilidad global de la presa (Paso 8).

Compara, tomando momentos respecto del pie aguas abajo (B, 0):

  - Solucion analitica: peso propio (M_est) y empuje hidrostatico (M_des).
  - Cargas nodales del modelo FEM (vector F): comprueba que la carga
    consistente reproduce los momentos analiticos.
  - Reacciones de vinculo en la base obtenidas del FEM (R = K u - F).

Convencion de signos: momentos positivos antihorarios (z saliente).
  - El peso (a la izquierda del pie) gira en sentido antihorario: estabilizante.
  - El empuje (+x a cota y > 0) gira en sentido horario: desestabilizante.
  - Equilibrio: M_peso + M_empuje + M_reacciones = 0.
"""

import numpy as np


def polygon_properties(vertices):
    """Area y centroide (xc, yc) de un poligono simple (formula de Gauss)."""
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * np.sum(cross)
    xc = np.sum((x + x1) * cross) / (6.0 * area)
    yc = np.sum((y + y1) * cross) / (6.0 * area)
    return abs(area), xc, yc


def dam_polygon(height=30.0, base=18.0, crest=3.0):
    """Perfil de la presa: cara aguas arriba vertical en x = 0, cresta de
    ancho `crest` y paramento aguas abajo inclinado hasta (base, 0)."""
    return [(0.0, 0.0), (base, 0.0), (crest, height), (0.0, height)]


def stability_report(
    system,
    height=30.0,
    base=18.0,
    crest=3.0,
    water_height=28.0,
    rho_concrete=2400.0,
    rho_water=1000.0,
    gravity=9.81,
    mu=0.7,
    y_base=0.0,
):
    """Calcula la tabla de estabilidad de un sistema ya resuelto.

    mu : coeficiente de friccion hormigon-roca para el deslizamiento.

    Devuelve un diccionario con las magnitudes analiticas, las obtenidas del
    FEM y los factores de seguridad.
    """
    # ---- Solucion analitica ------------------------------------------------
    area, xc, yc = polygon_properties(dam_polygon(height, base, crest))
    weight = rho_concrete * gravity * area                       # por metro de espesor
    arm_weight = base - xc                                       # brazo respecto del pie
    m_est = weight * arm_weight

    f_hydro = 0.5 * rho_water * gravity * water_height ** 2
    y_cp = water_height / 3.0
    m_des = f_hydro * y_cp

    # ---- Cargas nodales del FEM (vector F) --------------------------------
    nodes = system.mesh.nodes
    fx = system.F[0::2]
    fy = system.F[1::2]
    xs = np.array([n.x for n in nodes])
    ys = np.array([n.y for n in nodes])

    fem_weight = -np.sum(fy)
    fem_m_est = np.sum(-fy * (base - xs))
    fem_f_hydro = np.sum(fx)
    fem_m_des = np.sum(fx * (ys - y_base))

    # ---- Reacciones de vinculo del FEM ------------------------------------
    reac = system.get_base_reactions(y_val=y_base)
    rx = np.array([r["Rx"] for r in reac["nodes"]])
    ry = np.array([r["Ry"] for r in reac["nodes"]])
    rxs = np.array([r["x"] for r in reac["nodes"]])
    rys = np.array([r["y"] for r in reac["nodes"]])

    sum_rx = float(np.sum(rx))
    sum_ry = float(np.sum(ry))
    m_react = float(np.sum((rxs - base) * ry - (rys - y_base) * rx))   # respecto del pie

    # Momento de las reacciones que equilibra las cargas (valor positivo)
    m_react_needed = m_est - m_des
    x_resultant = float(np.sum(rxs * ry) / sum_ry)
    eccentricity = x_resultant - 0.5 * base       # respecto del centro de la base
    in_kern = abs(eccentricity) <= base / 6.0

    return {
        # analitico
        "area": area, "xc": xc, "yc": yc,
        "weight": weight, "arm_weight": arm_weight, "m_est": m_est,
        "f_hydro": f_hydro, "y_cp": y_cp, "m_des": m_des,
        "fs_overturning": m_est / m_des,
        "fs_sliding": mu * weight / f_hydro,
        # cargas FEM
        "fem_weight": fem_weight, "fem_m_est": fem_m_est,
        "fem_f_hydro": fem_f_hydro, "fem_m_des": fem_m_des,
        "fem_fs_overturning": fem_m_est / fem_m_des,
        # reacciones FEM
        "sum_rx": sum_rx, "sum_ry": sum_ry,
        "m_react": m_react, "m_react_needed": m_react_needed,
        "x_resultant": x_resultant, "eccentricity": eccentricity, "in_kern": in_kern,
        "fem_fs_sliding": mu * sum_ry / abs(sum_rx),
        "mu": mu, "base": base,
        # errores relativos
        "err_weight": (fem_weight - weight) / weight,
        "err_m_est": (fem_m_est - m_est) / m_est,
        "err_f_hydro": (fem_f_hydro - f_hydro) / f_hydro,
        "err_m_des": (fem_m_des - m_des) / m_des,
        "err_m_react": (m_react + m_react_needed) / m_react_needed,
        "moment_residual": m_est - m_des + m_react,
    }


def print_stability_table(rep, label=""):
    """Imprime la tabla formal de estabilidad."""
    line = "=" * 92
    print(line)
    print(" TABLA DE ESTABILIDAD (momentos respecto del pie aguas abajo, x = %.1f m)  %s" % (rep["base"], label))
    print(line)
    print("%-34s %16s %16s %12s" % ("Magnitud", "Analitico", "FEM", "Err. rel."))
    print("-" * 92)
    print("%-34s %16.1f %16.1f %11.2e" % ("Peso propio W [N/m]", rep["weight"], rep["fem_weight"], rep["err_weight"]))
    print("%-34s %16.4f %16s %12s" % ("Brazo del peso (B - xc) [m]", rep["arm_weight"], "-", "-"))
    print("%-34s %16.1f %16.1f %11.2e" % ("Momento estabilizante M_est [N m/m]", rep["m_est"], rep["fem_m_est"], rep["err_m_est"]))
    print("%-34s %16.1f %16.1f %11.2e" % ("Empuje hidrostatico Fh [N/m]", rep["f_hydro"], rep["fem_f_hydro"], rep["err_f_hydro"]))
    print("%-34s %16.1f %16.1f %11.2e" % ("Momento desestabilizante M_des [N m/m]", rep["m_des"], rep["fem_m_des"], rep["err_m_des"]))
    print("-" * 92)
    print("%-34s %16.1f %16.1f %11.2e" % ("Momento de reacciones |M_R| [N m/m]", rep["m_react_needed"], -rep["m_react"], rep["err_m_react"]))
    print("%-34s %16s %16.1f" % ("Reaccion horizontal Rx [N/m]", "-", rep["sum_rx"]))
    print("%-34s %16s %16.1f" % ("Reaccion vertical Ry [N/m]", "-", rep["sum_ry"]))
    print("%-34s %16s %16.4f" % ("Punto de paso de la resultante x_R [m]", "-", rep["x_resultant"]))
    print("%-34s %16s %16.4f" % ("Excentricidad e = x_R - B/2 [m]", "-", rep["eccentricity"]))
    print("%-34s %16s %16s" % ("Resultante en tercio medio (|e|<=B/6)", "-", "SI" if rep["in_kern"] else "NO"))
    print("%-34s %16s %16.3e" % ("Residuo de momentos [N m/m]", "-", rep["moment_residual"]))
    print("-" * 92)
    print("%-34s %16.3f %16.3f" % ("FS vuelco = M_est / M_des", rep["fs_overturning"], rep["fem_fs_overturning"]))
    print("%-34s %16.3f %16.3f" % ("FS deslizamiento = mu W / Fh (mu=%.2f)" % rep["mu"], rep["fs_sliding"], rep["fem_fs_sliding"]))
    print(line)
