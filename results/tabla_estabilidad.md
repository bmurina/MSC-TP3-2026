# Estabilidad de la presa (base empotrada, mu = 0.70)

Momentos respecto del pie aguas abajo (x = 18 m, y = 0), por metro de espesor.

## Solucion analitica

| Magnitud | Valor |
|---|---|
| Area del perfil | 315.00 m2 |
| Peso propio W | 7416360.0 N/m |
| Centroide xc | 6.1429 m (brazo B - xc = 11.8571 m) |
| M_est = W (B - xc) | 87936840.0 N m/m |
| Empuje Fh = rho_w g Hw^2 / 2 | 3845520.0 N/m |
| M_des = Fh Hw / 3 | 35891520.0 N m/m |
| FS vuelco = M_est / M_des | 2.450 |
| FS deslizamiento = mu W / Fh | 1.350 |

## Comparacion con el FEM

| Elemento | M_est carga FEM [N m/m] | M_des carga FEM [N m/m] | M_R reacciones [N m/m] | Err. M_R | Rx [N/m] | Ry [N/m] | x_R [m] | FS vuelco | FS desl. |
|---|---|---|---|---|---|---|---|---|---|
| CST | 87936839.5 | 35891520.0 | 52045319.5 | 9.1e-09 | -3845520.0 | 7416359.9 | 10.9824 | 2.450 | 1.350 |
| LST | 87936839.5 | 35891520.0 | 52045319.5 | 9.4e-09 | -3845520.0 | 7416359.9 | 10.9824 | 2.450 | 1.350 |
| Q4 | 87936839.5 | 35891520.0 | 52045319.5 | 9.1e-09 | -3845520.0 | 7416359.9 | 10.9824 | 2.450 | 1.350 |
| Q8 | 87936839.5 | 35891520.0 | 52045319.5 | 9.4e-09 | -3845520.0 | 7416359.9 | 10.9824 | 2.450 | 1.350 |
