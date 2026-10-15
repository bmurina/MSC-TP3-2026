# MSC - Trabajo Práctico N°3: Estado Plano de Deformación

**Mecánica de Sólidos Computacional (30.53) - ITBA**

Solver bidimensional por Elementos Finitos (FEM) en **Python** para análisis elástico lineal en estado plano de deformación ($\varepsilon_{zz}=0$, $\sigma_{zz}=\nu(\sigma_{xx}+\sigma_{yy})$). Aplicación al análisis estructural y de estabilidad de una presa de gravedad de hormigón.

---

## Integrantes
- Bruno Murina (`bmurina@itba.edu.ar`)
- Juan Manuel Rodriguez Spagnol (`[jrodriguezspagnol@itba.edu.ar]`)

---

## Objetivos del Trabajo Práctico
1. Implementación de una arquitectura FEM modular en Python.
2. Formulación y validación de elementos 2D continuos:
   - Triangulares: **CST** (lineal, 3 nodos) y **LST** (cuadrático, 6 nodos).
   - Cuadrangulares: **Q4** (bilineal isoparamétrico, Gauss $2\times 2$) y **Q8** (Serendipity cuadrático, Gauss $2\times 2$ o $3\times 3$).
3. Ingesta de mallas desde archivos de texto plano (`.mesh`).
4. Verificación rigurosa mediante **Patch Test** para todos los elementos implementados.
5. Modelado y análisis de una **Presa de Gravedad de Hormigón H°21** ($H=30\text{ m}, B=18\text{ m}, b_{\text{cresta}}=3\text{ m}, H_w=28\text{ m}$):
   - Peso propio y empuje hidrostático.
   - Distribución de tensiones principales ($\sigma_1$) sobre configuración deformada.
   - Verificación de equilibrio estático de momentos (estabilizante vs desestabilizante vs reacciones FEM en la base).
   - Factores de seguridad al vuelco y deslizamiento.
   - Estudio de optimizaciones geométricas.
   - Condiciones de borde elásticas de Robin en la base (resortes de Winkler, $k_t = 0{,}4\,k_n$).

---

## Estructura del Repositorio
```
MSC-TP3-2026/
├── source/
│   ├── elements/      # CST, LST, Q4, Q8
│   ├── materials/     # Material elástico lineal (deformación plana)
│   ├── mesh/          # Clases Mesh, Node y parser .mesh
│   ├── output/        # Visualización en deformada y exportación VTU
│   ├── solvers/       # Solvers del sistema de ecuaciones (K u = f)
│   └── systems/       # Ensamblador global, cargas y condiciones de borde
├── meshes/            # Archivos .mesh de prueba, patch test y de la presa
├── results/           # Archivos de resultados .vtu para ParaView
├── tests/             # Equilibrio de la presa (empotrada y Robin)
├── patch_test.py      # Patch test de CST, LST, Q4 y Q8
├── robin_analysis.py  # Bonus: barrido de kn con CB de Robin en la base
├── requirements.txt   # Dependencias de Python
└── main.py            # Script principal de ejecución
```

---

## Requisitos e Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/bmurina/MSC-TP3-2026.git
   cd MSC-TP3-2026
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux / macOS:
   source .venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uso

```bash
python main.py                       # interactivo (elemento, escala y condición de base)
python main.py Q8 2000               # base empotrada
python main.py Q8 2000 robin 1e9     # base sobre resortes de Winkler, kn = 1e9 N/m3
python main.py ALL 2000              # los cuatro elementos (guarda figuras sin mostrarlas)
python robin_analysis.py Q8          # bonus: empotrada vs. Robin (kn = 1e8 ... 1e11) vs. Navier
python patch_test.py all             # patch test de los cuatro elementos
python tests/test_presa_equilibrium.py
```

Cada corrida de `main.py` guarda en `results/` la deformada coloreada por $|u|$
(`deformada_*.png`) y la tensión principal máxima $\sigma_1$ 3D sobre la deformada
(`sigma1_*.png`). `robin_analysis.py` agrega `robin_base_syy_*.png` con $\sigma_{yy}$ en la base.
