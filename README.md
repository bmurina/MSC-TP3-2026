# MSC - Trabajo Práctico N°3: Estado Plano de Tensión

**Mecánica de Sólidos Computacional (30.53) - ITBA**

Solver bidimensional por Elementos Finitos (FEM) en **Python** para análisis elástico lineal en estado plano de tensión. Aplicación al análisis estructural y de estabilidad de una presa de gravedad de hormigón.

---

## Integrantes
- Bruno Murina (`bmurina@itba.edu.ar`)

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
   - Condiciones de borde elásticas de Robin en la base.

---

## Estructura del Repositorio
```
MSC-TP3-2026/
├── source/
│   ├── elements/      # CST, LST, Q4, Q8
│   ├── materials/     # Material elástico lineal (tensión plana)
│   ├── mesh/          # Clases Mesh, Node y parser .mesh
│   ├── output/        # Visualización en deformada y exportación VTU
│   ├── solvers/       # Solvers del sistema de ecuaciones (K u = f)
│   └── systems/       # Ensamblador global, cargas y condiciones de borde
├── meshes/            # Archivos .mesh de prueba, patch test y de la presa
├── results/           # Archivos de resultados .vtu para ParaView
├── tests/             # Validación del Patch Test
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
