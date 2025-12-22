import numpy as np
import pandas as pd
import pyvista as pv

# -------------------------------
# Leer y limpiar la tabla de nodos
# -------------------------------
node_disp = "results/node_disp.txt"
stress_data = "results/stress_components.txt"

SCALE = 0.5

def read_nodes(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("NODE") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 7:
                lines.append(parts)

    return pd.DataFrame(lines, columns=["NODE", "X", "Y", "Z", "DX", "DY", "DZ"]).astype(float)

df_disp = read_nodes(node_disp)

def read_stress_table(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("NODE") or line.startswith("#") or line.startswith("---"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                lines.append(parts[:7])
    return pd.DataFrame(lines, columns=["NODE", "SX", "SY", "SZ", "SXY", "SYZ", "SXZ"]).astype(float)

df_stress = read_stress_table(stress_data)

df = pd.merge(df_disp, df_stress, on="NODE")

# Calcular esfuerzo de von Mises aproximado (si no tienes SINT)
df["SINT"] = np.sqrt(df["SX"]**2 - df["SX"]*df["SY"] + df["SY"]**2 + 3*df["SXY"]**2)

# Magnitud del desplazamiento
df["disp_magnitude"] = np.linalg.norm(df[["DX", "DY", "DZ"]].values, axis=1)

# -------------------------------
# Preparar coordenadas deformadas
# -------------------------------
points = df[["X", "Y", "Z"]].values
displacements = df[["DX", "DY", "DZ"]].values
deformed_points = points + SCALE * displacements  # Geometría deformada
disp_magnitude = np.linalg.norm(displacements, axis=1)

# -------------------------------
# Crear PolyData
# -------------------------------
mesh = pv.PolyData(deformed_points)

# Añadir todos los campos como scalars
fields = {
    "Desplazamiento": "disp_magnitude",
    "Esfuerzo SX": "SX",
    "Esfuerzo SY": "SY",
    "Esfuerzo SXY": "SXY",
    "Esfuerzo SINT": "SINT"
}

for label, key in fields.items():
    mesh[key] = df[key].values

# -------------------------------
# Crear superficie aproximada
# -------------------------------
cloud = pv.PolyData(deformed_points)
cloud["disp_magnitude"] = disp_magnitude
surf = cloud.delaunay_2d()  # Triangulación automática

# -------------------------------
# Visualización
# -------------------------------
plotter = pv.Plotter()
plotter.add_mesh(surf, scalars="disp_magnitude", cmap="viridis",
                 point_size=9, render_points_as_spheres=True)

plotter.add_axes()
plotter.show()


# -------------------------------
# Visualización interactiva
# -------------------------------
plotter = pv.Plotter()
print("\n".join([f"{i+1}. {k}" for i, k in enumerate(fields.keys())]))
choice = int(input(f"Elige campo a visualizar (1-{len(fields)}): ")) - 1
chosen_label = list(fields.keys())[choice]
chosen_key = list(fields.values())[choice]

plotter.add_mesh(
    mesh,
    scalars=chosen_key,
    cmap="rainbow",
    point_size=10,
    render_points_as_spheres=True,
    show_scalar_bar=True,
    scalar_bar_args={"title": chosen_label}
)

plotter.add_text(f"Campo: {chosen_label}", position="upper_left", font_size=12)
plotter.add_axes()
plotter.show()
