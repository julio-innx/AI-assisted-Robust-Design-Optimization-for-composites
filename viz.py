import numpy as np
import pandas as pd
import pyvista as pv

# -------------------------------
# Leer y limpiar la tabla de nodos
# -------------------------------
TABLE_FILE = "node_disp.txt"  # Cambia si es otro nombre

lines = []
with open(TABLE_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        # Saltar líneas vacías o encabezados
        if not line or line.startswith("NODE") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 7:  # Debe tener exactamente 7 valores
            lines.append(parts)

# Crear DataFrame
df = pd.DataFrame(lines, columns=["NODE", "X", "Y", "Z", "DX", "DY", "DZ"])
df = df.astype(float)  # Convertir todo a float

# -------------------------------
# Preparar arrays
# -------------------------------
points = df[["X", "Y", "Z"]].values
displacements = df[["DX", "DY", "DZ"]].values
disp_magnitude = np.linalg.norm(displacements, axis=1)

# -------------------------------
# Crear PolyData
# -------------------------------
mesh = pv.PolyData(points)
mesh["disp_magnitude"] = disp_magnitude
mesh["disp_vectors"] = displacements

# -------------------------------
# Visualizar
# -------------------------------
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars="disp_magnitude", cmap="viridis", point_size=8, render_points_as_spheres=True)
plotter.add_arrows(mesh.points, mesh["disp_vectors"], mag=0.1, color="red")

plotter.add_axes()
plotter.show_grid()
plotter.show()
