import pyvista as pv
import numpy as np
import struct
import os

def load_simulation_data(vtk_file, bin_file):
    # 1. Geometrie laden
    if not os.path.exists(vtk_file):
        print(f"VTK Datei {vtk_file} fehlt!")
        return None, None
        
    mesh = pv.read(vtk_file)
    print(f"Netz geladen: {mesh.n_cells} Zellen")

    # 2. Binärdaten lesen
    if not os.path.exists(bin_file):
        print("Binärdatei fehlt!")
        return None, None

    steps = []
    times = []

    with open(bin_file, "rb") as f:
        # Header lesen
        try:
            header_data = f.read(12)
            if len(header_data) < 12:
                print("Datei zu klein für Header")
                return None, None
                
            magic, n_steps, n_cells = struct.unpack('iii', header_data)
            
            if magic != 0xB00B1E5:
                print(f"Ungültiges Dateiformat! Magic={magic:X}")
                return None, None
                
            print(f"Lese {n_steps} Zeitschritte für {n_cells} Zellen...")
            
            if n_cells != mesh.n_cells:
                print(f"Warnung: VTK hat {mesh.n_cells} Zellen, Binärdatei hat {n_cells}!")

            # Daten lesen
            for _ in range(n_steps):
                # Zeitstempel
                t_bytes = f.read(8)
                if len(t_bytes) < 8: break
                t = struct.unpack('d', t_bytes)[0]
                times.append(t)
                
                # Vektordaten (n_cells * 3 * float4)
                # 3 floats pro Zelle (Bx, By, Bz)
                data_size = n_cells * 3 * 4
                data_bytes = f.read(data_size)
                
                if len(data_bytes) < data_size:
                    print("Datei unerwartet zu Ende.")
                    break
                    
                vectors = np.frombuffer(data_bytes, dtype=np.float32).reshape((n_cells, 3))
                steps.append(vectors)
        except struct.error as e:
            print(f"Fehler beim Lesen: {e}")
            return None, None
                
    return mesh, steps, times

def main():
    mesh, steps, times = load_simulation_data("sim_plate.vtk", "field_data.bin")
    
    if not steps:
        print("Keine Daten geladen. Wurde die Simulation ausgeführt?")
        return

    print("Starte Visualisierung...")

    # Plotter Setup
    p = pv.Plotter()
    p.add_text("TEAM 28 - Magnetic Field (B) Visualization", font_size=12)
    
    # Initiale Daten an das Netz binden
    mesh.cell_data["B-Field"] = steps[0]
    mesh.cell_data["Magnitude"] = np.linalg.norm(steps[0], axis=1)
    
    # Pfeile erstellen (Glyphs)
    # Wir erzeugen Punkte in den Zellzentren
    centers = mesh.cell_centers()
    centers.cell_data["B-Field"] = steps[0]
    
    # Pfeile skalieren und orientieren
    # KORREKTUR: '#' statt '//'
    arrows = centers.glyph(orient="B-Field", scale="B-Field", factor=0.005) 
    
    # Das Netz (Farbe nach Magnitude)
    p.add_mesh(mesh, scalars="Magnitude", cmap="plasma", show_edges=True, opacity=0.9, label="Plate B-Mag")
    
    # Die Pfeile (Optional, kann man auskommentieren für Performance)
    # p.add_mesh(arrows, color="black") 
    
    slider_label = "Time Step"
    
    def update_step(value):
        idx = int(value)
        if idx >= len(steps): idx = len(steps) - 1
        
        current_B = steps[idx]
        t_current = times[idx]
        
        # Netzdaten aktualisieren
        mesh.cell_data["B-Field"] = current_B
        mesh.cell_data["Magnitude"] = np.linalg.norm(current_B, axis=1)
        
        # Skalarbar Bereich anpassen (Dynamisch)
        max_val = np.max(mesh.cell_data["Magnitude"])
        if max_val < 1e-9: max_val = 1.0 # Division durch Null verhindern
        p.update_scalar_bar_range([0, max_val])
        
        # Text aktualisieren
        p.add_text(f"Time: {t_current:.4f} s | Step: {idx}", name="info_text", position='upper_right')

    # Slider hinzufügen
    p.add_slider_widget(update_step, [0, len(steps)-1], title=slider_label, fmt="%.0f")
    
    # Initialer Text
    p.add_text(f"Time: {times[0]:.4f} s | Step: 0", name="info_text", position='upper_right')
    
    p.show_grid()
    p.show()

if __name__ == "__main__":
    main()