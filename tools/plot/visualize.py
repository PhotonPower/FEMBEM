import pyvista as pv
import os
import sys

def main():
    # Dateinamen, die vom C++ Programm generiert wurden
    files = {
        "plate": "sim_plate.vtk",
        "coil_inner": "sim_coil_inner.vtk",
        "coil_outer": "sim_coil_outer.vtk"
    }

    # Überprüfen, ob die Dateien existieren
    missing_files = [f for f in files.values() if not os.path.exists(f)]
    if missing_files:
        print("Fehler: Folgende Dateien wurden nicht gefunden:")
        for f in missing_files:
            print(f" - {f}")
        print("\nBitte zuerst das C++ Programm kompilieren und ausführen.")
        sys.exit(1)

    print("Starte 3D-Visualisierung...")

    # Plotter initialisieren
    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background("white")

    # --- 1. Die Platte laden und hinzufügen ---
    plate_mesh = pv.read(files["plate"])
    
    plotter.add_mesh(plate_mesh, 
                     color="silver", 
                     show_edges=True,
                     edge_color="black",
                     line_width=1.0,
                     smooth_shading=True,
                     specular=0.5,
                     label="Aluminium Platte (FEM)")

    # --- 2. Innere Spule laden ---
    coil_in_mesh = pv.read(files["coil_inner"])
    
    plotter.add_mesh(coil_in_mesh, 
                     color="#B87333", 
                     show_edges=True,
                     edge_color="black",
                     opacity=0.9,
                     label="Spule Innen (w1)")

    # --- 3. Äußere Spule laden ---
    coil_out_mesh = pv.read(files["coil_outer"])
    
    plotter.add_mesh(coil_out_mesh, 
                     color="#8B4513", 
                     show_edges=True,
                     edge_color="black",
                     opacity=0.9,
                     label="Spule Außen (w2)")

    # --- Szene aufhübschen ---
    
    plotter.add_axes()
    
    # KORRIGIERTE ZEILE: alpha entfernt, Farbe heller gemacht
    plotter.show_grid(color='lightgray') 
    
    plotter.add_legend(bcolor=(0.9, 0.9, 0.9), size=[0.2, 0.2])

    plotter.add_text("TEAM Problem 28 - Mesh Visualization\nHexahedral Mesh & GA Optimization", 
                     position='upper_left', 
                     font_size=10, 
                     color='black')

    plotter.view_isometric()

    print("Visualisierung läuft. Schließen Sie das Fenster zum Beenden.")
    print("Steuerung:")
    print(" - Linke Maus: Drehen")
    print(" - Shift + Linke Maus: Verschieben")
    print(" - Scrollen: Zoomen")
    print(" - 'w': Wireframe Modus umschalten")
    
    plotter.show()

if __name__ == "__main__":
    main()