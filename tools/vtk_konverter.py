import tkinter as tk
from tkinter import filedialog, messagebox
import re
import os

class NastranToVTKConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Nastran BDF zu VTK Konverter")
        self.root.geometry("500x450")
        
        # Datenstrukturen
        self.nodes = {}       # id: (x, y, z)
        self.elements = []    # list of dicts: {'id': id, 'pid': pid, 'type': type, 'nodes': [n1, n2...]}
        self.components = {}  # pid: name
        self.bdf_filepath = ""
        
        self.setup_gui()

    def setup_gui(self):
        # UI Elemente
        self.lbl_info = tk.Label(self.root, text="Bitte ein Nastran .bdf File laden:", font=("Arial", 10, "bold"))
        self.lbl_info.pack(pady=10)
        
        self.btn_load = tk.Button(self.root, text="1. BDF Datei laden", command=self.load_bdf, width=30)
        self.btn_load.pack(pady=5)
        
        self.text_stats = tk.Text(self.root, height=15, width=55, state=tk.DISABLED)
        self.text_stats.pack(pady=10)
        
        self.btn_convert = tk.Button(self.root, text="2. In VTK konvertieren", command=self.export_vtk, width=30, state=tk.DISABLED)
        self.btn_convert.pack(pady=5)

    def log(self, message):
        self.text_stats.config(state=tk.NORMAL)
        self.text_stats.insert(tk.END, message + "\n")
        self.text_stats.see(tk.END)
        self.text_stats.config(state=tk.DISABLED)

    def parse_nastran_float(self, s):
        s = s.strip()
        if not s: return 0.0
        # Behandle Nastrans exponentielle Schreibweise ohne 'E' (z.B. 1.234-4 anstelle von 1.234E-4)
        if '-' in s[1:] and 'e' not in s.lower():
            idx = s.rfind('-')
            s = s[:idx] + 'e' + s[idx:]
        elif '+' in s[1:] and 'e' not in s.lower():
            idx = s.rfind('+')
            s = s[:idx] + 'e' + s[idx:]
        return float(s)

    def load_bdf(self):
        self.bdf_filepath = filedialog.askopenfilename(filetypes=[("Nastran BDF", "*.bdf *.dat"), ("All Files", "*.*")])
        if not self.bdf_filepath:
            return
        
        self.text_stats.config(state=tk.NORMAL)
        self.text_stats.delete(1.0, tk.END)
        self.text_stats.config(state=tk.DISABLED)
        
        self.nodes.clear()
        self.elements.clear()
        self.components.clear()
        
        # Regex für Komponenten-Namen
        comp_regex = re.compile(r'\$HMNAME COMP\s+(\d+)\s*"(.*)"')
        
        self.log(f"Lade Datei: {os.path.basename(self.bdf_filepath)}...")
        
        try:
            with open(self.bdf_filepath, 'r') as f:
                lines = f.readlines()
                
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # GRID (Knoten)
                if line.startswith("GRID"):
                    try:
                        nid = int(line[8:16].strip())
                        x = self.parse_nastran_float(line[24:32])
                        y = self.parse_nastran_float(line[32:40])
                        z = self.parse_nastran_float(line[40:48])
                        self.nodes[nid] = (x, y, z)
                    except ValueError:
                        pass
                
                # CQUAD4 (4-Knoten Shell)
                elif line.startswith("CQUAD4"):
                    try:
                        eid = int(line[8:16].strip())
                        pid = int(line[16:24].strip())
                        n1 = int(line[24:32].strip())
                        n2 = int(line[32:40].strip())
                        n3 = int(line[40:48].strip())
                        n4 = int(line[48:56].strip())
                        self.elements.append({'id': eid, 'pid': pid, 'type': 'CQUAD4', 'nodes': [n1, n2, n3, n4]})
                    except ValueError:
                        pass
                
                # CHEXA (8-Knoten Solid) - Benötigt 2 Zeilen!
                elif line.startswith("CHEXA"):
                    try:
                        eid = int(line[8:16].strip())
                        pid = int(line[16:24].strip())
                        n1 = int(line[24:32].strip())
                        n2 = int(line[32:40].strip())
                        n3 = int(line[40:48].strip())
                        n4 = int(line[48:56].strip())
                        n5 = int(line[56:64].strip())
                        n6 = int(line[64:72].strip())
                        
                        i += 1 # Nächste Zeile für n7 und n8 einlesen
                        next_line = lines[i]
                        n7 = int(next_line[8:16].strip())
                        n8 = int(next_line[16:24].strip())
                        
                        self.elements.append({'id': eid, 'pid': pid, 'type': 'CHEXA', 'nodes': [n1, n2, n3, n4, n5, n6, n7, n8]})
                    except ValueError:
                        pass
                
                # Komponenten Name mapping
                elif line.startswith("$HMNAME COMP"):
                    match = comp_regex.search(line)
                    if match:
                        pid = int(match.group(1))
                        name = match.group(2)
                        self.components[pid] = name
                        
                i += 1
                
            self.show_statistics()
            self.btn_convert.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Lesen der Datei:\n{str(e)}")

    def show_statistics(self):
        self.log("--- STATISTIK ---")
        self.log(f"Anzahl Knoten: {len(self.nodes)}")
        self.log(f"Anzahl Elemente Gesamt: {len(self.elements)}")
        
        # Elemente nach Knotenanzahl sortieren
        elem_stats = {}
        for el in self.elements:
            n_count = len(el['nodes'])
            elem_stats[n_count] = elem_stats.get(n_count, 0) + 1
            
        for n_nodes, count in sorted(elem_stats.items()):
            self.log(f" -> Elemente mit {n_nodes} Knoten: {count}")
            
        self.log("\nGefundene Komponenten (HMNAME COMP):")
        if self.components:
            for pid, name in self.components.items():
                self.log(f" -> ID {pid}: '{name}'")
        else:
            self.log(" -> Keine benannten Komponenten gefunden (Nutze Default IDs).")

    def export_vtk(self):
        default_out = os.path.splitext(self.bdf_filepath)[0] + ".vtk"
        vtk_filepath = filedialog.asksaveasfilename(defaultextension=".vtk", 
                                                    initialfile=os.path.basename(default_out),
                                                    filetypes=[("VTK File", "*.vtk")])
        if not vtk_filepath:
            return
        
        try:
            with open(vtk_filepath, 'w') as f:
                # 1. VTK Header
                f.write("# vtk DataFile Version 3.0\n")
                f.write("Converted from Nastran\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                
                # Mapping Nastran Node ID -> VTK Node Index (0 bis N-1)
                node_id_to_vtk_idx = {}
                
                # 2. Knoten schreiben
                f.write(f"POINTS {len(self.nodes)} float\n")
                for idx, (nid, coords) in enumerate(self.nodes.items()):
                    f.write(f"{coords[0]} {coords[1]} {coords[2]}\n")
                    node_id_to_vtk_idx[nid] = idx
                
                # 3. Zellen vorbereiten (Anzahl und Größe berechnen)
                # Größe = Summe aus (1 + anzahl_knoten_pro_element) für alle Elemente
                total_cell_size = sum(1 + len(el['nodes']) for el in self.elements)
                f.write(f"\nCELLS {len(self.elements)} {total_cell_size}\n")
                
                for el in self.elements:
                    node_indices = [str(node_id_to_vtk_idx[n]) for n in el['nodes']]
                    f.write(f"{len(node_indices)} {' '.join(node_indices)}\n")
                
                # 4. Zell-Typen schreiben
                f.write(f"\nCELL_TYPES {len(self.elements)}\n")
                for el in self.elements:
                    if el['type'] == 'CQUAD4':
                        f.write("9\n")  # VTK_QUAD = 9
                    elif el['type'] == 'CHEXA':
                        f.write("12\n") # VTK_HEXAHEDRON = 12
                
                # 5. Component/Property ID als Zell-Daten anfügen (zum Einfärben in ParaView)
                f.write(f"\nCELL_DATA {len(self.elements)}\n")
                f.write("SCALARS Component_ID int 1\n")
                f.write("LOOKUP_TABLE default\n")
                for el in self.elements:
                    f.write(f"{el['pid']}\n")
                    
            self.log(f"\nErfolgreich exportiert nach:\n{vtk_filepath}")
            messagebox.showinfo("Erfolg", "VTK Datei erfolgreich erstellt!")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Schreiben der VTK Datei:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NastranToVTKConverter(root)
    root.mainloop()