import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV einlesen
from pathlib import Path
DATA = Path(__file__).resolve().parents[2] / "data" / "team28" / "levitation_data.csv"
df = pd.read_csv(DATA)

# Alle Zeit- und Höhenwerte in eine einzige Liste zusammenführen
t_all = []
z_all = []

# Durch alle Spaltenpaare iterieren (t1, z1), (t2, z2), ...
for i in range(0, len(df.columns), 2):
    t_col = df.columns[i]
    z_col = df.columns[i + 1]
    
    # Gültige Werte extrahieren (ohne NaN)
    t_vals = df[t_col].dropna().values
    z_vals = df[z_col].dropna().values
    
    # Alle Werte an die Gesamtliste anhängen
    t_all.extend(t_vals)
    z_all.extend(z_vals)

# Sicherstellen, dass die Daten nach Zeit sortiert sind
sorted_indices = np.argsort(t_all)
t_sorted = np.array(t_all)[sorted_indices]
z_sorted = np.array(z_all)[sorted_indices]

# Plot erstellen
plt.figure(figsize=(12, 7))

# Einzelne Linie für die gesamte Messung
plt.plot(t_sorted, z_sorted, 
         linewidth=2.5, 
         color="#4B0082",  # Indigo für professionellen Look
         alpha=0.95,
         label="Magnetische Levitation")

# Achsenbeschriftung und Design
plt.xlabel('Zeit $t$ (s)', fontsize=12, fontweight='bold')
plt.ylabel('Höhe $z$ (mm)', fontsize=12, fontweight='bold')
plt.title('Magnetische Levitation: Höhenverlauf über Zeit', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.7)
plt.legend(framealpha=0.9)
plt.tight_layout()

# Plot speichern und anzeigen
plt.savefig('levitation_single_measurement.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot als 'levitation_single_measurement.png' gespeichert.")
print(f"Anzahl der Datenpunkte: {len(t_sorted)}")