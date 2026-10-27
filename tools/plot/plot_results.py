import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_simulation():
    filename = "results.csv"
    
    if not os.path.exists(filename):
        print(f"Fehler: Datei '{filename}' nicht gefunden.")
        print("Bitte führen Sie erst die C++ Simulation aus.")
        return

    # Daten laden
    try:
        data = pd.read_csv(filename)
    except Exception as e:
        print(f"Fehler beim Lesen der CSV: {e}")
        return

    # Plot erstellen (3 untereinander)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Titel
    fig.suptitle('TEAM Problem 28: Electrodynamic Levitation Results', fontsize=16)

    # 1. Hubhöhe (z)
    ax1.plot(data['t'], data['z_mm'], 'b-', linewidth=2, label='Levitation Height')
    ax1.axhline(y=3.8, color='r', linestyle='--', alpha=0.5, label='Rest Position (3.8mm)')
    ax1.set_ylabel('Height z [mm]', fontsize=12)
    ax1.set_title('Levitation Height', fontsize=14)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # 2. Strom (I)
    ax2.plot(data['t'], data['current'], 'g-', alpha=0.8, label='Excitation Current')
    ax2.set_ylabel('Current I [A]', fontsize=12)
    ax2.set_title('Coil Current (50 Hz)', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    # 3. Lorentzkraft (F_z)
    ax3.plot(data['t'], data['force_z'], 'm-', linewidth=2, label='Total Lift Force')
    # Optional: Gewichtskraft einzeichnen
    mass = 0.107
    gravity = 9.81
    weight = mass * gravity
    ax3.axhline(y=weight, color='k', linestyle='-.', alpha=0.5, label=f'Weight ({weight:.2f} N)')
    
    ax3.set_ylabel('Force F_z [N]', fontsize=12)
    ax3.set_xlabel('Time t [s]', fontsize=12)
    ax3.set_title('Electrodynamic Levitation Force', fontsize=14)
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für Suptitle lassen
    
    # Speichern und Anzeigen
    plt.savefig("simulation_results.png", dpi=300)
    print("Plot gespeichert als 'simulation_results.png'")
    plt.show()

if __name__ == "__main__":
    plot_simulation()