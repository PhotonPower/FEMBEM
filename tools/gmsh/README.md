# Netze mit Gmsh

Werkzeuge, die Netze für FEMBEM mit [Gmsh](https://gmsh.info) erzeugen.
Plan, feste Entscheidungen und Stand: [docs/design/gmsh-meshing-plan.md](../../docs/design/gmsh-meshing-plan.md).

## Einrichten

    python3 -m venv ~/venv-gmsh
    ~/venv-gmsh/bin/pip install gmsh numpy

Gmsh lädt auch ohne Grafik die GL-Laufzeitbibliotheken (Ubuntu: `libglu1-mesa`,
`libgl1` und einige X-Bibliotheken; siehe
[Bericht zu Phase 0](../../docs/design/gmsh-phase0-report.md)). Getestet mit
Gmsh 4.15.2.

## `team28_hex.py`: TEAM-28-Netz aus reinen Hexaedern

    ~/venv-gmsh/bin/python tools/gmsh/team28_hex.py -o team28_gmsh.vtk
    ~/venv-gmsh/bin/python tools/gmsh/team28_hex.py --help

Für den BEM-FEM-Pfad (nur Hexaeder, Leiter ohne Luft). Die Platte ist immer
eine **Vollscheibe**; ein Loch ist ausgeschlossen und nicht einstellbar.

| Option | Vorgabe | Bedeutung |
|---|---|---|
| `--n-quarter` | 12 | Zellen je Viertelkreis, für alle Körper gleich |
| `--coil-nr`, `--coil-nz` | 4, 8 | Spulenzellen radial und in z |
| `--plate-nr`, `--plate-nz` | 10, 3 | Plattenzellen radial (Kern bis Rand) und in z |
| `--core` | 0.4 | Kerngröße: Abstand der Kern-Ecken von der Achse relativ zu 65 mm |
| `--plate-grading` | 1.0 | radiale Progression; < 1 verfeinert zum Plattenrand |
| `--plate-z0` | 3.8 | Unterkante der Platte in mm, muss > 0 sein |
| `--scale` | 1.0 | Faktor auf alle Teilungen, für Netzserien |

Mit den Vorgaben entstehen 4944 Hexaeder (Standardnetz: 4805). Nach dem
Schreiben prüft das Werkzeug das Netz mit `tools/meshcheck.py` und
`tools/check_map.py` und endet mit Exit-Code 1, wenn etwas nicht stimmt.

Gemessene Qualität der Platte (kleinste skalierte Jacobi-Determinante) nach
Kerngröße, sonst Vorgaben:

| `--core` | 0.10 | 0.20 | 0.30 | **0.40** | 0.50 | 0.60 |
|---|---|---|---|---|---|---|
| Platte | 0.555 | 0.705 | 0.770 | **0.802** | 0.814 | 0.811 |

0.4 legt die Kern-Ecken bei 26 mm, knapp innerhalb der Innenspule (27 mm), so
dass der Bereich unter den Spulen aus Ringzellen besteht. Ob das die
Genauigkeit verbessert, ist nicht gemessen (Phase 6 des Plans).

Was das Programm danach prüft: Der Löser muss genau **2**
Kohomologie-Generatoren finden (je Ringspule einer). Geprüft für die
Vorgaben, `--scale 0.5` und `--n-quarter 9 --plate-grading 0.85`.
