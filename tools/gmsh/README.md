# Netze mit Gmsh

Werkzeuge, die Netze für FEMBEM mit [Gmsh](https://gmsh.info) erzeugen.
Plan, feste Entscheidungen und Stand: [docs/design/gmsh-meshing-plan.md](../../docs/design/gmsh-meshing-plan.md).

## Einrichten

    python3 -m venv ~/venv-gmsh
    ~/venv-gmsh/bin/pip install -r tools/requirements-gmsh.txt

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

## `team28_fem_air.py`: Netz mit Luftgebiet für FEM-direkt

    ~/venv-gmsh/bin/python tools/gmsh/team28_fem_air.py -o team28_fem_air.vtk
    ~/venv-gmsh/bin/python tools/gmsh/team28_fem_air.py --help

Für `./fembem --fem-direct` (bzw. `--fem-direct-iterative`). Reines
Prismennetz: Dreiecksnetz in der Ebene, schichtweise senkrecht extrudiert, damit
alle Prismen affin sind. Die Platte ist eine **Vollscheibe**. Prismen werden vor
dem Schreiben in die Konvention des Lösers gedreht.

| Option | Vorgabe | Bedeutung |
|---|---|---|
| `--h` | 5 | Kantenlänge in der Ebene an Leitern und Platte, mm |
| `--h-air` | 35 | Kantenlänge am Außenrand der Luft, mm |
| `--coil-nz`, `--gap-nz`, `--plate-nz` | 6, 3, 4 | Schichten über Spulenhöhe, Spalt und Plattendicke |
| `--air-first`, `--air-growth` | 3, 1.4 | erste Luftschicht am Leiter (mm) und Wachstum nach außen |
| `--truncation` | 3 | Abschneidefaktor radial und axial (einzeln: `--truncation-radial`, `--truncation-axial`) |
| `--plate-z0` | 3.8 | Unterkante der Platte in mm, muss > 0 sein |
| `--scale` | 1.0 | Feinheit: teilt Kantenlängen, vervielfacht Schichtzahlen |

Der Abschneidefaktor folgt der Definition des Programms (Ausdehnung der Luft /
Ausdehnung der Leiter, je Achse; `FemDirectRun.cpp`). Das Programm warnt unter 2.

Gemessene Größen (Prismen gesamt, davon Platte):

| Aufruf | Prismen | Platte | Blocksystem (2n) |
|---|---|---|---|
| `--scale 0.5` | 24 684 | | 104 106 Zeilen |
| Vorgabe | 140 616 | 5 904 | |
| `--truncation 2` / `5` | 96 416 / 249 798 | | |
| `--plate-nz 6 --h 4` | 206 944 | | |

Die meisten Prismen liegen in der Luft: Jede Schicht, die Spulen und Platte
auflöst, reicht radial durch das ganze Luftgebiet. Das ist der Preis des reinen
Prismennetzes (AP2).

Im Programm geprüft (grobes Netz): **0 Kohomologie-Generatoren**, wie für eine
Vollscheibe mit Luft erwartet; die Vorprüfung des FEM-direkt-Pfads meldet
keine Beanstandung und den eingestellten Abschneidefaktor.
