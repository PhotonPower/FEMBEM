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
| `--truncation-radial`, `--truncation-axial` | 5, 10 | Abschneidefaktor in x/y und in z (beide gleich: `--truncation`) |
| `--plate-z0` | 3.8 | Unterkante der Platte in mm, muss > 0 sein |
| `--scale` | 1.0 | Feinheit: teilt Kantenlängen, vervielfacht Schichtzahlen |

Der Abschneidefaktor folgt der Definition des Programms (Ausdehnung der Luft /
Ausdehnung der Leiter, je Achse; `FemDirectRun.cpp`). Das Programm warnt unter 2,
aber 2 oder 3 reichen bei weitem nicht: Die Vorgaben 5 / 10 stammen aus einer
Messreihe am Schwebegleichgewicht ([Bericht](../../docs/design/gmsh-phase6-report.md)).
Mit 3 / 3 lag F_z 24 % zu hoch; axial ist bei 10 ausgeschöpft, radial bei 7,
radial 5 liegt 0,7 % daneben.

Gemessene Größen mit den Vorgaben 5 / 10 (Prismen gesamt, davon Platte), dazu
die Größe der Zerlegung aus Serverläufen (cuDSS, nnz(LU)):

| Aufruf | Prismen | Platte | nnz(LU) |
|---|---|---|---|
| `--scale 0.5` | 52 300 | 812 | |
| `--scale 0.75` | 131 544 | 2 484 | 1,5·10⁹ |
| Vorgabe (`--scale 1`) | 282 030 | 5 904 | |

Bei großen Zerlegungen `FEMBEM_CUDSS_HYBRID=1` setzen: cuDSS lagert dann Teile
in den Hauptspeicher aus. `--scale 0.75` liegt bei der Kraft nur rund 1 % neben
`--scale 1` und ist für Reihen die günstige Wahl.

Die meisten Prismen liegen in der Luft: Jede Schicht, die Spulen und Platte
auflöst, reicht radial durch das ganze Luftgebiet. Das ist der Preis des reinen
Prismennetzes (AP2).

Im Programm geprüft: **0 Kohomologie-Generatoren**, wie für eine Vollscheibe
mit Luft erwartet; die Vorprüfung des FEM-direkt-Pfads meldet keine
Beanstandung und den eingestellten Abschneidefaktor. Am Schwebegleichgewicht
(`--plate-z0 11.3`) liefert der Lauf F_z = 1,037 N bei ausgeschöpftem
Luftgebiet und `--scale 0.75`, Sollwert 1,0497 N.
