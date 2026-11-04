# Daten zu TEAM Problem 28

| Datei | Inhalt |
|---|---|
| `team28_hex.vtk` | **Standardnetz.** Gmsh-Hexaedernetz, erzeugt mit `python3 tools/gmsh/team28_hex.py --scale 1.0 -o data/team28/team28_hex.vtk` (Gmsh 4.15.2; die Kopfzeile der Datei hält Parameter und Version fest). Millimeter, 4944 Hexaeder, 6916 Knoten, `component_id` wie unten: 1 Außenspule (1536 Zellen), 2 Innenspule (1536), 3 Platte (1872). Geprüft mit `meshcheck.py --require-hex --separate-bodies` und `check_map.py`, 2 Kohomologie-Generatoren. Die Gmsh-Ausgabe kann zwischen Versionen byteweise abweichen; die CI prüft deshalb nur `meshcheck`/`check_map` und die Zählwerte, nicht die Bytes. |
| `team28.vtk` | Legacy-Netz (früheres Standardnetz) der TEAM-28-Geometrie: Legacy-VTK, ASCII, Unstructured Grid, **Millimeter**, 4805 Hexaeder, 18690 Kanten. Jede Zelle trägt eine `component_id`: 1 Außenspule (1200 Zellen), 2 Innenspule (1400), 3 Platte (2205). Der Außenraum ist nicht vernetzt, ihn übernimmt die BEM. |
| `levitation_data.csv` | Zeit-Höhen-Kurve der Platte zum Vergleich: Spalten `t (s)` und `z (mm)`. Wird von `tools/plot/plot_team_data.py` dargestellt. |

Das Programm öffnet `team28_hex.vtk` über den relativen Namen im
Arbeitsverzeichnis. CMake kopiert beide Netze beim Konfigurieren ins
Build-Verzeichnis; alternativ `--mesh=PFAD` (z. B. `--mesh=team28.vtk`).

## Herkunft

Die Geometrie und die Materialdaten folgen der Beschreibung von
*TEAM Workshop Problem 28 (An Electrodynamic Levitation Device)*, einem
Benchmark der Compumag-Gemeinde; die Quellenangabe steht in `src/fem.cu` bei
den physikalischen Parametern. Werkzeuge zum Konvertieren und Erzeugen von
Netzen: `tools/vtk_konverter.py` (Nastran nach VTK), `tools/import_bdf.py`
(BDF für den FEM-direkt-Pfad), `tools/make_coarse_team28.py` (grobe
Ersatznetze).

Die genaue Herkunft der Kurve in `levitation_data.csv` ist im Repository
bisher nicht dokumentiert. Vor einem Vergleich mit Veröffentlichungen bitte
gegen die Originalbeschreibung des Benchmarks prüfen.

## Spulenzuordnung prüfen

Vertauschte Spulen-IDs erzeugen keinen Absturz, sondern ein falsches Feld,
weil die Vorzeichen der Ströme an der `component_id` hängen. Nach jeder
Konvertierung:

    python3 tools/check_map.py data/team28/team28_hex.vtk
    python3 tools/check_map.py data/team28/team28.vtk
