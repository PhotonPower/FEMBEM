# Daten zu TEAM Problem 28

| Datei | Inhalt |
|---|---|
| `team28.vtk` | Standardnetz der TEAM-28-Geometrie: Legacy-VTK, ASCII, Unstructured Grid, **Millimeter**, 4805 Hexaeder, 18690 Kanten. Jede Zelle trägt eine `component_id`: 1 Außenspule (1200 Zellen), 2 Innenspule (1400), 3 Platte (2205). Der Außenraum ist nicht vernetzt, ihn übernimmt die BEM. |
| `levitation_data.csv` | Zeit-Höhen-Kurve der Platte zum Vergleich: Spalten `t (s)` und `z (mm)`. Wird von `tools/plot/plot_team_data.py` dargestellt. |

Das Programm öffnet `team28.vtk` über den relativen Namen im
Arbeitsverzeichnis. CMake kopiert die Datei beim Konfigurieren ins
Build-Verzeichnis; alternativ `--mesh=PFAD`.

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

    python3 tools/check_map.py data/team28/team28.vtk
