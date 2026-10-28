# Netze

Wie Netze in FEMBEM hineinkommen, welche Regeln sie erfüllen müssen und wie
man sie prüft. Plan und Hintergrund der Gmsh-Werkzeuge:
[design/gmsh-meshing-plan.md](design/gmsh-meshing-plan.md).

## Woher ein Netz kommen kann

| Quelle | Werkzeug | Ergebnis |
|---|---|---|
| Standardnetz | — | `data/team28/team28.vtk`, 4805 Hexaeder; liegt nach dem Konfigurieren im Build-Verzeichnis |
| Gmsh, parametrisch, BEM-FEM | `tools/gmsh/team28_hex.py` | reines Hexaedernetz der Leiter, beliebig fein, geprüft |
| Gmsh, parametrisch, FEM-direkt | `tools/gmsh/team28_fem_air.py` | Prismennetz mit Luftgebiet, Abschneidefaktor einstellbar, geprüft |
| Nastran-BDF | `tools/import_bdf.py`, `tools/vtk_konverter.py` | Netz aus einem externen Vernetzer |
| grobe Ersatznetze | `tools/make_coarse_team28.py` | strukturierte Netze für die Netzstudie |

Gestartet wird mit `./fembem --mesh=netz.vtk`.

## Format

Legacy-VTK, ASCII, `UNSTRUCTURED_GRID` (`src/VtkReader.cpp`):

- Zelltypen: Tetraeder (10), Hexaeder (12), Prismen (13), **erster Ordnung**.
  Elemente zweiter Ordnung und Pyramiden werden abgelehnt.
- `component_id` je Zelle im ersten `SCALARS`-Block unter `CELL_DATA`:
  0 Luft, 1 Außenspule, 2 Innenspule, 3 Platte. Die Vorzeichen der
  Spulenströme hängen an dieser Nummer, nicht an der Geometrie —
  vertauschte Spulen rechnen ohne Fehlermeldung falsch.
- Einheiten: **Millimeter**.

## Regeln

1. **Die Platte ist eine Vollscheibe. Ein Loch ist ausgeschlossen.** Ein Loch
   ändert die Topologie (zusätzlicher Kohomologie-Generator) und die
   Geometrie. `tools/check_map.py` lehnt ein solches Netz ab.
2. **BEM nur mit reinen Hexaedernetzen.** Enthält das Netz ein einziges
   Prisma oder einen Tetraeder, baut das Programm die BEM nicht auf
   (`mesh.is_pure_hex()`); dann bleibt nur `--fem-direct`.
3. **Körper getrennt.** Im BEM-Pfad (Leiter ohne Luft) darf kein Knoten zu
   zwei Körpern gehören; Platte und Spule wären sonst leitend verbunden.
4. **Platte oberhalb der Spulen**, ohne Berührung.
5. **FEM-direkt: Luft umschließt alles.** Jede Randfläche gehört zur Luft
   (`component_id` 0); dort sitzt die Randbedingung A × n = 0. Prismen müssen
   affin sein (senkrecht extrudiert), sonst bleibt ein Fehler, der linear mit
   der Verzerrung wächst. Der Abschneidefaktor (Ausdehnung Luft / Ausdehnung
   Leiter) braucht **radial mindestens 5 und axial 10**: Mit 3 / 3 lag die
   Kraft am Schwebegleichgewicht 24 % zu hoch (Messreihe in
   [design/gmsh-phase6-report.md](design/gmsh-phase6-report.md)). Das Programm
   warnt erst unter 2.
6. **Konform, richtig orientiert:** jede Fläche gehört höchstens zwei Zellen,
   die Jacobi-Determinante ist an allen Ecken positiv.

## Prüfen

    python3 tools/meshcheck.py netz.vtk --require-hex --separate-bodies    # BEM-FEM
    python3 tools/meshcheck.py netz.vtk --air-enclosed --prism-affinity    # FEM-direkt
    python3 tools/check_map.py netz.vtk

`meshcheck.py` prüft Zelltypen, doppelte und verwaiste Knoten, getrennte
Körper, Konformität, Orientierung (Hexaeder, Prismen, Tetraeder) und Volumen.
`import_bdf.py` prüft die von ihm geschriebene Datei damit ein zweites Mal. `check_map.py` prüft die
Zuordnung der Spulen, die Maße und die Vollscheibe. Beide enden mit
Exit-Code 1, wenn etwas nicht stimmt. `team28_hex.py` ruft beide für jedes
erzeugte Netz selbst auf.

Was nur das Programm prüfen kann: Beim Start meldet es die gefundenen
Kohomologie-Generatoren. Für TEAM 28 sind es

- im **BEM-FEM-Pfad** (Leiter ohne Luft) genau **2**, je Ringspule einer;
- im **FEM-direkt-Pfad** (mit Luft) genau **0**: leitend ist nur die Platte,
  und die ist eine Vollscheibe.

    ./fembem 1 --mesh=netz.vtk | grep Kohomologie-Generatoren

Eine andere Zahl heißt: Die Topologie stimmt nicht (zum Beispiel ein Loch in
der Platte, das einen Generator mehr ergibt, oder zwei verbundene Körper).

## Netze mit Gmsh erzeugen

    python3 -m venv ~/venv-gmsh
    ~/venv-gmsh/bin/pip install -r tools/requirements-gmsh.txt
    ~/venv-gmsh/bin/python tools/gmsh/team28_hex.py -o team28_gmsh.vtk

Optionen und gemessene Qualität: [tools/gmsh/README.md](../tools/gmsh/README.md).
Gmsh braucht auch ohne Grafik die GL-Laufzeitbibliotheken (Ubuntu:
`libglu1-mesa`, `libgl1` und einige X-Bibliotheken; RHEL: vermutlich
`mesa-libGLU`).

Für den FEM-direkt-Pfad mit Luftgebiet (Vorgabe radial 5 / axial 10):

    ~/venv-gmsh/bin/python tools/gmsh/team28_fem_air.py -o team28_fem_air.vtk
    ~/venv-gmsh/bin/python tools/gmsh/team28_fem_air.py --plate-z0 11.3 --scale 0.75 -o gleichgewicht.vtk

Eine Netzserie für die Frage, ob das Verhalten des Lösers von der
Netzfeinheit abhängt:

    for s in 0.5 1 2 3; do
        ~/venv-gmsh/bin/python tools/gmsh/team28_hex.py --scale $s -o serie_$s.vtk
    done

## Tests

    cmake -B build -DPython3_EXECUTABLE=~/venv-gmsh/bin/python
    ctest --test-dir build -L tools      # Prüfmodule, braucht numpy
    ctest --test-dir build -L gmsh       # Generator, braucht gmsh

Ohne gmsh werden die `gmsh`-Tests beim Konfigurieren ausgelassen (mit
Hinweis); alle anderen Tests bleiben unverändert.

## Fallstricke

- **Prismen statt Hexaeder aus Gmsh, ohne Meldung:** Netzvorgaben im Geo-Kern
  müssen über `gmsh.model.geo.mesh.*` gesetzt werden; über `gmsh.model.mesh.*`
  gehen sie beim nächsten `synchronize()` verloren.
- **Knoten nicht verschmelzen** (`removeDuplicateNodes`): Es vereint Knoten
  verschiedener Körper, sobald sie zufällig zusammenfallen.
- **Prismen aus Gmsh sind umgekehrt orientiert** (in der Konvention des
  Lösers, `include/NedelecPrism.cuh`). Vor der Verwendung die Knotenfolge
  (0, 2, 1, 3, 5, 4) anwenden; `meshcheck.py` meldet die Orientierung.
  Tetraeder aus Gmsh sind richtig orientiert.
- **Kleinster Radius ist kein Lochkriterium:** Das Standardnetz hat keinen
  Knoten unter r = 1,84 mm und ist trotzdem eine Vollscheibe (Mittelzelle über
  der Achse).
