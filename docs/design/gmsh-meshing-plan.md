# Vernetzung mit Gmsh: Plan

Ziel: TEAM-28-Netze reproduzierbar und parametrisch mit Gmsh erzeugen, statt
von Hand konvertierte Netze zu verwenden. Gmsh ist ein **externes
Python-Werkzeug**; das Programm selbst bekommt keine Gmsh-Abhängigkeit.

## Feste Entscheidungen

### E1 — Kein Loch in der Platte. Ausdrücklich ausgeschlossen.

Die Platte wird **immer als Vollscheibe** vernetzt. Ein Loch auf der Achse,
auch ein sehr kleines (etwa 0,5 mm), ist **nicht zulässig** — weder als
Vorgabe noch als Option noch als Notlösung für die Vernetzung der Mitte.

Begründung:

- Ein Loch ändert die **Topologie**. Die Platte wird zum Volltorus, der Raum
  um sie ist nicht mehr einfach zusammenhängend, und der Löser findet einen
  zusätzlichen Kohomologie-Generator (siehe Kommentar in `src/fem.cu` bei der
  Generatorensuche). Das ist ein anderes Problem, nicht dasselbe mit kleinem
  Fehler.
- Ein Loch ändert die **Geometrie** des Benchmarks.
- Ergebnisse wären nicht mehr mit dem Standardnetz vergleichbar, das eine
  Vollscheibe ist (Mittelzelle über der Achse).

Durchsetzung: `tools/check_map.py` lehnt ein Plattennetz ab, das die Achse
nicht überdeckt („LOCH auf der Achse — ausgeschlossen“, Exit-Code 1).

### E2 — Hauptpfad: reine Hexaedernetze

Die BEM wird nur aufgebaut, wenn das Netz **ausschließlich aus Hexaedern**
besteht (`bem_possible = mesh.is_pure_hex()` in `src/fem.cu`); Sauter-Schwab
arbeitet auf Vierecks-Panels. Schon ein einziges Prisma, auch im Inneren,
schaltet die BEM ab. Netze für den BEM-FEM-Pfad sind deshalb reine
Hexaedernetze.

### E3 — Mitte der Platte: Hexaederkern (O-Gitter)

Aus E1 und E2 folgt: Die Mitte der Scheibe wird mit einem Hexaederkern
vernetzt — ein Viereck um die Achse, darum vier Randstücke bis zum
Außenradius, dann in z extrudiert. Ein Querschnitt, der bis zur Achse gedreht
wird, liefert Prismen in der Mitte und scheidet nach E2 aus.

### E4 — Weitere Festlegungen

- Geometrie als Parameter, Einheiten mm, Abmessungen wie im Standardnetz
  (Außenspule r 80–95, Innenspule r 27–55, beide z −52…0; Platte r ≤ 65,
  Dicke 3, Lage in z frei).
- Spulen: Rechteck-Querschnitt, viermal um 90° gedreht (der Geo-Kern dreht
  nur um weniger als 180°; die Viertel teilen sich ihre Schnittflächen, das
  Netz ist konform und ohne Naht).
- `component_id` = Nummer der Physical Group (1 Außenspule, 2 Innenspule,
  3 Platte, 0 Luft). Gmsh schreibt diese Nummer in `CellEntityIds`.
- Elemente erster Ordnung; der Löser lehnt zweite Ordnung ab.
- Gmsh-Version festhalten (getestet: 4.15.2 aus pip).

## Phasen

| Phase | Inhalt | Stand |
|---|---|---|
| 0 | Klärung: 360°-Ring, Hexaederkern, BEM und Dreiecke, Lesbarkeit, Topologie | **abgeschlossen**, siehe [gmsh-phase0-report.md](gmsh-phase0-report.md) |
| 1 | `tools/meshcheck.py`: Prüfungen aus `import_bdf.py` herauslösen und gemeinsam nutzen; Tests mit absichtlich kaputten Netzen | offen |
| 2 | `tools/gmsh/team28_hex.py`: Generator mit Parametern (Teilungen in r, z, φ; Kerngröße; Plattenlage), aus dem Prototyp | offen |
| 3 | `tools/gmsh/team28_fem_air.py`: Netz mit Luft für den FEM-direkt-Pfad (`component_id` 0), Größenfelder | offen |
| 4 | `tools/gmsh/import_msh.py`: eigene `.msh`/`.geo`-Dateien, Abbildung Gruppenname → `component_id` | offen |
| 5 | Dokumentation (`docs/meshing.md`), CI-Job (pip + GL-Bibliothek), CTest-Marke `gmsh`, CHANGELOG | offen |
| 6 | optional: Netzserie auf dem Server rechnen — hängt die Stagnation des Lösers vom Netz ab? | offen |

## Abnahme je Netz (ab Phase 2 automatisch)

- `tools/check_map.py` besteht (Zuordnung, Maße, Vollscheibe).
- Keine doppelten Knoten, jede Fläche gehört zu höchstens zwei Zellen.
- Positive Jacobi-Determinante an allen Ecken.
- Volumen je Körper gleich dem exakten N-Eck (N Segmente am Umfang) bis auf
  Rundung.
- Der Löser findet genau **2** Kohomologie-Generatoren (je Ringspule einer;
  die Platte trägt keinen bei).

## Risiken

- Qualität des Hexaederkerns hängt von der Kerngröße ab; ein zu kleiner oder
  zu großer Kern verzerrt die Ecken (gemessen: min. skalierte
  Jacobi-Determinante 0,770 bei Kerngröße 0,3).
- Gmsh braucht auf Servern ohne Grafik trotzdem die GL-Laufzeitbibliotheken.
- Volumina ohne Physical Group fehlen im Ergebnis ohne Fehlermeldung; die
  Prüfung muss das abfangen (vor allem die Luft in Phase 3).
