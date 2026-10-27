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

- **Das aktuelle TEAM-28-Problem hat kein Loch.** Ein Loch steht im Paper nur
  als Ausblick. Die Masse von 0,107 kg (Sollkraft im Schwebegleichgewicht
  0,107 kg · 9,81 = 1,0497 N) passt zur Vollscheibe (0,1075 kg bei
  2700 kg/m³), nicht zu einer gelochten Platte.
- Ein Loch ändert die **Topologie**. Die Platte wird zum Volltorus, der Raum
  um sie ist nicht mehr einfach zusammenhängend, und der Löser findet einen
  zusätzlichen Kohomologie-Generator (siehe Kommentar in `src/fem.cu` bei der
  Generatorensuche). Das ist ein anderes Problem, nicht dasselbe mit kleinem
  Fehler.
- Ein Loch ändert die **Geometrie** des Benchmarks.
- Ergebnisse wären nicht mehr mit dem Standardnetz vergleichbar, das eine
  Vollscheibe ist (Mittelzelle über der Achse).
- Die älteren BDF-Netze für FEM-direkt (`team28_simple_air*.bdf`) haben eine
  Platte mit r = 10,7…65 mm und weichen damit vom Benchmark ab; siehe den
  Korrekturhinweis in [fem-direct-plan.md](fem-direct-plan.md). Phase 3
  ersetzt sie durch Netze mit Vollscheibe.

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
- **Kein Verschmelzen von Knoten** (`removeDuplicateNodes`): es würde Knoten
  verschiedener Körper vereinen, wenn sie zufällig zusammenfallen, und Platte
  und Spule leitend verbinden. Der Geo-Kern liefert die Netze ohnehin
  konform; doppelte Knoten meldet die Prüfung.
- Die Platte liegt immer echt oberhalb der Spulen (Unterkante > 0); berührende
  Flächen zweier Körper sind für die BEM unbrauchbar.

## Phasen

| Phase | Inhalt | Stand |
|---|---|---|
| 0 | Klärung: 360°-Ring, Hexaederkern, BEM und Dreiecke, Lesbarkeit, Topologie | **abgeschlossen**, siehe [gmsh-phase0-report.md](gmsh-phase0-report.md) |
| 1 | `tools/meshcheck.py`: gemeinsame Prüfung für alle Importeure; Tests mit absichtlich kaputten Netzen | **abgeschlossen**, siehe „Ergebnis Phase 1“ unten |
| 2 | `tools/gmsh/team28_hex.py`: Generator mit Parametern (Teilungen in r, z, φ; Kerngröße; Plattenlage), aus dem Prototyp | **abgeschlossen**, siehe [tools/gmsh/README.md](../../tools/gmsh/README.md) |
| 3 | `tools/gmsh/team28_fem_air.py`: Netz mit Luft für den FEM-direkt-Pfad (`component_id` 0) | **abgeschlossen**, siehe „Ergebnis Phase 3“ unten |
| 4 | `tools/gmsh/import_msh.py`: eigene `.msh`/`.geo`-Dateien, Abbildung Gruppenname → `component_id` | offen |
| 5 | Dokumentation (`docs/meshing.md`), CI-Job (pip + GL-Bibliothek), CTest-Marke `gmsh`, CHANGELOG | **abgeschlossen** bis auf den CHANGELOG-Eintrag: der folgt beim Zusammenführen, um einen Konflikt mit offenen PRs zu vermeiden |
| 6 | optional: Netzserie auf dem Server rechnen — hängt die Stagnation des Lösers vom Netz ab? | offen |

## Ergebnis Phase 1

Die Prüfungen in `tools/import_bdf.py` werden **nicht** durch `meshcheck.py`
ersetzt: Sie laufen vor dem Schreiben auf den BDF-Daten, können Elemente
reparieren (`--fix-orientation`), werten die Jacobi-Determinante an
Gauß-Punkten mit den Formfunktionen des Lösers aus und kommen ohne numpy aus.
Stattdessen prüft `import_bdf.py` die **geschriebene** Datei zusätzlich mit
`meshcheck.py`. Diese zweite, unabhängige Prüfung deckt den Schreibschritt
selbst ab (Knotenreihenfolge, Nummerierung); ein absichtlich eingebauter
Fehler im Schreiben wird nur von ihr gefunden. Ohne numpy wird sie mit Hinweis
übersprungen.

`meshcheck.py` prüft jetzt auch Prismen (Orientierung in der Konvention des
Lösers, `include/NedelecPrism.cuh`, Volumen). Beide Prüfungen stimmen überein.

**Folge für Phase 3: Gmsh schreibt Prismen umgekehrt orientiert.** Alle
Prismen aus Gmsh-Versuchen sind in der Konvention des Lösers negativ
(240 von 240, unabhängig bestätigt mit der Gauß-Punkt-Rechnung aus
`import_bdf.py`). Die Knotenfolge (0, 2, 1, 3, 5, 4) macht sie positiv.
Tetraeder aus Gmsh sind richtig orientiert. Der Generator für Phase 3 muss
Prismen deshalb vor dem Schreiben umdrehen; `meshcheck.py` würde ein
vergessenes Umdrehen melden.

## Ergebnis Phase 3

`tools/gmsh/team28_fem_air.py` erzeugt ein reines Prismennetz mit Luftzylinder,
wie AP2 in [fem-direct-plan.md](fem-direct-plan.md) es verlangt: Dreiecksnetz in
der Ebene mit allen Radien als Kanten, schichtweise senkrecht extrudiert
(Prismen affin, Defekt ~1e-15), Schichtgrenzen an Spulen, Spalt und Platte,
Platte mit mehreren Lagen, Luftschichten zum Rand hin gröber. Der
Abschneidefaktor ist direkt einstellbar und folgt der Definition des Programms.

Geprüft:

- Je Netz: nur Prismen, konform, positiv orientiert, affin, Luft umschließt
  alles, Volumina **exakt** gegen das Vieleck aus den Knoten auf jedem Kreis
  (bis auf Rundung, ~1e-16), `check_map.py`, Abschneidefaktor. Zehn
  Parameterkombinationen und zwei unzulässige Eingaben. Eine zuerst gewählte
  Toleranz von 1 % gegen die Kreisformel war falsch: bei grobem Netz fehlen
  einem Vieleck aus 17 Segmenten rund 2 % Fläche. Gegenprobe: eine absichtlich
  falsche Zuordnung des Plattenrands wird erkannt (Volumen −29 %, `check_map`).
- Im Programm (grobes Netz, 24 684 Prismen): der Leser akzeptiert es,
  **0 Kohomologie-Generatoren** wie für eine Vollscheibe mit Luft erwartet.
  Die Vorprüfung des FEM-direkt-Pfads (`preflight_fem_direct`, eigens
  aufgerufen, weil der Lauf ohne cuDSS vorher abbricht) meldet keine
  Beanstandung und Abschneidefaktor 3 / 3 / 3 wie eingestellt.
- Nicht geprüft: ein vollständiger FEM-direkt-Lauf mit Kraft — dafür braucht es
  cuDSS bzw. die GPU (Phase 6 auf dem Server).

Gmsh liefert die Prismen dieser Extrusion übrigens bereits richtig orientiert
(0 von 140 616 gedreht); die Drehung im Generator bleibt als Absicherung, sie
richtet jedes Prisma einzeln aus.

## Abnahme je Netz (ab Phase 2 automatisch)

- `tools/check_map.py` besteht (Zuordnung, Maße, Vollscheibe).
- Keine doppelten Knoten, kein Knoten gehört zu zwei Körpern, jede Fläche
  gehört zu höchstens zwei Zellen.
- Positive Jacobi-Determinante an allen Ecken.
- Volumen je Körper gleich dem exakten N-Eck (N Segmente am Umfang) bis auf
  Rundung.
- Der Löser findet genau **2** Kohomologie-Generatoren (je Ringspule einer;
  die Platte trägt keinen bei).

## Risiken

- Qualität des Hexaederkerns hängt von der Kerngröße ab (gemessen: kleinste
  skalierte Jacobi-Determinante 0,555 bei 0,1 bis 0,814 bei 0,5; Vorgabe 0,4
  mit 0,802).
- Gmsh braucht auf Servern ohne Grafik trotzdem die GL-Laufzeitbibliotheken.
- Volumina ohne Physical Group fehlen im Ergebnis ohne Fehlermeldung; die
  Prüfung muss das abfangen (vor allem die Luft in Phase 3).
