# Änderungen

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unveröffentlicht]

### Hinzugefügt

- **Netze mit Gmsh** (`tools/gmsh/`, Anleitung in `docs/meshing.md`):
  - `team28_hex.py`: TEAM-28-Netz aus reinen Hexaedern für den BEM-FEM-Pfad;
    Spulen als konformer 360°-Ring, Platte als Vollscheibe mit Hexaederkern.
  - `team28_fem_air.py`: Prismennetz mit Luftgebiet für den FEM-direkt-Pfad,
    schichtweise senkrecht extrudiert (Prismen affin), Abschneidefaktor wie im
    Programm definiert. Vorgabe radial 5 / axial 10 aus einer Messreihe am
    Schwebegleichgewicht: dort liefert der Pfad F_z = 1,027 N gegen den
    Sollwert 1,0497 N (`docs/design/gmsh-phase6-report.md`).
  - Beide prüfen jedes erzeugte Netz selbst und enden bei einem Fehler mit 1.
- **Netzprüfung** `tools/meshcheck.py`: Zelltypen, doppelte und verwaiste
  Knoten, getrennte Körper, Konformität, Orientierung (Hexaeder, Prismen,
  Tetraeder), Prismen-Affinität, Luft umschließt alles, Volumina. Mit Tests an
  absichtlich kaputten Netzen (`tools/tests/`).
- `tools/import_bdf.py` prüft die geschriebene Datei zusätzlich mit
  `meshcheck.py`.
- CTest-Marken `tools` und `gmsh` (nur registriert, wenn Python, numpy und
  gmsh vorhanden sind); CI-Job „Netze mit Gmsh“; `tools/requirements-gmsh.txt`.
- CI: apt-Installation mit Zeitgrenze, Wiederholung und Diagnose beim Hänger
  (`.github/scripts/apt-install.sh`), Zeitgrenze je Job.
- `--fem-direct-nullspace=eps|gauge|ams` (Kurzform `--fem-direct-ams`): Im Modus
  `ams` wird der Curl-Curl-Nullraum gar nicht regularisiert; das singuläre, aber
  konsistente System wird iterativ gelöst, und der vektorielle AMS-Hilfsraum
  trägt den Nullraum. Der Modus erzwingt den iterativen Löser, weil cuDSS eine
  singuläre Matrix nicht meldet.
- Alpha-Poisson-Operator auf Tetraedern und Prismen als Grundlage des
  vektoriellen AMS-Hilfsraums; Knotenformfunktionen für alle Elementtypen
  (`include/NodalDispatch.cuh`).
- Test `test_harmonic_dirichlet_split`: Dirichlet-Rand im n-Raum gegen die
  2n-Blockform, mit Gegenprobe des eingebauten Fehlers; prüft außerdem die
  Knotengradienten je Elementtyp.

### Geändert

- Bericht `docs/design/gmsh-phase7-report.md`: FEM-direkt mit getrennt
  verfeinerten Richtungen. Die axiale Auflösung der Spulen bestimmt den
  Netzfehler (Ordnung 2; die Vorgabe `--coil-nz 6` überschätzt F_z um 2,2 %).
  Hochgerechnet F_z ≈ 1,015–1,018 N gegen den Sollwert 1,0497 N.
- `tools/gmsh/team28_fem_air.py`: Vorgabe `--coil-nz` von 6 auf 12 angehoben
  (Fehler in F_z 2,2 % → 0,55 %). Die Läufe der Phasen 6 und 7 nachrechnen
  mit `--coil-nz 6`.
- `tools/check_map.py` lehnt ein Plattennetz mit Loch ab: **TEAM 28 hat kein
  Loch in der Platte** (ein Loch steht im Paper nur als Ausblick). Gewicht im
  Schwebegleichgewicht 0,107 kg · 9,81 = 1,0497 N.
- FEM-direkt gibt F_z mit sechs Stellen aus.
- Die Meldung der Kohomologie-Generatoren nennt die Erwartung für beide Pfade
  (BEM-FEM 2, FEM-direkt mit Luft 0).
- `test_alpha_poisson` erweitert.
- Bericht `docs/reports/harmonic-fem.tex` um die Ergebnisse des AMS-Modus ergänzt.

### Behoben

- Bau unter Linux mit CUDA 13: neue Signatur von `cudaMemPrefetchAsync`.
- `test_block_system` unter Compilern mit FMA (z. B. GCC unter RHEL 10).
- CMake lehnt cuDSS älter als 0.8 ab, statt mitten in der Übersetzung zu
  scheitern.
- Kommentare, Hilfetext und Dokumentation, die eine Bohrung in der Platte
  oder drei Volltori annahmen (richtig: Vollscheibe, zwei Volltori).
- `test_ams_preconditioner` übergibt die Alpha-Poisson-Koeffizienten und läuft
  als Host-Test mit.

## [0.1.0] – 2026-10-27

Erste öffentliche Fassung.

### Enthalten

- Transienter Wirbelstrom-Löser mit BEM-FEM-Kopplung und Nédélec-Kantenelementen
  (Hexaeder, Tetraeder, Prismen) für TEAM Workshop Problem 28.
- GPU-Betrieb (CUDA) und reiner CPU-Betrieb (`--cpu`, optional OpenMP).
- Baum-Kobaum-Eichung mit Kohomologie-Erweiterung.
- ILU(0)- und AMS-Vorkonditionierer auf Aggregations-AMG, blockdiagonal und
  blockdreieckig.
- Harmonischer Löser für den eingeschwungenen Zustand und ein FEM-direkt-Pfad
  mit dünn besetztem Direktlöser (optional, cuDSS).
- Unabhängiger Referenzlöser (`--reference`) und analytische Biot-Savart-Referenz.
- Diagnosewerkzeuge: Konditionszahl, Ritz-Spektrum, Vorkonditionierer-Studie.
- 44 Testprogramme (plus eine Python-Modellrechnung), bei CTest registriert;
  29 davon mit der Marke `host` (ohne GPU lauffähig), gebündelt im CMake-Ziel
  `host_tests`.
- Dokumentation: Architektur, Theorie, Messberichte, Entwurfsnotizen.
- Python-Werkzeuge: BDF-Import, Ersatznetze für die Netzstudie, Prüfung der
  Spulenzuordnung, Plots.

### Bekannte Einschränkungen

- Die iterative Lösung des gekoppelten Systems konvergiert langsam
  (Konditionszahl etwa 10⁶); siehe
  [docs/reports/convergence-findings.tex](docs/reports/convergence-findings.tex).
- Die Ergebnisse sind nicht gegen die Benchmark-Referenz validiert.
- Der FEM-direkt-Pfad braucht cuDSS und ist experimentell.
