# Änderungen

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unveröffentlicht]

### Hinzugefügt

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

- `test_alpha_poisson` erweitert.
- Bericht `docs/reports/harmonic-fem.tex` um die Ergebnisse des AMS-Modus ergänzt.

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
  28 davon mit der Marke `host` (ohne GPU lauffähig), gebündelt im CMake-Ziel
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
