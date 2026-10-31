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
- Test `test_piola_production` (Marke `host`): ruft die produktiven
  Feldkörper (VtkExporter-Zellmittel, Maxwell-Kraft des transienten Pfades,
  `harmonic_lorentz_body`, `harmonic_maxwell_body`) auf einem gedrehten und
  gescherten Hexaeder, Tetraeder und Prisma auf und vergleicht mit einem
  homogenen B, Kraft und Moment je Fläche (Fehler 2e-16 bis 1,4e-15). Der
  Kontrollfall mit transponierter Piola-Formel fällt mit 100 % Fehler in B
  durch.

### Geändert

- `tools/check_map.py` lehnt ein Plattennetz mit Loch ab: **TEAM 28 hat kein
  Loch in der Platte** (ein Loch steht im Paper nur als Ausblick). Gewicht im
  Schwebegleichgewicht 0,107 kg · 9,81 = 1,0497 N.
- FEM-direkt gibt F_z mit sechs Stellen aus.
- Die Meldung der Kohomologie-Generatoren nennt die Erwartung für beide Pfade
  (BEM-FEM 2, FEM-direkt mit Luft 0).
- `test_alpha_poisson` erweitert.
- Bericht `docs/reports/harmonic-fem.tex` um die Ergebnisse des AMS-Modus ergänzt.
- Die Piola-Transformationen stehen nur noch in `piola_covariant` und
  `piola_contravariant` (`NedelecDispatch.cuh`); Assemblierung,
  `verify_bz_nedelec_body` und `--fem-direct`-Diagnose (`fem.cu`),
  Maxwell-Kraft (`TransientSolver`), `VtkExporter`, `HarmonicLorentz.cuh`,
  `HarmonicMaxwell.cuh` und `NonlinearSolver.cu` rufen sie auf, statt die
  Formel abzuschreiben. Gleiche Rechnung, gleiche Reihenfolge der Operationen.
- Die Rümpfe von `compute_maxwell_loads_kernel` und
  `compute_cell_centered_fields_kernel` stehen unverändert in
  `include/MaxwellLoads.cuh` und `include/CellCenteredFields.cuh`, damit ein
  Host-Test sie ohne den restlichen Löser aufrufen kann.
- `test_ams_preconditioner`, `test_preconditioner_end_to_end`,
  `test_discrete_gradient` und `test_curl_gradient_nullspace` assemblieren
  mit der produktiven Konvention (vorher: W spaltenweise über invJ, rot W
  zeilenweise über J, auf gestörten Netzen also ein etwas anderer Operator).
  Schwellen unverändert; die Messwerte verschieben sich nur leicht: Jacobi-CG
  298–693 → 296–697 Iterationen, AMS unverändert 23–28; GMRES ohne
  Vorkonditionierer 161 Iterationen wie bisher, ILU(0) und AMS ebenfalls
  unverändert. Hilfetext zu `--ams` entsprechend nachgezogen.

### Behoben

- Windows mit cuDSS: `cudss64_0.dll` wird nach dem Bauen neben `fembem.exe`
  und `test_sparse_direct.exe` kopiert. Ohne die DLL im `PATH` startete der
  Test nicht und lief unter CTest ohne Ausgabe in seine Zeitgrenze (120 s).
- Bau unter Linux mit CUDA 13: neue Signatur von `cudaMemPrefetchAsync`.
- `test_block_system` unter Compilern mit FMA (z. B. GCC unter RHEL 10).
- CMake lehnt cuDSS älter als 0.8 ab, statt mitten in der Übersetzung zu
  scheitern.
- Kommentare, Hilfetext und Dokumentation, die eine Bohrung in der Platte
  oder drei Volltori annahmen (richtig: Vollscheibe, zwei Volltori).
- `test_ams_preconditioner` übergibt die Alpha-Poisson-Koeffizienten und läuft
  als Host-Test mit.
- `harmonic_maxwell_body` (Gegenprobe Maxwell-Spannungstensor im
  harmonischen Pfad) gewichtete jede Viereckfläche (Hexaeder, Seitenflächen
  des Prismas) mit einem Viertel: der Faktor 4 der Flächenabbildung auf
  [-1, 1]² fehlte. Dreiecksflächen waren richtig; die dokumentierten Zahlen
  der Gegenprobe stammen von reinen Tetraedernetzen.
- Kommentare und Dokumentation zur Piola-Transformation: `compute_jacobian`
  legt J transponiert ab (J[i][j] = ∂x_j/∂ξ_i), richtig ist W = invJ·Ŵ und
  rot W = Jᵀ·rot Ŵ / det J. Berichtigt in `fem.cu` („invJᵀ·W_ref“),
  `HarmonicLorentz.cuh` (TransientSolver transformiere zeilenweise – längst
  korrigiert), `docs/design/bem-block-preconditioning.md` (offener Punkt als
  geklärt markiert) und `docs/reports/harmonic-fem.tex` (verstümmelte Formel;
  die Verwechslung ist nicht „auf affinen Elementen folgenlos“, sondern nur
  bei symmetrischem J).

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
