# Standardnetz auf `team28_hex.py --scale 1.0` umgestellt

Stand: 2026-11-03. Arbeitsauftrag: Standardnetz von `team28.vtk` auf ein
eingechecktes Gmsh-Hexaedernetz umstellen. Löser unverändert.

## Ergebnis in Kürze

- Neues Standardnetz `data/team28/team28_hex.vtk`, erzeugt mit
  `python3 tools/gmsh/team28_hex.py --scale 1.0 -o data/team28/team28_hex.vtk`
  (Gmsh 4.15.2). `team28.vtk` bleibt byteweise unverändert.
- Der Referenzpfad (`--reference --harmonic --fixed-plate --plate-z=11.3`)
  liefert auf dem neuen Standard F_z = 0,966492 N, 10 FGMRES-Iterationen,
  450 innere Iterationen: gleich A_s1 der Netzserie.
- **Nicht gemessen** (auf Wunsch ausgelassen, zu zeitaufwändig): Kurztest des
  transienten BEM-FEM-Pfads (6b), harmonischer BEM-FEM-Pfad (6c), die langen
  GPU-Tests. Siehe „Offene Punkte“.

## Netz (Aufgabe 1)

| Prüfung | Ergebnis |
|---|---|
| `meshcheck.py --require-hex --separate-bodies` | bestanden: 4944 Hexaeder, 6916 Knoten, keine doppelten/verwaisten Knoten, Körper getrennt, Konformität ok, Jacobi > 0 (kleinste 8,45), Volumina auf 1e-16 gleich dem 48-Eck-Soll |
| `check_map.py` | bestanden: Außenspule 1536, Innenspule 1536, Platte 1872 Zellen; Platte z = 3,8 … 6,8 mm |
| Kohomologie-Generatoren | 2 (`test_cohomology_gauge`, eine je Spule, orthogonal) |

## Vorher/Nachher

| Größe | `team28.vtk` (vorher) | `team28_hex.vtk` (nachher) |
|---|---|---|
| Hexaeder | 4805 | 4944 |
| Knoten | 7013 | 6916 |
| Zellen je Bauteil (Außen/Innen/Platte) | 1200 / 1400 / 2205 | 1536 / 1536 / 1872 |
| BEM-Randflächen | 4136 | 3696 (Netzprüfung) |
| F_z Referenzpfad, 11,3 mm | 0,94356 N (Netzserie D_team28) | 0,966492 N (diese Messung) |
| FGMRES / innere Iterationen | 11 / 518 (Netzserie) | 10 / 450 (diese Messung) |
| Abweichung zum Ringmodell (1,0068 N) | −6,3 % | −4,0 % |
| Transiente BEM-FEM (6b) | nicht gemessen | nicht gemessen |
| Harmonischer BEM-FEM (6c) | nicht gemessen | nicht gemessen |

Die Werte der Spalte „vorher“ stammen aus
`reference-harmonic-mesh-series.md`, nicht aus einem neuen Lauf. Die 3696 Randflächen des neuen Netzes stammen aus der Netzprüfung
(„Rand 3696“), die 4136 des alten aus dem Cluster-Baum-Log. Zeit des Referenzlaufs hier: 11,7 s gesamt (BEM 4,8 s,
FGMRES 5,7 s) auf der RTX 6000 Ada, die GPU war dabei mit einem fremden
Python-Prozess geteilt.

Unabhängiger Befund: Die Prüfsummen von `test_determinism_fingerprint` auf
`team28.vtk` stimmen mit der Tabelle in `tests/README.md` überein (H1–H6). Auf
dem neuen Netz sind sie erwartungsgemäß andere (und werden nicht in die
README übernommen).

## Testentscheidungen (Aufgabe 3)

| Test | Entscheidung | Begründung |
|---|---|---|
| `test_determinism_fingerprint` | bleibt auf `team28.vtk`, Pfad ausdrücklich in CMake | Feste Erwartung: die Prüfsummen H1–H6 in `tests/README.md` gehören zu `team28.vtk`. Läuft dort grün (alle sieben gleich). |
| `test_bem_index_mapping` | neuer Standard, Pfad in CMake | Strukturprüfung (Permutation, Kantenabbildung, Trefferzahl 2); keine festen Zahlen. Grün auf beiden Netzen, Erwartungen unverändert. |
| `test_tree_cotree_gauge` | neuer Standard, Pfad in CMake | Vergleicht gegen aus dem Netz berechnete Größen. Grün auf beiden Netzen, Erwartungen unverändert. |
| `test_cohomology_gauge` | neuer Standard, Pfad in CMake | Erwartet 2 Generatoren; gilt auf beiden Netzen. Grün auf beiden, Erwartungen unverändert. |

Bei den drei umgestellten Tests wurde zusätzlich die Vorgabe im Quelltext von
`"team28.vtk"` auf `"team28_hex.vtk"` geändert, damit ein Start ohne Argument
dasselbe Netz nimmt wie CTest.

## Literal-Suche (Aufgabe 4)

Gesucht: 4805, 18690, 5100, 10315, 1200, 1400, 2205 in `src/`, `include/`,
`tests/`. 2205 hat keinen Treffer.

| Fundstelle | Wert | Einordnung | Maßnahme |
|---|---|---|---|
| `src/Backend.cu:306` | 4805 | **Literal im Hilfetext** (Verfeinerung Stufe 1: 4805 → 38440 Elemente) | auf das neue Netz umgestellt: Stufe 1 4944 → 39552 Elemente, 6916 → 47191 Knoten, Stufe 2 → 316416 Elemente. Die Knotenzahl folgt aus der Euler-Formel (Kanten 18651, Flächen 16680), nicht aus einem Lauf. |
| `include/MeshRefinement.cuh:44` | 4805 | Kommentar (Größen je Stufe „des TEAM-28-Netzes“) | unverändert, beschreibt das alte Netz |
| `src/NonlinearSolver.cu:147`, `include/MaxwellLoads.cuh:128`, `src/fem.cu:124` | 1200 | Kommentar: „Hexaeder 1200 der Innenspule“, ein Zellindex des alten Netzes | unverändert |
| `src/fem.cu:1681-1685` | 10315, 5100, 4805, 18690 | Kommentar, der die früher fest eincodierten Vergleichswerte dokumentiert; der Code rechnet sie jetzt aus dem Netz | unverändert |
| `src/fem.cu:2157`, `src/fem.cu:2705` | 18690 | Kommentar (Dimension, Stagnationsbeispiel) | unverändert |
| `tests/model_block_triangular_pc.py:49` | 1200 | Literal in einem Python-Modell (`n_fem, n_bem = 1200, 300`), unabhängig vom Netz | unverändert |

**Kein netzabhängiges Literal in der Logik gefunden.** Beim Lauf auf dem alten
Netz erscheinen im Log „Hexaeder 2600 / 1200 / 0“; das ist
`hexes.front()` (`src/fem.cu:987`), also berechnet, nicht eincodiert. Der
RHS-Diagnoseblock gibt „10315 / 18690“ als berechnete Zahlen aus. Der Lauf auf
dem neuen Netz wurde nicht bis dorthin ausgewertet.

## Prüfungen (Aufgabe 5)

- `tools_standard_mesh` (meshcheck) und neu `tools_legacy_mesh`,
  `tools_standard_mesh_map`, `tools_legacy_mesh_map` (check_map) für beide
  Dateien: `ctest -L tools` 7/7 grün.
- CI: `check_map` und `meshcheck` für beide Dateien. Kein Byte-Vergleich mit
  der Gmsh-Ausgabe.
- Beim Umbau fiel in `.github/workflows/ci.yml` eine schon vorher falsch
  eingerückte Zeile auf: der `meshcheck`-Aufruf stand als Fortsetzung der
  `run:`-Zeile des Ringmodell-Schritts und lief damit nicht als eigener Befehl.
  Ich habe daraus einen eigenen Schritt gemacht. Das ist eine Korrektur über
  den Auftrag hinaus.

## Testlauf

- `ctest -L host`: 35/35 grün (mit `LD_LIBRARY_PATH` auf die HPC-SDK-CUDA-
  Bibliotheken; ohne sie finden die Tests `libcudart.so.12` nicht, das ist ein
  Umgebungs- und kein Netzproblem).
- `ctest -L tools`: 7/7 grün.
- Einzeln gelaufen und grün: die vier Netztests, je auf beiden Netzen
  (Fingerprint nur auf `team28.vtk`).
- Die übrigen GPU-Tests wurden **nicht** gelaufen (zeitaufwändig, auf
  Anweisung ausgelassen).

## Offene Punkte

1. **6b/6c nicht gemessen.** Ein erster Versuch der transienten Messung auf
   `team28.vtk` lief nach Beginn mehr als 25 Minuten ohne ersten Schritt
   (GMRES bei relativem Residuum ≈ 0,15 nach 240 Iterationen). Das ist ein
   Befund zum alten Netz, aber kein Vergleich; er wurde abgebrochen. Beim
   neuen Netz fehlt jede Messung. Die GPU war mit einem fremden Prozess
   geteilt, die Zeiten sind nicht belastbar.
2. **Hilfetext `--refine`:** umgestellt; die Knotenzahl 47191 ist gerechnet, nicht gemessen (`--refine=1` nicht gelaufen).
3. `tools/gmsh/team28_hex.py` nennt im Kopf `team28.vtk` „das Standardnetz“;
   jetzt veraltet, nicht geändert.
4. Die Randflächenzahlen stammen aus Netzprüfung und Log, nicht aus einem gemeinsamen Lauf.
5. `tools/reference_series.py` führt `team28.vtk` weiter als Lauf D; passt, da
   das Legacy-Netz erhalten bleibt.
