# Tests

Jeder Test ist ein eigenständiges Programm, das mit Exit-Code 0 endet, wenn
alles bestanden ist. CMake registriert sie bei CTest (`CMakeLists.txt`, Option
`FEMBEM_BUILD_TESTS`, standardmäßig an).

```sh
cmake -B build
cmake --build build --target host_tests        # nur die GPU-freien Tests
ctest --test-dir build -L host --output-on-failure

cmake --build build                            # alle Testprogramme
ctest --test-dir build --output-on-failure
```

## Marken

| Marke | Bedeutung |
|---|---|
| `host` | Läuft ohne GPU und ohne Dateien von der Platte; baut Netze und Matrizen selbst. Geeignet für CI. |
| `tools` | Python-Prüfmodule für Netze und das Ringmodell als Kraftreferenz (`tools/tests/`), brauchen numpy; nur registriert, wenn Python mit numpy gefunden wird. |
| `gmsh` | Netzgeneratoren `tools/gmsh/team28_hex.py` und `team28_fem_air.py` (prüfen jedes erzeugte Netz) und der Import `tools/gmsh/import_msh.py` (`tools/tests/test_import_msh.py`); nur registriert, wenn das Python-Paket gmsh gefunden wird (`-DPython3_EXECUTABLE=...`). |
| (keine) | Braucht teils eine GPU, teils das Netz `team28.vtk`. Das Netz liegt nach dem Konfigurieren im Build-Verzeichnis, CTest startet die Tests dort. |

`host_tests` ist ein CMake-Ziel, das genau die Tests mit der Marke `host`
baut. Das CUDA-Toolkit wird zum Übersetzen auch für diese gebraucht.

## Was geprüft wird

Die Tests prüfen jeweils eine Eigenschaft, die ein früherer Fehler verletzt
hat, oder eine Invariante, die der Rest des Codes voraussetzt. Der Kopfkommentar
jedes Tests nennt den Anlass.

| Bereich | Tests |
|---|---|
| Elemente, Netz, Orientierung | `test_nedelec_orientation`, `test_edge_sign_convention`, `test_curl_gradient_nullspace`, `test_nedelec_tet`, `test_nedelec_prism`, `test_element_dispatch`, `test_piola_production`, `test_mixed_topology`, `test_mesh_refinement` |
| Eichung | `test_tree_cotree_gauge`, `test_cohomology_gauge`, `test_gauge_on_tets`, `test_discrete_gradient` |
| BEM und Kopplung | `test_bem_index_mapping`, `test_coupling_sign_convention`, `test_gbem_symmetry`, `test_adjacent_bem`, `test_singular_quadrature`, `test_sauter_schwab_coverage` |
| Vorkonditionierer | `test_amg`, `test_amg_convergence`, `test_amg_anisotropy`, `test_ams_preconditioner`, `test_ams_scalar_masking`, `test_alpha_poisson`, `test_block_triangular_consistency`, `test_block_preconditioner_study`, `test_preconditioner_end_to_end`, `test_ritz_spectrum`, `test_asymmetry_estimator` |
| Löser und Systeme | `test_dense_direct`, `test_sparse_direct`, `test_block_system`, `test_dirichlet_bc`, `test_fem_direct_preflight`, `test_patch_assembly`, `test_source_projection`, `test_reference_solver` |
| Harmonischer Pfad | `test_harmonic_phase_average`, `test_harmonic_dirichlet_split` |
| Numerische Robustheit | `test_dgemv_beta_zero`, `test_norm_nan_propagation`, `test_solver_nonfinite_abort` |
| Mechanik | `test_free_motion_gate` (kraftgetriebene Objekte müssen `is_moving()` melden) |
| Backend, Reproduzierbarkeit | `test_cpu_backend`, `test_determinism_fingerprint` |
| Feld | `test_biot_savart` |
| Modellrechnung (nicht bei CTest) | `model_block_triangular_pc.py`: woher die Vorgaben für `--pc-inner-iters` und `--pc-inner-tol` stammen; braucht nur numpy und scipy |

## Einen Test hinzufügen

1. Programm unter `tests/test_<name>.cpp` oder `.cu` anlegen; Kopfkommentar mit
   dem Anlass und der geprüften Eigenschaft; Exit-Code ungleich 0 bei Fehler.
2. In `CMakeLists.txt` ein `add_executable` mit genau den Quelldateien, die der
   Test braucht, und ein `add_test` anlegen. Läuft der Test ohne GPU und ohne
   Dateien, `LABELS "host"` setzen und das Ziel zu `host_tests` hinzufügen.
3. Der Test sollte einen Fehler tatsächlich finden können. Mehrere der
   vorhandenen Tests enthalten deshalb einen Kontrollfall mit dem
   fehlerhaften Original, der durchfallen muss (zu erkennen an der Ausgabe
   `DURCHGEFALLEN`; der Test selbst endet trotzdem mit 0).

## Determinismus-Fingerprint

`test_determinism_fingerprint` bildet bitweise FNV-1a-Prüfsummen über jede
Stufe der Kette Netz → Topologie → Schwerpunkte → Sortierung → Cluster-Baum.
Er dient dazu, Plattformabweichungen zu lokalisieren: zwei Rechner vergleichen
ihre Ausgaben, die erste abweichende Zeile benennt die Stufe.

| Stufe | Inhalt |
|---|---|
| H1 | Knotenkoordinaten aus `VtkReader` |
| H2 | Randflächen-Konnektivität (ganzzahlig) |
| H2b | globale Kantenliste |
| H3 | Quad-Schwerpunkte, nachgerechnet wie in `ClusterTree::build_tree` |
| H4 | Permutation nach dem Wurzel-Sortierschritt |
| H5 | Blattpartition des fertigen Baums |
| H6 | `boundary_edges_to_fem_edges` (ganzzahlig) |

Referenzwerte für das Standardnetz `data/team28/team28.vtk`, gemessen mit
g++ 12.4 (libstdc++), CUDA 12.0, x86-64, Release (`-O3`), CPU-Modus; ein
Wiederholungslauf liefert dieselben Werte:

    FEMBEM_BACKEND=cpu ./test_determinism_fingerprint

| Stufe | Prüfsumme |
|---|---|
| H1 | `74cc7b024f66a565` |
| H2 | `c4229056f7b5ce43` |
| H2b | `ae7e85b1af6ee8de` |
| H3 | `5ea76ec488be9ad1` |
| H4 | `37bd2b7519bbbe1f` |
| H5 | `78052475e70da64f` |
| H6 | `b8d75e359f226767` |

Abweichungen auf einer anderen Plattform sind genau das, was der Test finden
soll. Ändert sich Netzeinlesen, Topologie oder Cluster-Baum absichtlich,
ändern sich die Werte mit; dann hier nachtragen.

Lesart: weicht H1 ab, liegt es an der Text-nach-double-Konvertierung im
`VtkReader`. Weicht H3 bei gleichem H1 ab, an der Fließkomma-Auswertung der
Schwerpunkte. Weicht H5 bei gleichem H4 ab, an der Rekursion selbst
(Achsenwahl, Bounding Box, Median-Teilung).

Der Test gibt zusätzlich Rohwerte einzelner Schwerpunkte mit `%.17g` aus und
die Zahl exakter Gleichstände je Achse. Beim Standardnetz sind es für x, y, z
1576, 1510 und 56 verschiedene Werte bei 4136 Quads (2560, 2626 und 4080
Gleichstands-Paare), also viele Gleichstände, an denen eine instabile
Sortierung plattformabhängig würde.
