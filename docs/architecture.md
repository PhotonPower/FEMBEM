# Architektur

Überblick über die Module. Die Header in `include/` tragen jeweils einen
Kopfkommentar, der Zweck, Annahmen und Fallstricke des Moduls beschreibt und
die maßgebliche Quelle ist; diese Seite zeigt nur, wo was liegt.

## Ablauf

`src/main.cpp` ruft `mainloop()` in `src/fem.cu` auf. Dort geschieht, in
dieser Reihenfolge:

1. **Optionen lesen** (`Backend`): GPU oder CPU, Löser, Vorkonditionierer,
   Diagnoseschalter. `--help` listet alle.
2. **Netz einlesen** (`VtkReader`), ggf. verfeinern (`MeshRefinement`,
   `--refine=N`) und Materialien über die `component_id` der Zellen zuordnen.
3. **Topologie aufbauen** (`TopologyExtractor`): Kanten, Randflächen,
   Zuordnung der BEM-Freiheitsgrade zu FEM-Kanten (`BemCoupling`).
4. **Eichung vorbereiten** (`TreeCotreeGauge`, `CohomologyGauge`).
5. **BEM-Blöcke aufbauen** (`ClusterTree`, `AcaPlus`, Sauter-Schwab-Quadratur).
6. **Zeitschleife** (`TransientSolver`, `CoupledSolver`) oder eingeschwungener
   Zustand (`HarmonicSolver`); danach Kraft und Bewegung der Platte
   (`PhysicalObject`, `Kinematics`) und Ausgabe (`VtkExporter`).

## Module

| Bereich | Dateien | Aufgabe |
|---|---|---|
| **Eingabe, Netz** | `VtkReader`, `Mesh`, `MeshRefinement`, `TopologyExtractor`, `ClusterTree` | Netz lesen, verfeinern, Kanten und Ränder bestimmen, Cluster-Baum für die BEM-Blockstruktur |
| **Elemente** | `NedelecHexahedron`, `NedelecTetrahedron`, `NedelecPrism`, `NedelecDispatch` | Kantenbasisfunktionen erster Art je Elementtyp; `NedelecDispatch` wertet typabhängig an einer Stelle aus |
| **Material, Mechanik** | `MaterialTypes`, `PhysicalObject`, `Kinematics`, `NonlinearSolver` | Materialregionen, Starrkörperbewegung der Platte, B-H-Kurven |
| **FEM** | `fem.cu`, `fem.cuh` | Assemblierung, Programmablauf |
| **BEM** | `BemCoupling`, `BemIntegration`, `AcaPlus`, `SauterSchwab`, `SingularPair`, `SingularQuadPair`, `AdjacentIntegration`, `CouplingKernels` | Randelemente, Kompression, singuläre und benachbarte Integrale, Kopplungsmatrix T |
| **Eichung** | `TreeCotreeGauge`, `CohomologyGauge`, `DiscreteGradient` | Nullraum des Curl-Curl-Operators beseitigen |
| **Löser** | `CoupledSolver`, `TransientSolver` | gekoppeltes System (GMRES), impliziter Euler, Maxwell-Spannungstensor |
| **Harmonisch** | `HarmonicSolver`, `HarmonicBlockSystem`, `HarmonicLorentz`, `HarmonicMaxwell`, `HarmonicPhaseAverage` | eingeschwungener Zustand, reelle 2n-Blockform, zeitgemittelte Kraft |
| **FEM-direkt** | `FemDirectBC`, `FemDirectRun`, `SparseDirectSolver` | Dirichlet-Rand, Vorbedingungen, dünn besetzter Direktlöser (cuDSS, optional) |
| **Vorkonditionierer** | `AmsPreconditioner`, `AlgebraicMultigrid`, `BlockPreconditionerStudy`, `RitzSpectrum` | AMS auf Aggregations-AMG, Deckenmessung für den BEM-Block, Konditionsschätzung |
| **Referenz** | `ReferenceSolver`, `NodalHexahedron`, `NodalBem`, `NodalMaxwell`, `DenseLinalg`, `BiotSavart`, `FieldVerification` | unabhängige Zweitrechnung mit Knotenelementen und vollbesetzter BEM, analytisches Quellfeld |
| **Backend** | `Backend`, `DeviceArray`, `HostLinalg` | GPU/CPU-Umschaltung, Speicher, CPU-Ersatz für cuBLAS und cuSPARSE |
| **Ausgabe** | `VtkExporter` | Felder als VTK für ParaView |

## Zwei Pfade, bewusst getrennt

- **Produktiver Pfad:** Nédélec-Kantenelemente, Baum-Kobaum-Eichung plus
  Kohomologie, BEM mit ACA+.
- **Referenzpfad** (`--reference`): trilineare Knotenelemente, Eichung über
  einen Strafterm, vollbesetzte BEM. Er ist absichtlich einfach gehalten und
  teilt keinen Code mit dem produktiven Pfad, damit er als unabhängige
  Zweitrechnung taugt.

## GPU und CPU

Die rechenintensiven Kernel sind als `__host__ __device__`-Funktionen
geschrieben und laufen auf beiden Seiten gleich. Der CPU-Modus (`--cpu`)
ersetzt cuBLAS und cuSPARSE durch `HostLinalg` und legt Speicher mit
`std::malloc` statt `cudaMallocManaged` an. Das CUDA-Toolkit braucht man
trotzdem zum Bauen und Linken. Details in [BUILD.md](../BUILD.md).

## Konventionen

- `component_id` einer Zelle bestimmt das Material: Außenspule 1, Innenspule 2,
  Platte 3, Luft 0. Luftzellen gibt es nur in Netzen für den FEM-direkt-Pfad;
  das Standardnetz vernetzt den Außenraum nicht, den übernimmt die BEM. Vertauschte Spulen erzeugen keinen Absturz, sondern
  ein falsches Feld; `tools/check_map.py` prüft die Zuordnung.
- Das Netz ist in Millimetern, die Physik in SI-Einheiten; die Umrechnung
  geschieht beim Einlesen.
