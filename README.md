# FEMBEM

[![CI](https://github.com/PhotonPower/FEMBEM/actions/workflows/ci.yml/badge.svg)](https://github.com/PhotonPower/FEMBEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**English summary.** FEMBEM is an experimental transient eddy-current solver
(Maxwell, A-formulation) that couples a finite-element discretisation with
edge elements (Nédélec, first kind) in the conductor to a boundary-element
discretisation of the surrounding air. It targets the *TEAM Workshop Problem 28*
benchmark (electrodynamic levitation of an aluminium plate above two coils),
runs on NVIDIA GPUs (CUDA) or entirely on the CPU (`--cpu`), and ships a
test suite, an independent reference solver and a written record of the
numerical findings. The documentation is mostly in German; the source code
comments too. See [BUILD.md](BUILD.md) for build instructions.

---

Transienter Wirbelstrom-Löser mit **BEM-FEM-Kopplung** und **Kantenelementen**,
entwickelt am Benchmark *TEAM Workshop Problem 28* (elektrodynamische
Levitation einer Aluminiumplatte über zwei Spulen).

## Was das Projekt kann

- **FEM mit Nédélec-Kantenelementen** auf Hexaedern, Tetraedern und Prismen
  (Mesh-Eingabe als Legacy-VTK, ASCII).
- **BEM für den Außenraum**, mit ACA+-Kompression, Cluster-Baum und
  Sauter-Schwab-Quadratur für singuläre und benachbarte Elementpaare.
- **Eichung** über Baum-Kobaum-Zerlegung mit Kohomologie-Erweiterung.
- **Iterativer Löser** (GMRES) für das gekoppelte System mit ILU(0) oder dem
  Auxiliary-Space-Maxwell-Vorkonditionierer (AMS, Hiptmair-Xu) auf
  Aggregations-AMG.
- **Harmonischer Löser** für den eingeschwungenen Zustand (zeitgemittelte
  Lorentzkraft) und ein **FEM-direkt-Pfad** mit dünn besetztem Direktlöser
  (optional, cuDSS).
- **GPU oder CPU:** dieselben Kernelrümpfe laufen auf der GPU oder – mit
  `--cpu` – ohne Grafikkarte und Treiber, optional mit OpenMP.
- **Unabhängige Zweitrechnung** (`--reference`): Knotenelemente mit vollbesetzter
  BEM, dazu eine analytische Biot-Savart-Referenz für das Quellfeld.
- **Diagnosewerkzeuge** für Konditionszahl, Ritz-Spektrum und
  Vorkonditionierer-Studien.

## Status

Das Projekt ist ein **Forschungs- und Arbeitsstand**, kein fertiges Produkt.
Der Löser rechnet, aber die iterative Lösung des gekoppelten Systems
konvergiert langsam: gemessen wurde eine Konditionszahl von etwa 10⁶, die als
Sockel aus der Struktur des Problems erklärt wird. Welche Vorkonditionierer
getestet wurden, was sich als falsch erwiesen hat und welche Fehler dabei
behoben wurden, steht in
[docs/reports/convergence-findings.tex](docs/reports/convergence-findings.tex).
Die Ergebnisse sind nicht als validiert anzusehen, solange sie nicht gegen die
Referenzwerte des Benchmarks geprüft wurden.

Der gesamte Quelltext und die Dokumentation wurden mit Hilfe von
KI-Systemen erzeugt. Prüfen Sie Ergebnisse, bevor Sie sich darauf verlassen.

## Schnellstart

Voraussetzungen: CMake ≥ 3.20, ein C++17-Compiler und das CUDA-Toolkit 12.x
(auch für den CPU-Modus, zum Bauen und Linken). Für die Rechnung auf der GPU
wird Compute Capability ≥ 6.0 gebraucht.

```sh
# Compute Capability der GPU ohne Punkt angeben (7.5 -> 75, 8.9 -> 89)
cmake -B build -DFEMBEM_CUDA_ARCH=75
cmake --build build -j

cd build
./fembem --help
./fembem 3 --cpu          # drei Zeitschritte, ohne GPU
```

Das Standardnetz `data/team28/team28.vtk` wird beim Konfigurieren ins
Build-Verzeichnis kopiert; das Programm erwartet es im Arbeitsverzeichnis.
Plattformdetails, GCC/CUDA-Versionen, OpenMP und cuDSS stehen in
[BUILD.md](BUILD.md).

## Tests

```sh
cmake --build build --target host_tests
ctest --test-dir build -L host --output-on-failure
```

Die Tests mit der Marke `host` brauchen keine GPU und lesen keine Dateien.
Weitere Tests laufen mit `ctest --test-dir build` ohne `-L`; sie brauchen
teils eine GPU oder das Netz. Aufbau und Konventionen:
[tests/README.md](tests/README.md).

## Aufbau des Repositories

```
FEMBEM/
├── CMakeLists.txt        Build, Tests (CTest), Option FEMBEM_BUILD_TESTS
├── BUILD.md              Bauen und Ausführen (Windows, Linux, CPU/GPU, cuDSS)
├── src/                  Implementierung (.cpp, .cu); Einstieg: main.cpp -> fem.cu
├── include/              Header, je Modul mit erklärendem Kopfkommentar
├── tests/                Testprogramme, bei CTest registriert
├── tools/                Python-Werkzeuge: Netzimport, Netzstudie, Plots
│   └── plot/
├── data/team28/          Standardnetz und Vergleichsdaten zu TEAM 28
└── docs/
    ├── architecture.md   Überblick über die Module
    ├── theory/           Formulierung und Numerik (LaTeX)
    ├── reports/          Mess- und Befundberichte (LaTeX)
    └── design/           Entwurfs- und Arbeitsnotizen (Markdown)
```

## Dokumentation

Einstieg und Übersicht: [docs/README.md](docs/README.md). Die LaTeX-Dokumente
lassen sich mit `make -C docs` zu PDF übersetzen.

## Mitmachen

Siehe [CONTRIBUTING.md](CONTRIBUTING.md). Änderungen stehen im
[CHANGELOG](CHANGELOG.md).

## Lizenz

[MIT](LICENSE) © 2026 PhotonPower
