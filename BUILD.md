# Bauen

Das Projekt baut mit demselben `CMakeLists.txt` unter Windows 11 (MSVC) und
Linux (GCC). Plattformabhängig sind nur die Warn- und Optimierungsflags; die
Unterscheidung passiert automatisch über `if (MSVC)`.

## Voraussetzungen

| | Windows 11 | RHEL 10 |
|---|---|---|
| Compiler | Visual Studio 2022, Workload „Desktop development with C++" | GCC, siehe Versionshinweis unten |
| CUDA | Toolkit 12.x | Toolkit 12.x |
| CMake | ≥ 3.20 | ≥ 3.20 |
| GPU | Compute Capability ≥ 6.0 | dito |

Compute Capability ≥ 6.0 ist nicht optional, **sofern auf der GPU gerechnet
wird**: der Löser benutzt `atomicAdd(double*, double)`, und das gibt es erst ab
Pascal. Für den CPU-Modus (`--cpu`, siehe unten) wird keine GPU benötigt — das
CUDA-Toolkit muss zum Bauen und Linken aber trotzdem vorhanden sein.

Eigene Compute Capability ermitteln, Punkt weglassen (7.5 → 75):

    nvidia-smi --query-gpu=name,compute_cap --format=csv

## Windows 11

In VS Code die **Developer PowerShell for VS 2022** verwenden, nicht die
normale PowerShell — sonst liegen `cl.exe` und die MSVC-Umgebung nicht im PATH,
und nvcc braucht MSVC als Hostcompiler.

    cmake -B build -S . -DFEMBEM_CUDA_ARCH=75
    cmake --build build --config Release
    cd build
    .\Release\fembem.exe --help

Der Visual-Studio-Generator ist multi-config, deshalb wählt `--config` die
Konfiguration und nicht `CMAKE_BUILD_TYPE`.

## RHEL 10 und andere Linux-Distributionen

    cmake -B build -S . -DFEMBEM_CUDA_ARCH=75
    cmake --build build -j$(nproc)
    cd build && ./fembem --help

Make und Ninja sind Einzelkonfigurations-Generatoren. `CMAKE_BUILD_TYPE` wird
automatisch auf `Release` gesetzt, falls nicht angegeben — ohne das würde ohne
jede Optimierung gebaut, was bei diesem Löser Größenordnungen kostet, ohne
dass man es bemerkt. Explizit überschreibbar:

    cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo

### GCC-Version

Jede CUDA-Version akzeptiert nur GCC bis zu einer bestimmten Hauptversion.
CUDA 12.0 bis 12.3 gehen bis GCC 12, RHEL 10 liefert aber GCC 14 als Standard.
Bricht nvcc mit `unsupported GNU version` ab:

    sudo dnf install gcc-toolset-12
    cmake -B build -S . \
      -DFEMBEM_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/g++

`FEMBEM_CUDA_HOST_COMPILER` muss beim **ersten** Konfigurieren gesetzt werden,
weil CMake den CUDA-Hostcompiler beim Aktivieren der Sprache in `project()`
prüft. Nachträglich hilft nur ein frisches Build-Verzeichnis.

Der Konfigurationslauf gibt die erkannten Versionen aus und warnt, wenn die
Kombination voraussichtlich nicht passt.

## Ausführen

Aus dem **Build-Verzeichnis** starten (`cd build`). CMake kopiert das
Standardnetz `data/team28/team28.vtk` beim Konfigurieren dorthin; `src/fem.cu`
öffnet es über den relativen Namen `team28.vtk`, und die Ausgabedateien
`team28_out_NNNN.vtk` landen ebenfalls im Arbeitsverzeichnis. Von einem
anderen Verzeichnis aus: `--mesh=PFAD`. Unter Windows liegt die Datei
`fembem.exe` in `build\Release\`, das Netz aber in `build\`; dort aus `build\`
mit `.\Release\fembem.exe` starten.

Optionales Argument: Anzahl Zeitschritte, für Kurztests.

    ./fembem 3          # 3 statt 1000 Schritten

Vollständige Optionsliste:

    ./fembem --help

### Rechenwerk wählen: GPU oder CPU

Standard ist die GPU. Mit `--cpu` läuft **die gesamte** Rechnung auf der CPU;
es wird dann kein einziger CUDA-Kernel gestartet und kein Gerätespeicher belegt,
das Programm braucht also weder GPU noch funktionsfähigen Treiber.

    ./fembem --cpu               # alles auf der CPU
    ./fembem 3 --cpu             # Kurztest, CPU
    ./fembem 3 --cpu --threads=8 # mit 8 OpenMP-Threads
    ./fembem 5 nobem --cpu       # Diagnoseschalter bleiben nutzbar
    ./fembem --gpu               # GPU erzwingen (= Standard)

Die Reihenfolge der Argumente ist beliebig. Alternativ per Umgebungsvariable,
praktisch für Batch-Läufe:

    FEMBEM_BACKEND=cpu ./fembem 3

Eine Option auf der Kommandozeile überstimmt die Umgebungsvariable.

**Wie das funktioniert.** Jeder Kernel besteht aus einer
`__host__ __device__`-Rumpffunktion `<name>_body(i, …)`, die die eigentliche
Rechnung trägt, und einem dünnen `__global__`-Kernel, der nur den Index bildet.
Die Aufrufstelle benutzt `BF_LAUNCH_1D` / `BF_LAUNCH_1D_PAR` / `BF_LAUNCH_2D`
aus `include/Backend.cuh` und entscheidet zur Laufzeit zwischen Kernelstart und
Host-Schleife. GPU- und CPU-Pfad führen damit buchstäblich denselben Code aus
und können nicht auseinanderlaufen — anders als bei zwei getrennten
Implementierungen desselben Integranden.

Ersetzt sind außerdem: `cudaMalloc`/`Memcpy`/`Memset`/`DeviceSynchronize`
(→ `bf::`-Wrapper), `cudaMallocManaged` in `DeviceArray` (→ `std::malloc`),
sowie die benutzten cuBLAS- und cuSPARSE-Operationen inklusive
ILU(0)-Faktorisierung und Dreieckssubstitutionen (→ `src/HostLinalg.cu`).

**Was Sie erwarten sollten.**

- *Tempo.* Der CPU-Modus ist erheblich langsamer. Teuerster Posten ist die
  BEM-Nahfeld-Integration (Sauter-Schwab-Selbstpatch, 4⁴ Gaußpunkte pro
  Quad-Paar); die parallelisiert OpenMP. Ohne OpenMP-Build läuft alles seriell.
- *Reproduzierbarkeit.* Die streuenden Kernel (RHS-Assemblierung, CSR-Scatter,
  T-Kopplung, Maxwell-Integration) laufen im CPU-Modus bewusst **seriell**, weil
  `bf::atomic_add` dort eine gewöhnliche Addition ist. Die Summationsreihenfolge
  liegt damit fest, und die Ergebnisse sind von Lauf zu Lauf bitidentisch. Auf
  der GPU ist die Reihenfolge von `atomicAdd` nicht festgelegt. Der CPU-Modus
  eignet sich deshalb als Referenz für `test_determinism_fingerprint`.
- *Numerik.* Die Reduktionen (`dnrm2`, `ddot`) summieren seriell, cuBLAS
  baumförmig. Über ~10⁵ Elemente unterscheiden sich die Ergebnisse in den
  letzten Bits. Für die Abbruchkriterien des Krylov-Lösers ist das unerheblich,
  ein bitweiser Vergleich GPU gegen CPU schlägt aber fehl.

### Referenzloeser (`--reference`)

Zweite, unabhaengige Diskretisierung desselben Problems — gedacht als
Vergleichsrechnung, wenn dem produktiven Pfad nicht zu trauen ist.

    ./fembem 3 --reference

| | produktiv | `--reference` |
|---|---|---|
| Elemente | Nédélec-Kanten | Q1-**Knoten**, 3 DOF/Knoten |
| Eichung | Baum-Kobaum + Kohomologie | div-div-**Strafterm** |
| BEM | ACA / H-Matrix | **vollbesetzt**, keine Approximation |
| Unbekannte | 2×2-Block mit q | q eliminiert (Dirichlet-zu-Neumann) |

**Voraussetzung:** µ_r = 1 im gesamten Gebiet. Bei TEAM 28 erfuellt (Alu,
Kupfer, Luft); der Loeser prueft es beim Start und bricht sonst ab. Nur unter
dieser Bedingung ist curl-curl + div-div gleich dem komponentenweisen
Vektor-Laplace, und nur dann ist der Randterm der BEM-Kopplung eindeutig.
Angenehme Folge: die drei Komponenten von A entkoppeln in drei skalare
Diffusionsprobleme, und V und K sind fuer alle drei **dieselbe** Matrix — die
vollbesetzte BEM kostet deshalb nur 2 · 4138² · 8 B = 0,26 GiB statt des
Dreifachen.

### Validierungsstand des Referenzlösers

Alle Zahlen ausgeführt, nicht geschätzt.

| Prüfung | Referenz | Ergebnis |
|---|---|---|
| Quellfeld B_z auf der Achse (`--verify-field`, σ=0) | Biot-Savart, analytisch | **0,06…0,21 %** |
| Abklingverhältnis B_z(oben)/B_z(unten) | Biot-Savart | **0,15 %** |
| Wirbelstrom-Verlustleistung | Born-Grenzfall, elliptische Integrale | **1,7 %** |
| magnetisches Moment m_z | dito | 15 % |
| **Zyklusmittel F_z bei z = 11,3 mm** | **exakt m·g (Ruhelage der Messung)** | **7,0 %** |
| vorhergesagte Ruhelage | 11,3 mm (Messung) | 10,90 mm (−0,4 mm) |
| Beschleunigung am Scheitel 18,2 mm | −8,9…−10,2 m/s² (Messung) | −7,5 m/s² |

Der Test bei 11,3 mm ist der einzige *streng exakte*: dort muss das Zyklusmittel
der Vertikalkraft m·g sein, sonst schwebt die Platte nicht. Alles andere aus der
Messung (Anfangsbeschleunigung, Scheitelhöhe) ist eine Ableitung mit
Fehlerbalken — und die Aufgabenstellung selbst weist die Abhebephase als
unsicher aus („These effects may influence the plate during its initial lift-off
phase").

**Kraftschätzer: nur das Lorentz-Volumenintegral benutzen.** Bei 11,3 mm liefert
der Maxwell-Spannungstensor 0,172·m·g statt 1,0 (83 % Abweichung), bei 18,2 mm
sogar ein negatives Zyklusmittel. Netzverfeinerung ändert daran fast nichts
(Verhältnis 0,894 → 0,946). Grund: der Spannungstensor braucht das Feld genau
auf der Unstetigkeitsfläche und bildet dort die Differenz zweier großer, fast
gleicher Terme, während der gesuchte Gleichanteil nur etwa ein Drittel der
Schwankungsamplitude ist. Er wird weiter mitgerechnet, taugt aber nur als
Momentanwert-Diagnose.

**Bekannte Grenze, gemessen:** die Neumann-Spur q wird mit *stetiger*
Knotenbasis dargestellt. An glatten Flaechen konvergiert die BEM damit mit
O(h²) (an der Kugel nachgemessen: 1,40e-2 → 4,00e-3 → 1,83e-3). An
geometrischen **Kanten** ist q physikalisch unstetig, was eine stetige Basis
nicht darstellen kann — am Wuerfel bleiben ~10 % Restfehler und die Ordnung
faellt auf etwa O(h^0,75). Platte und Spulen *sind* Kaesten mit scharfen
Kanten, dieser Fehler ist also vorhanden. Standardabhilfe ware unstetiges q
(Doppelknoten an Kanten); bewusst nicht in dieser ersten Fassung.
`test_reference_solver` misst beide Faelle und schreibt sie fest.

### Netzverfeinerung (`--refine=N`)

    ./fembem 400 --reference --refine=1

Uniforme Verfeinerung, je Stufe **1 → 8** Kinder (2×2×2). Nicht 1→4: eine
Teilung in nur zwei Richtungen liesse die Plattendicke unveraendert, und genau
die ist knapp — drei Elemente auf 3 mm.

| Stufe | Elemente | Knoten | Randquads | N_bem | dichte BEM |
|---|---|---|---|---|---|
| 0 | 4 805 | 7 013 | 4 136 | 4 138 | 0,38 GiB |
| 1 | 38 440 | 46 991 | 16 544 | 16 542 | **6,1 GiB** |
| 2 | 307 520 | ~361 000 | 66 176 | ~66 000 | ~100 GiB |

Neue Knoten entstehen an Kantenmitten, Flaechenmitten und Zellmitten; ihre Lage
folgt der trilinearen Geometrieabbildung des Elternelements, ist also mit der
Q1-Diskretisierung konsistent. Kanten- und Flaechenmitten werden zwischen
Nachbarelementen **geteilt** (kanonischer Schluessel aus den sortierten
Elternknoten-IDs) — sonst entstuenden haengende Knoten und das Netz waere
zerrissen. Component-IDs werden vererbt, die Materialzuweisung funktioniert
unveraendert.

**Gilt fuer BEIDE Loeser.** `--refine` sitzt in der gemeinsamen Setup-Phase,
vor der Topologieextraktion — Kanten, Randflaechen, Materialzuweisung,
Kinematik-Caches, ACA-Baum und CSR-Muster werden alle auf dem verfeinerten Netz
aufgebaut. `--refine=1` allein verfeinert also den produktiven
Kantenelement-Pfad, `--refine=1 --reference` den Knotenelement-Pfad:

| | Stufe 0 | Stufe 1 | Stufe 2 |
|---|---|---|---|
| Nédélec-DOF (Kanten, produktiv) | 18 690 | 132 142 | 989 292 |
| BEM-DOF (Rooftop, produktiv, ACA) | 16 544 | 66 176 | 264 704 |
| Knoten-DOF (`--reference`) | 21 039 | 140 973 | 1 023 495 |
| N_bem nodal, dichte BEM | 4 138 | 16 542 | ~66 000 |

Der produktive Pfad vertraegt mehr Stufen als `--reference`, weil die ACA nur
etwa N·log N Speicher braucht statt N². Der Rangpuffer der ACA-Bloecke
(`d_temp_z`) waechst bei Bedarf mit, da ist keine feste Grenze im Weg.

**Zwei Einschraenkungen.** Erstens: mit vollbesetzter BEM passt auf 16 GiB genau
**eine** Stufe. Der Bedarf waechst mit N_bem², eine Stufe vervierfacht N_bem und
damit die 16-fache Matrix. Der Referenzloeser prueft das gegen den freien
VRAM und bricht vorher mit einer Meldung ab, statt mitten im Lauf an einer
Allokation zu scheitern. Fuer Stufe 2 braucht es den produktiven ACA-Pfad.

Zweitens: die **Geometrie wird nicht runder**. Die Mitte einer geraden Kante
liegt auf dieser Kante, der polyedrische Rand bleibt derselbe. Die Facettierung
des Plattenrands und der Spulenmaentel aendert sich nicht — verfeinert wird die
Auflaesung des *Feldes*, nicht die der *Form*.

### OpenMP

Standardmäßig eingeschaltet, sofern CMake OpenMP findet. Der
Konfigurationslauf gibt aus, ob es aktiv ist. Abschalten:

    cmake -B build -S . -DFEMBEM_ENABLE_OPENMP=OFF

## Tests

Alle Tests sind bei CTest registriert. Die Tests mit der Marke `host` laufen
ohne GPU und ohne Dateien; das CUDA-Toolkit wird zum Übersetzen trotzdem
gebraucht.

    # Linux
    cmake --build build --target host_tests -j$(nproc)
    ctest --test-dir build -L host --output-on-failure

    # Windows
    cmake --build build --config Release --target host_tests
    ctest --test-dir build -C Release -L host --output-on-failure

Alle Testprogramme bauen und starten (einige brauchen eine GPU oder das Netz
im Build-Verzeichnis):

    cmake --build build
    ctest --test-dir build --output-on-failure

Ein einzelner Test, zum Beispiel:

    cmake --build build --target test_cpu_backend
    ./build/test_cpu_backend

Mit `-DFEMBEM_BUILD_TESTS=OFF` werden keine Testprogramme gebaut. Marken,
Aufbau und das Hinzufügen eines Tests: [tests/README.md](tests/README.md).

Was sie prüfen:

- **test_nedelec_orientation** — beide Curl-Implementierungen gegen die
  numerische Ableitung der jeweiligen Basisfunktion, plus `detJ > 0` für alle
  von `generateCoil` erzeugten Elemente.
- **test_edge_sign_convention** — lokale Elementmatrizen müssen unorientiert
  aufgebaut werden; die Orientierung gehört in den Scatter. Der Fehler war
  symmetrieerhaltend, eine Symmetrieprüfung findet ihn also nicht.
- **test_curl_gradient_nullspace** — `K · a = 0` für `a_e = φ(n_B) − φ(n_A)`.
  Physikalische Eigenschaft, unabhängig von der Curl-Herleitung. Gibt zwei
  Zeilen aus; die zweite ist der Kontrollfall und soll `DURCHGEFALLEN` zeigen.
- **test_bem_index_mapping** — sechs Invarianten der BEM-zu-FEM-Abbildung.
- **test_cpu_backend** — die CPU-Linearalgebra aus `src/HostLinalg.cu`, die im
  CPU-Modus cuBLAS und cuSPARSE ersetzt. Geprüft werden CSR-SpMV und `dgemv`
  (beide Transpositionen) gegen dichte Referenzrechnungen, die definierende
  ILU(0)-Eigenschaft `(L·U)_ij = A_ij` auf dem Besetzungsmuster, die
  Dreieckssubstitutionen als *exakter* Löser im Tridiagonalfall (dort entsteht
  kein Fill-in, ILU(0) ist also die exakte LU — deutlich schärfer als eine
  Toleranzabfrage) und die Überlaufsicherheit von `dnrm2`. Braucht keine GPU.
- **test_reference_solver** — die Bausteine von `--reference`: Q1-Formfunktionen
  (Zerlegung der Eins, linearer Patch-Test), die Duffy-Transformation fuer das
  singulaere Einfachschichtintegral gegen den analytischen Wert
  2·ln(1+√2)/(4π), die Randmassenmatrix gegen die Flaeche, beide
  Randintegralgleichungen auf einer geschlossenen Flaeche samt gemessener
  Konvergenzordnung, und die dichte LU. Braucht keine GPU.
- **test_determinism_fingerprint** — bitweise Prüfsummen jeder Stufe der Kette
  Netz → Topologie → Schwerpunkte → Sortierung → Cluster-Baum. Dafür gedacht,
  Plattformabweichungen zu lokalisieren: zwei Rechner vergleichen ihre
  Ausgaben, die erste abweichende Zeile benennt die Stufe. Referenzwerte und Lesart: [tests/README.md](tests/README.md).

Alle müssen auf beiden Plattformen identische Zahlen liefern. Zwei
Determinismus-Bugs, die das verhindert haben, sind behoben: eine instabile
Sortierung in `ClusterTree` und die Iteration über eine `unordered_map` in
`TopologyExtractor::extract_boundary_faces`.

## cuDSS (nur fuer `--fem-direct`)

Der FEM-direkt-Pfad loest das dünn besetzte System mit einer Zerlegung statt
iterativ. Dafür wird **cuDSS** gebraucht, eine separate NVIDIA-Bibliothek, die
nicht Teil des CUDA-Toolkits ist.

Die Einbindung ist **optional**. Ohne cuDSS baut alles wie bisher, nur
`--fem-direct` fehlt und meldet das mit Begründung. Wer am transienten oder am
BEM-FEM-Pfad arbeitet, braucht cuDSS nicht.

Unter Windows mit cuDSS 0.8 und CUDA 13 **bestätigt funktionierend**:

    cmake -B build -S . -DFEMBEM_CUDA_ARCH=120 -DFEMBEM_WITH_CUDSS=ON -DCMAKE_PREFIX_PATH="C:/Program Files/NVIDIA cuDSS/v0.8/lib/13/cmake/cudss"

Der Pfad muss auf das Verzeichnis **mit** `cudss-config.cmake` zeigen, nicht
auf die Installationswurzel. `.../v0.8` allein reicht nicht: `find_package`
sucht dort nur die üblichen Unterordner (`lib/cmake/cudss`), und cuDSS legt
die Konfiguration unter Windows eine Ebene tiefer nach CUDA-Hauptversion ab.
Findet CMake nichts, gibt es eine Warnung und es wird ohne `--fem-direct`
gebaut — der Lauf scheitert dann erst spät und an einer Stelle, die mit dem
Pfad nichts zu tun hat.

Das `13` im Pfad ist die CUDA-Hauptversion und muss zum Toolkit passen (siehe
den Abschnitt darunter).

Entwickelt und geprüft gegen **cuDSS 0.8.0**. **Mindestens 0.8 ist nötig:** in
0.7.x fehlen Symbole, die `SparseDirectSolver.cpp` braucht (zum Beispiel
`CUDSS_R_32I`, `cudssReorderingAlg_t`, `CUDSS_PIVOT_AUTO`,
`CUDSS_CONFIG_HYBRID_MEMORY_MODE`). CMake lehnt ältere Versionen mit einer
Warnung ab und baut dann ohne `--fem-direct`.

### Zwei Unverträglichkeiten, beide am Determinismus

Gemessen an cuDSS 0.8.0, durch Bisektion über alle Einzel- und
Paarkombinationen, auf RTX 5060 (sm_120) und RTX 2080 (sm_75):

```
DETERMINISTIC_MODE + IR_N_STEPS > 0    -> ANALYSIS NOT_SUPPORTED (4)
DETERMINISTIC_MODE + HYBRID_MEMORY     -> ANALYSIS NOT_SUPPORTED (4)
```

Ordnung und Pivotierung sind an beidem unbeteiligt. Beide Male schränkt der
Determinismus die interne Ablaufplanung ein, und sowohl die Nachbesserung als
auch die Auslagerung brauchen dort Freiheit.

`SparseDirectSolver` löst das nach einer festen Rangfolge auf und meldet jede
Entscheidung:

1. **Hybridmodus gewinnt immer.** Ohne ihn passt der Faktor eines verfeinerten
   Netzes nicht in den Gerätespeicher, und dann rechnet das System gar nicht.
   Ein nicht gerechnetes Ergebnis ist nicht reproduzierbar, sondern abwesend.
   **Läufe mit `FEMBEM_CUDSS_HYBRID=1` sind nicht bitweise reproduzierbar**,
   `test_determinism_fingerprint` gilt für sie nicht.
2. Sonst gewinnt der Determinismus über die Nachbesserung.

Nützliche Folge: ist der Determinismus wegen des Hybridmodus ohnehin aus,
wird die Nachbesserung wieder eingeschaltet statt stillschweigend
weggelassen — gerade auf einem verfeinerten Netz, wo die Kondition schlechter
wird, ist sie dort am nützlichsten.

Zur Fehlersuche sind die Einstellungen zur Laufzeit überschreibbar, nach dem
Muster von `FEMBEM_BACKEND`:

| Variable | Werte |
|---|---|
| `FEMBEM_CUDSS_MINIMAL` | `1` = kein einziges `cudssConfigSet`, alles auf cuDSS-Vorgabe |
| `FEMBEM_CUDSS_REORDER` | `default`, `amd`, `nd`, `skip` |
| `FEMBEM_CUDSS_PIVOT` | `skip` |
| `FEMBEM_CUDSS_DETERMINISTIC` | `0`, `1`, `skip` |
| `FEMBEM_CUDSS_IR` | Zahl, `skip` |
| `FEMBEM_CUDSS_HYBRID` | `0`, `1` -- verlagert Teile des Faktors in den Hauptspeicher |
| `FEMBEM_CUDSS_HYBRID_LIMIT` | Obergrenze des Geräteanteils in MiB |

`skip` lässt den jeweiligen `cudssConfigSet`-Aufruf weg. Das ist nicht dasselbe
wie ihn auf 0 zu setzen: `cudssConfigSet` nimmt auch entgegen, was die
Bibliothek nicht trägt, und beanstandet wird erst beim Ausführen. Jeder Lauf
druckt die tatsächlich gesetzte Kombination als `[cuDSS] Konfiguration:`.

`PIVOT` sollte man nicht weglassen. Ohne Pivotierung läuft die Zerlegung
durch, liefert aber ein relatives Residuum um 0,5 statt 1e-12.

### Die CUDA-Hauptversion muss passen

cuDSS wird je CUDA-Hauptversion gebaut, und die Archive heißen entsprechend
(`…_cuda12-archive`, `…_cuda13-archive`). Die Installation MUSS zur
Hauptversion des Toolkits passen, mit dem hier gebaut wird. Eine
cuda12-Variante gegen ein 13.x-Toolkit gelinkt fällt **erst beim Laden der
DLL** auf, nicht beim Übersetzen — also im Zweifel vorher nachsehen:

    nvcc --version
    dir "C:\Program Files\NVIDIA cuDSS\v0.8\lib"

Liegen dort Unterordner `12` und `13`, wählt `find_package` anhand des
gefundenen Toolkits; liegt nur einer da, muss er stimmen.

Findet CMake das Paket nicht, gibt es eine Warnung mit den erwarteten Pfaden
statt eines Abbruchs. Gesucht wird `cudss-config.cmake`, üblicherweise unter
`<cuDSS>/lib/cmake/cudss/` beziehungsweise `<cuDSS>/lib/<CUDA-Version>/cmake/cudss/`.

### Zur Laufzeit

Die DLL muss auffindbar sein. Entweder liegt `<cuDSS>/bin` im `PATH`, oder
`cudss64_0.dll` wird neben die Exe kopiert.
