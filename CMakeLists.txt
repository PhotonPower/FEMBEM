cmake_minimum_required(VERSION 3.20)

# =============================================================================
# Hostcompiler fuer nvcc, MUSS vor project() gesetzt werden
# =============================================================================
# CMake prueft den CUDA-Hostcompiler beim Aktivieren der CUDA-Sprache in
# project(), danach ist die Variable wirkungslos.
#
# Relevant unter Linux: jede CUDA-Version akzeptiert nur GCC bis zu einer
# bestimmten Hauptversion. CUDA 12.0 geht bis GCC 12, RHEL 10 liefert aber
# GCC 14 als Standard. In dem Fall:
#
#   sudo dnf install gcc-toolset-12
#   cmake -B build -DFEMBEM_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/g++
#
# Alternativ eine neuere CUDA-Version installieren.
if (DEFINED FEMBEM_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${FEMBEM_CUDA_HOST_COMPILER}" CACHE FILEPATH
        "Hostcompiler, den nvcc benutzt" FORCE)
endif()

project(fembem LANGUAGES CXX CUDA)

# =============================================================================
# Build-Typ: bei Einzelkonfigurations-Generatoren einen Standard setzen
# =============================================================================
# Der Visual-Studio-Generator ist multi-config, dort waehlt --config Release.
# Unter Linux erzeugt CMake standardmaessig Makefiles oder Ninja, also
# Einzelkonfiguration -- und ohne CMAKE_BUILD_TYPE wird OHNE Optimierung
# gebaut. Bei diesem Loeser waere das um Groessenordnungen langsamer, ohne dass
# man es merkt. Deshalb ein Standard.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build-Typ: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
    message(STATUS "FEMBEM: CMAKE_BUILD_TYPE war leer, setze Release.")
endif()

# 🧮 M_PI ist kein C++-Standard, sondern eine POSIX-Erweiterung. MSVC definiert
# es in <math.h> nur, wenn _USE_MATH_DEFINES vorher gesetzt ist -- sonst
# scheitern fem.cu, SauterSchwab.cu, AdjacentIntegration.cu und
# BemIntegration.cuh mit "identifier M_PI is undefined".
# Auf GCC/Clang ist das Makro wirkungslos. Muss vor add_executable stehen.
add_compile_definitions(_USE_MATH_DEFINES)

# =============================================================================
# OpenMP fuer den CPU-Modus (--cpu)
# =============================================================================
# Im CPU-Modus laufen die rechenintensiven, nachweislich rennfreien Kernel
# (lokale FEM-Matrizen, BEM-Dense-Bloecke, Feldinterpolation) ueber
# BF_LAUNCH_1D_PAR / BF_LAUNCH_2D und damit ueber "omp parallel for". Ohne
# OpenMP laeuft alles seriell -- korrekt, aber deutlich langsamer.
#
# Wichtig: OpenMP::OpenMP_CXX setzt seine Flags nur fuer die Sprache CXX. Die
# .cu-Dateien uebersetzt nvcc, dort muss das Flag per -Xcompiler an den
# Hostcompiler durchgereicht werden -- sonst werden die Pragmas dort
# stillschweigend ignoriert und genau die teuren Kernel bleiben seriell.
#
# Abschaltbar mit: cmake -B build -DFEMBEM_ENABLE_OPENMP=OFF
option(FEMBEM_ENABLE_OPENMP "OpenMP fuer den CPU-Modus verwenden" ON)

set(FEMBEM_OPENMP_AVAILABLE OFF)
if (FEMBEM_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        set(FEMBEM_OPENMP_AVAILABLE ON)
        # Flagliste fuer -Xcompiler kommasepariert aufbereiten (kann mehrere
        # Einträge enthalten, z. B. bei Intel-Toolchains).
        string(REPLACE ";" "," FEMBEM_OMP_NVCC_FLAGS "${OpenMP_CXX_FLAGS}")
        string(REPLACE " " "," FEMBEM_OMP_NVCC_FLAGS "${FEMBEM_OMP_NVCC_FLAGS}")
    else()
        message(STATUS "FEMBEM: OpenMP nicht gefunden -- der CPU-Modus laeuft seriell.")
    endif()
endif()

# 🔍 CUDA Toolkit finden
find_package(CUDAToolkit REQUIRED)

# =============================================================================
# cuDSS -- dünn besetzter Direktlöser für den FEM-direkt-Pfad (AP6)
# =============================================================================
# ABSICHTLICH OPTIONAL. cuDSS ist eine separate Installation, nicht Teil des
# CUDA-Toolkits. Wäre es Pflicht, könnte niemand mehr bauen, der nur am
# transienten oder am BEM-FEM-Pfad arbeitet -- und die brauchen es nicht.
#
# Ohne cuDSS fehlt genau eine Fähigkeit: --fem-direct. Der Schalter meldet das
# dann zur Laufzeit mit Begründung, statt zu einem Linkfehler zu führen.
#
#     cmake -B build -DFEMBEM_WITH_CUDSS=ON \
#           -DCMAKE_PREFIX_PATH="C:/Program Files/NVIDIA cuDSS/v0.8/lib/13/cmake/cudss"
#
# Der Pfad zeigt auf das Verzeichnis MIT cudss-config.cmake, nicht auf die
# Installationswurzel. Unter Windows liegt es je CUDA-Hauptversion getrennt
# (lib/13/cmake/cudss); "…/v0.8" allein findet find_package nicht.
#
# WICHTIG ZUR VARIANTE: cuDSS wird je CUDA-Hauptversion gebaut. Dieses Projekt
# baut mit CUDA 12 oder 13; die cuDSS-Installation MUSS zur selben
# Hauptversion passen. Eine cuda12-Variante gegen ein 13.x-Toolkit gelinkt
# fällt erst beim Laden der DLL auf, nicht beim Bauen.
option(FEMBEM_WITH_CUDSS "Sparse-Direktlöser cuDSS einbinden (fuer --fem-direct)" OFF)

set(FEMBEM_CUDSS_FOUND FALSE)
if (FEMBEM_WITH_CUDSS)
    find_package(cudss QUIET)
    # Entwickelt und geprueft gegen cuDSS 0.8.0. Aeltere Versionen (0.7.x)
    # kennen Symbole nicht, die SparseDirectSolver.cpp braucht
    # (CUDSS_R_32I, cudssReorderingAlg_t, CUDSS_PIVOT_AUTO,
    # CUDSS_CONFIG_HYBRID_MEMORY_MODE, ...); der Bau scheitert dann erst
    # mitten in der Uebersetzung. Deshalb hier ablehnen, wie "nicht gefunden".
    set(FEMBEM_CUDSS_MIN_VERSION 0.8)
    if (cudss_FOUND AND NOT "${cudss_VERSION}" STREQUAL ""
            AND cudss_VERSION VERSION_LESS FEMBEM_CUDSS_MIN_VERSION)
        message(WARNING
            "cuDSS ${cudss_VERSION} gefunden, gebraucht wird mindestens ${FEMBEM_CUDSS_MIN_VERSION}.\n"
            "  Die aeltere Version kennt Symbole nicht, die --fem-direct braucht.\n"
            "  Eine neuere cuDSS-Installation (gleiche CUDA-Hauptversion wie das\n"
            "  Toolkit) ueber -DCMAKE_PREFIX_PATH=<...>/cmake/cudss angeben.\n"
            "  Ohne cuDSS wird ohne --fem-direct gebaut; alles andere bleibt nutzbar.")
    elseif (cudss_FOUND)
        set(FEMBEM_CUDSS_FOUND TRUE)
        message(STATUS "cuDSS gefunden: Version ${cudss_VERSION}")
    else()
        # Kein FATAL_ERROR: der Rest des Projekts ist ohne cuDSS voll
        # funktionsfähig. Aber deutlich sichtbar, weil der Nutzer es
        # ausdrücklich angefordert hat.
        message(WARNING
            "FEMBEM_WITH_CUDSS=ON, aber find_package(cudss) hat nichts gefunden.\n"
            "  Erwartet wird das Verzeichnis MIT cudss-config.cmake, nicht die\n"
            "  Installationswurzel:\n"
            "    <cuDSS>/lib/cmake/cudss/            (Linux)\n"
            "    <cuDSS>/lib/<CUDA-Version>/cmake/cudss/  (Windows)\n"
            "  Bestaetigt funktionierend unter Windows mit cuDSS 0.8 und CUDA 13:\n"
            "    -DCMAKE_PREFIX_PATH=\"C:/Program Files/NVIDIA cuDSS/v0.8/lib/13/cmake/cudss\"\n"
            "  Ohne cuDSS wird ohne --fem-direct gebaut; alles andere bleibt nutzbar.")
    endif()
endif()

# Unter Windows liegt cudss64_0.dll in <cuDSS>/bin/<CUDA-Version>, und das
# steht normalerweise nicht im PATH. Ohne die DLL startet ein Programm gar
# nicht erst (STATUS_DLL_NOT_FOUND) -- und unter CTest gibt es nicht einmal
# eine Meldung: test_sparse_direct lief ohne jede Ausgabe in seine
# Zeitgrenze von 120 s, waehrend derselbe Test mit der DLL im PATH in unter
# einer Sekunde besteht. Deshalb die DLL nach dem Bauen neben jedes Programm
# kopieren, das gegen cuDSS linkt. Das importierte Ziel kennt den Pfad
# (IMPORTED_LOCATION aus cudss-targets-release.cmake).
function(fembem_link_cudss target)
    target_link_libraries(${target} PRIVATE cudss)
    target_compile_definitions(${target} PRIVATE FEMBEM_HAVE_CUDSS=1)
    if (WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:cudss> $<TARGET_FILE_DIR:${target}>
            VERBATIM)
    endif()
endfunction()

# 📐 Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 🎯 GPU-Architektur, ueberschreibbar per
#   cmake -B build -DFEMBEM_CUDA_ARCH=86
# Eigene Compute Capability ermitteln (Punkt weglassen, 7.5 -> 75):
#   nvidia-smi --query-gpu=name,compute_cap --format=csv
# RTX 2080: 75 | RTX 30xx: 86 | RTX 40xx: 89 | A100: 80 | H100: 90
# RTX 50xx (Blackwell, z. B. RTX 5060): 120 -- braucht MINDESTENS CUDA 12.8.
#   Mit einem aelteren Toolkit uebersetzt der Build durch, aber der erste
#   Kernel-Start scheitert mit "named symbol not found": das Binary enthaelt
#   keinen Code fuer die Karte. Dasselbe passiert bei falsch gesetztem
#   FEMBEM_CUDA_ARCH -- alles bis zum ersten Kernel laeuft host-seitig und
#   sieht gesund aus.
# Tesla P100: 60
# Hinweis: CUDA 12.0 unterstuetzt maximal compute_90.
#
# WICHTIG: hier keine "if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)"-Abfrage
# verwenden. CMake setzt diese Variable beim Aktivieren der CUDA-Sprache in
# project() selbst auf den nvcc-Standard (compute_52), die Bedingung ist also
# immer falsch. Folge waere ein Build mit CC 5.2, in dem atomicAdd(double*)
# nicht existiert. Stattdessen eine eigene Cache-Variable, die -D ueberschreibt.
set(FEMBEM_CUDA_ARCH 75 CACHE STRING "CUDA Compute Capability ohne Punkt, z. B. 75")
set(CMAKE_CUDA_ARCHITECTURES ${FEMBEM_CUDA_ARCH})

# 📄 Quelldateien (Explizit auflisten ist besser als GLOB!)
set(SOURCES
    # Füge weitere .cpp/.cu Dateien hier hinzu
    src/Backend.cu              # GPU/CPU-Umschaltung (--cpu)
    src/HostLinalg.cu           # CPU-Ersatz fuer cuBLAS/cuSPARSE
    src/DenseLinalg.cu          # dichte LU (cuSOLVER / eigene)
    src/ReferenceSolver.cu      # Baseline Knoten-FEM + vollbesetzte BEM
    src/MeshRefinement.cu       # uniforme Verfeinerung (--refine=N)
    src/BiotSavart.cpp          # analytische Referenz (--verify-field)
    src/FieldVerification.cpp   # gemeinsamer Vergleich beider Loeserpfade
    src/DiscreteGradient.cpp    # Grundbausteine Hiptmair-Vorkonditionierer
    src/AmsPreconditioner.cpp   # Auxiliary-Space Maxwell Solver
    src/AlgebraicMultigrid.cpp  # Aggregations-AMG fuer die AMS-Hilfsraeume
    src/RitzSpectrum.cpp        # Konditionsschaetzung aus dem Arnoldi-Prozess
    src/BlockPreconditionerStudy.cpp  # Deckenmessung fuer den BEM-Block (--pc-study)
    src/HarmonicSolver.cu             # harmonischer Loeser (C-to-R)
    src/HarmonicPhaseAverage.cpp      # Zeitmittelung ueber Phasen
    src/FemDirectBC.cpp               # Dirichlet-Rand und Regularisierung (FEM-direkt)
    src/HarmonicBlockSystem.cpp       # reelle 2n-Blockform des harmonischen Systems
    src/SparseDirectSolver.cpp        # cuDSS-Anbindung (optional, siehe FEMBEM_WITH_CUDSS)
    src/FemDirectRun.cpp              # Vorbedingungen des FEM-direkt-Pfades
    src/AcaPlus.cpp             
    src/NonlinearSolver.cpp    
    src/TransientSolver.cu
    src/AdjacentIntegration.cu  
    src/NonlinearSolver.cu     
    src/VtkExporter.cu
    src/fem.cu         
    src/PhysicalObject.cpp     
    src/VtkReader.cpp
    src/BemTopology.cpp         
    src/BemCoupling.cpp
    src/TreeCotreeGauge.cpp
    src/CohomologyGauge.cpp
    src/Kinematics.cu  
    src/SauterSchwab.cu
    src/ClusterTree.cpp         
    src/main.cpp       
    src/CoupledSolver.cu        
    src/TopologyExtractor.cpp
)

# 🔨 Executable erstellen
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 🔗 Mit CUDA verlinken
# CoupledSolver.cu/.cuh nutzt cuSPARSE (cusparseSpMV, cusparseCreateCsr, ...)
# und cuBLAS (cublasDnrm2/Ddot/Dscal/Daxpy/Dcopy/Dgemv), NonlinearSolver.cpp
# ebenfalls cuBLAS. Ohne diese beiden Ziele scheitert das Host-Linking mit
# 16 x LNK2019/LNK2001 auf cusparse*- und cublas*-Symbolen.
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::cusolver      # dichte LU fuer den Referenzloeser (V^-1)
)

# FEMBEM_HAVE_CUDSS steuert im Quellcode, ob der Loeseraufruf einkompiliert
# wird. Ohne cuDSS bleibt der Rest unveraendert baubar.
if (FEMBEM_CUDSS_FOUND)
    fembem_link_cudss(${PROJECT_NAME})
endif()

if (FEMBEM_OPENMP_AVAILABLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${FEMBEM_OMP_NVCC_FLAGS}>)
endif()

# 🧩 Relocatable Device Code ist fuer dieses Projekt zwingend, nicht optional:
#   - fem.cu ruft integrate_singular_self_patch auf, ein (jetzt
#     __host__ __device__) Symbol aus der Uebersetzungseinheit SauterSchwab.cu.
#     Ohne -rdc=true scheitert ptxas mit "Unresolved extern function".
#
# HINWEIS: Der zweite frueher hier genannte Grund ist entfallen. SauterSchwab.cuh
# deklarierte "extern __constant__ double GL_PTS[4]" (definiert in
# SauterSchwab.cu). Fuer den CPU-Modus muessen die Gauss-Stuetzstellen aber auch
# aus Host-Code lesbar sein, und __constant__ ist das nicht -- sie liegen jetzt
# als inline-Funktionen GL_PT()/GL_WG() im Header. Damit verschwindet zugleich
# das extern-__constant__-Problem (Warnung #20044-D / C2086).
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ⚙️ Compiler-Flags
# -Wall/-Wextra ist GCC/Clang-Syntax. MSVC kennt kein /Wextra und bricht mit
# "D8021: Ungueltiges numerisches Argument" ab; ausserdem gibt es kein /O3,
# maximal /O2. Die Optimierung ueberlassen wir dem Build-Typ, --config Release
# setzt bereits /O2 bzw. -O3. Unter MSVC bleibt CMakes Standardwarnstufe aktiv.
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra -lineinfo>
    )
endif()

# =============================================================================
# Daten und Tests
# =============================================================================
# Das Standardnetz liegt unter data/team28/. Programm und Tests oeffnen es ueber
# den relativen Namen "team28.vtk", also wird es ins Build-Verzeichnis kopiert
# (CTest startet die Tests dort).
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/data/team28/team28.vtk"
    "${CMAKE_CURRENT_BINARY_DIR}/team28.vtk"
    COPYONLY)

option(FEMBEM_BUILD_TESTS "Testprogramme bauen und bei CTest registrieren" ON)

if (FEMBEM_BUILD_TESTS)

# Reine C++-Tests ohne Abhaengigkeiten. Sie enthalten die geprueften Formeln
# selbst und brauchen weder CUDA-Laufzeit noch Netzdateien.
foreach(_t test_nedelec_orientation test_edge_sign_convention test_curl_gradient_nullspace)
    add_executable(${_t} tests/${_t}.cpp)
    set_target_properties(${_t} PROPERTIES LINKER_LANGUAGE CXX)
endforeach()

# 🧪 Abnahmepruefung fuer die BEM->FEM-Indexabbildung.
# Reiner Host-Code, wird aber als .cu uebersetzt, damit er direkt gegen die
# echten Header (Mesh.cuh, DeviceArray.cuh) baut statt gegen Kopien.
# Startet keine Kernel, braucht aber einen CUDA-Kontext fuer den Unified Memory
# von DeviceArray. Bauen mit:
#   cmake --build build --config Release --target test_bem_index_mapping
add_executable(test_bem_index_mapping
    tests/test_bem_index_mapping.cu
    src/Backend.cu              # bf::-Symbole fuer DeviceArray
    src/VtkReader.cpp
    src/TopologyExtractor.cpp
    src/ClusterTree.cpp
    src/BemCoupling.cpp
)
target_include_directories(test_bem_index_mapping PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_bem_index_mapping PRIVATE CUDA::cudart)

# 🧪 Vorzeichenkonvention der Kopplungs-Massenmatrix [T]
# Haelt die Invariante zwischen Operator (-T oben rechts) und blocktriangulaerem
# Vorkonditionierer (+T in der Rueckwaertssubstitution) fest. Reiner Host-Code,
# ohne Netz und ohne GPU-Rechnung -- ruft die Ruempfe aus
# include/CouplingKernels.cuh direkt auf. Bauen mit:
#   cmake --build build --config Release --target test_coupling_sign_convention
add_executable(test_coupling_sign_convention
    tests/test_coupling_sign_convention.cu
    src/Backend.cu              # bf::-Symbole
)
target_include_directories(test_coupling_sign_convention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_coupling_sign_convention PRIVATE CUDA::cudart)

# 🧪 G_BEM ist als Operator exakt symmetrisch
# Prueft <x, G y> == <G x, y> am angewandten Operator, mit Gegenprobe. Bauen:
#   cmake --build build --config Release --target test_gbem_symmetry
add_executable(test_gbem_symmetry
    tests/test_gbem_symmetry.cu
    src/CoupledSolver.cu
    src/Backend.cu
    src/HostLinalg.cu
    src/AmsPreconditioner.cpp
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/RitzSpectrum.cpp
)
target_include_directories(test_gbem_symmetry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_gbem_symmetry PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 Selbstpruefung des --direct-Verfahrens
# --direct deutet ~1e-15 als "loesbar" und ~1e-2 als "inkonsistent". Diese
# Deutung traegt nur, wenn das Verfahren selbst funktioniert. Bauen mit:
#   cmake --build build --config Release --target test_dense_direct
add_executable(test_dense_direct
    tests/test_dense_direct.cu
    src/DenseLinalg.cu
    src/Backend.cu
)
target_include_directories(test_dense_direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_dense_direct PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)

# 🧪 Selbstpruefung der Deckenmessung fuer den BEM-Block (--pc-study)
# Reiner Host-Code, KEINE CUDA-Abhaengigkeit -- deshalb als .cpp und ohne
# target_link_libraries auf CUDA. Baut damit auch auf einer Maschine ohne
# Toolkit, was fuer genau dieses Werkzeug der Punkt ist: die Frage, ob sich
# der Aufwand fuer Calderon lohnt, soll man beantworten koennen, ohne erst
# eine GPU-Umgebung aufzusetzen. Bauen mit:
#   cmake --build build --config Release --target test_block_preconditioner_study
# 🧪 Zeitmittelung des harmonischen Loesers -- reiner Host-Code, kein CUDA.
# Der Teil, an dem ein Vorzeichenfehler still bliebe: er erzeugt keinen
# Absturz, sondern eine systematisch falsche Kraft und damit eine falsche
# Schwebehoehe.
add_executable(test_harmonic_phase_average
    tests/test_harmonic_phase_average.cpp
    src/HarmonicPhaseAverage.cpp
)
target_include_directories(test_harmonic_phase_average PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_block_preconditioner_study
    tests/test_block_preconditioner_study.cpp
    src/BlockPreconditionerStudy.cpp
)
target_include_directories(test_block_preconditioner_study PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 🧪 Nedelec-Kantenelement auf dem Prisma (penta6) -- der FEM-direkt-Pfad.
# Reiner Host-Code, startet keine Kernel, braucht kein Netz. Prueft
# Kantenzirkulation, Rotation gegen zentrale Differenzen, Volumen,
# Gradienten-Nullraum und den Patchtest mit konstantem B. Bauen mit:
#   cmake --build build --config Release --target test_nedelec_prism
add_executable(test_nedelec_prism
    tests/test_nedelec_prism.cu
)
target_include_directories(test_nedelec_prism PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_nedelec_prism PRIVATE CUDA::cudart)

# 🧪 Nedelec-Kantenelement auf dem Tetraeder (CTETRA) -- der Elementtyp des
# gelieferten Netzes. Der Patchtest laeuft ueber 20000 Zufallselemente, weil
# der lineare Tetraeder immer affin abgebildet ist und konstantes B deshalb
# auf JEDER Elementform exakt sein muss.
add_executable(test_nedelec_tet
    tests/test_nedelec_tet.cu
)
target_include_directories(test_nedelec_tet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_nedelec_tet PRIVATE CUDA::cudart)

# 🧪 Topologie gemischter Netze. Ruft den ECHTEN TopologyExtractor auf und
# prueft ihn gegen von Hand nachgerechnete Werte (Kuhn-Wuerfel aus 6
# Tetraedern: 19 Kanten, 12 Randdreiecke). Prueft ausserdem, dass die
# Kantentabellen in Mesh.cuh und in den Nedelec-Headern nicht auseinander
# laufen, und dass ein reines Hexaedernetz sich bitgleich wie vorher verhaelt.
add_executable(test_mixed_topology
    tests/test_mixed_topology.cu
    src/TopologyExtractor.cpp
    src/Backend.cu
)
target_include_directories(test_mixed_topology PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_mixed_topology PRIVATE CUDA::cudart)

# 🧪 Patchtest UEBER Elementgrenzen und Konsistenz des Quellterms.
# test_nedelec_tet prueft den Patchtest schon, aber auf EINEM Element mit
# selbst gesetzten Koeffizienten. Hier laeuft er ueber ein Netz aus 162
# Kuhn-Tetraedern und sammelt die Koeffizienten aus den GLOBALEN
# Freiheitsgraden -- damit sind erstmals die Kantennummerierung, die
# Orientierungsvorzeichen und die Konformitaet mitgeprueft. Ein Vorzeichen-
# fehler in dieser Schicht erzeugt keinen Absturz, sondern ein falsches Feld.
#
# Der eigentliche Zweck ist aber die Frage hinter M2: warum steht
# ||G^T b||/||b|| auf dem BDF-Netz bei 6,1e-03 statt 1e-4. Der Test trennt
# Quadraturfehler von der facettierten Spulenoberflaeche und zeigt, dass nur
# der erste konvergiert. Siehe docs/design/fem-direct-plan.md.
# 🧪 Gradientenprojektion des Quellterms: reiner Gradient verschwindet,
# Idempotenz, Orthogonalitaet. Punkt 2 ist der, der sonst nicht auffaellt --
# ein Verfahren, das jedes Mal ein Stueck mehr wegnimmt, sieht beim ersten
# Aufruf richtig aus.
# 🧪 Baum-Kobaum-Eichung auf einem TETRAEDERNETZ mit Randflaechen. Genau die
# Kombination, in der die Randabbildung -1 auf dem vierten Platz eines
# Dreiecks liefert -- ohne extract_boundary_faces() ist der Fehler nicht
# erreichbar, und daran ist er den bestehenden Tests durchgegangen.
add_executable(test_gauge_on_tets
    tests/test_gauge_on_tets.cu
    src/TreeCotreeGauge.cpp
    src/CohomologyGauge.cpp
    src/TopologyExtractor.cpp
    src/Backend.cu
)
target_include_directories(test_gauge_on_tets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_gauge_on_tets PRIVATE CUDA::cudart)

add_executable(test_source_projection
    tests/test_source_projection.cu
    src/FemDirectBC.cpp
    src/TopologyExtractor.cpp
    src/Backend.cu
)
target_include_directories(test_source_projection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_source_projection PRIVATE CUDA::cudart)

add_executable(test_patch_assembly
    tests/test_patch_assembly.cu
    src/TopologyExtractor.cpp
    src/Backend.cu
)
target_include_directories(test_patch_assembly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_patch_assembly PRIVATE CUDA::cudart)

# 🧪 Dirichlet-Rand und Regularisierung (AP4). Der Kern ist die
# Aequivalenzpruefung: das eingearbeitete volle System muss dasselbe liefern
# wie das auf die freien Freiheitsgrade reduzierte. Dazu der Kuhn-Wuerfel, bei
# dem von Hand nachgerechnet 18 von 19 Kanten auf dem Rand liegen.
# 🧪 Typabhaengige Auswertung (AP7). Prueft NedelecDispatch.cuh gegen die
# Elementheader und rekonstruiert ein konstantes B aus Kantenkoeffizienten --
# genau das, was VtkExporter und HarmonicLorentz tun.
add_executable(test_element_dispatch
    tests/test_element_dispatch.cu
    src/Backend.cu
)
target_include_directories(test_element_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_element_dispatch PRIVATE CUDA::cudart)

# 🧪 Piola-Transformation in den PRODUKTIVEN Feldkoerpern. test_element_dispatch
# prueft die Helfer piola_covariant/piola_contravariant; dieser Test ruft die
# Koerper selbst auf (VtkExporter-Zellmittel, Maxwell-Kraft des transienten
# Pfades, harmonische Lorentz- und Maxwell-Kraft) -- auf einem gedrehten und
# gescherten Element, weil die Transposition auf einem Quader unsichtbar ist.
# Mit Kontrollfall (transponierte Formel muss durchfallen).
add_executable(test_piola_production
    tests/test_piola_production.cu
    src/Backend.cu
)
target_include_directories(test_piola_production PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_piola_production PRIVATE CUDA::cudart)

# 🧪 Reelle 2n-Blockform (AP5). Loest dasselbe Problem zweimal auf
# verschiedenen Wegen -- komplex mit std::complex, und ueber die Blockform --
# und vergleicht. Das prueft alle Vorzeichen gleichzeitig.
add_executable(test_block_system
    tests/test_block_system.cu
    src/HarmonicBlockSystem.cpp
)
target_include_directories(test_block_system PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_block_system PRIVATE CUDA::cudart)

# 🧪 Dirichlet im n-Raum gegen Dirichlet in der 2n-Blockform (AP5).
# Der AMS-Modus (--fem-direct-nullspace=ams) baut die 2n-Matrix nicht, sondern
# haelt in W = K + omega*C UND in C getrennt fest -- und die beiden brauchen
# VERSCHIEDENE Diagonalwerte auf den Randkanten (d beziehungsweise 0). Mit
# demselben Wert bleibt das System loesbar, koppelt aber Real- und
# Imaginaerteil der Randkante; sichtbar waere das nur als seltsame
# Iterationszahl. Der Test enthaelt die Gegenprobe mit dem eingebauten Fehler.
# Bauen mit:
#   cmake --build build --config Release --target test_harmonic_dirichlet_split
add_executable(test_harmonic_dirichlet_split
    tests/test_harmonic_dirichlet_split.cu
    src/HarmonicBlockSystem.cpp
    src/FemDirectBC.cpp
    src/Backend.cu
)
target_include_directories(test_harmonic_dirichlet_split PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_harmonic_dirichlet_split PRIVATE CUDA::cudart)

# 🧪 Musterpruefung und Verhalten ohne cuDSS (AP6). Laeuft MIT und OHNE cuDSS
# und prueft in beiden Faellen etwas Sinnvolles: die Zerlegung selbst kann er
# nicht nachrechnen, die Eingabepruefung und die Schnittstelle zur Blockform
# aus AP5 schon.
# 🧪 Vorbedingungen des FEM-direkt-Pfades (AP8). Wichtigste Pruefung: ein Netz
# OHNE Luftgebiet muss abgelehnt werden -- dort ist der aeussere Rand die
# Leiteroberflaeche, und A x n = 0 darauf ergibt eine konvergierte, plausibel
# aussehende, falsche Loesung.
add_executable(test_fem_direct_preflight
    tests/test_fem_direct_preflight.cu
    src/FemDirectRun.cpp
    src/FemDirectBC.cpp
    src/TopologyExtractor.cpp
    src/Backend.cu
)
target_include_directories(test_fem_direct_preflight PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_fem_direct_preflight PRIVATE CUDA::cudart)

add_executable(test_sparse_direct
    tests/test_sparse_direct.cu
    src/SparseDirectSolver.cpp
    src/HarmonicBlockSystem.cpp
)
target_include_directories(test_sparse_direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_sparse_direct PRIVATE CUDA::cudart)
if (FEMBEM_CUDSS_FOUND)
    fembem_link_cudss(test_sparse_direct)
endif()

add_executable(test_dirichlet_bc
    tests/test_dirichlet_bc.cu
    src/FemDirectBC.cpp
    src/TopologyExtractor.cpp
    src/Backend.cu
)
target_include_directories(test_dirichlet_bc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_dirichlet_bc PRIVATE CUDA::cudart)

# 🧪 Wirken die Vorkonditionierer am AEUSSEREN Loeser? (End-to-End)
# Die einzige Pruefung, die die Kennzahl misst, auf die es ankommt: die Zahl der
# GMRES-Iterationen. Alle anderen messen Teilgroessen. Bauen mit:
#   cmake --build build --config Release --target test_preconditioner_end_to_end
add_executable(test_preconditioner_end_to_end
    tests/test_preconditioner_end_to_end.cu
    src/CoupledSolver.cu
    src/Backend.cu
    src/HostLinalg.cu
    src/AmsPreconditioner.cpp
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/RitzSpectrum.cpp
)
target_include_directories(test_preconditioner_end_to_end PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_preconditioner_end_to_end PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 GMRES und BiCGStab brechen bei NaN/Inf sofort ab, statt bis max_iter zu
# laufen und ein NaN-x zurueckzugeben (docs/design/nan-and-thresholds.md).
# Bauen mit:
#   cmake --build build --config Release --target test_solver_nonfinite_abort
add_executable(test_solver_nonfinite_abort
    tests/test_solver_nonfinite_abort.cu
    src/CoupledSolver.cu
    src/Backend.cu
    src/HostLinalg.cu
    src/AmsPreconditioner.cpp
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/RitzSpectrum.cpp
)
target_include_directories(test_solver_nonfinite_abort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_solver_nonfinite_abort PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 --harmonic-inner-direct: C-to-R mit dichter Zerlegung von (W+T) gegen die
# dichte Loesung des komplexen Systems; Abtastung column-major.
# Bauen mit:
#   cmake --build build --config Release --target test_harmonic_inner_direct
add_executable(test_harmonic_inner_direct
    tests/test_harmonic_inner_direct.cu
    src/HarmonicSolver.cu
    src/DenseLinalg.cu
    src/CoupledSolver.cu
    src/Backend.cu
    src/HostLinalg.cu
    src/AmsPreconditioner.cpp
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/RitzSpectrum.cpp
)
target_include_directories(test_harmonic_inner_direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_harmonic_inner_direct PRIVATE
    CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)

# 🧪 apply_many(): BEM-Bloecke gesammelt (dgemm) gegen apply() je Spalte,
# nicht quadratische dichte/transponierte/ACA-Bloecke, Eichung, Kohomologie.
# Bauen mit:
#   cmake --build build --config Release --target test_apply_many
add_executable(test_apply_many
    tests/test_apply_many.cu
    src/HarmonicSolver.cu
    src/DenseLinalg.cu
    src/CoupledSolver.cu
    src/Backend.cu
    src/HostLinalg.cu
    src/AmsPreconditioner.cpp
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/RitzSpectrum.cpp
)
target_include_directories(test_apply_many PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_apply_many PRIVATE
    CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)

# 🧪 Vektor-Poisson-Operator als AMG-Grundlage des vektoriellen AMS-Hilfsraums
# Prueft ausserdem die Knotengradienten je Elementtyp (Tetraeder, Prisma,
# Hexaeder) auf zufaellig verzerrten Einzelelementen -- ohne die ist der
# Operator auf den Tetraedernetzen des FEM-direkt-Pfades nicht aufbaubar.
# Bauen mit:
#   cmake --build build --config Release --target test_alpha_poisson
add_executable(test_alpha_poisson
    tests/test_alpha_poisson.cu
    src/DiscreteGradient.cpp
    src/AlgebraicMultigrid.cpp
    src/Backend.cu
)
target_include_directories(test_alpha_poisson PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_alpha_poisson PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 AMG auf anisotropen Gittern -- die Ursache des schlechten skalaren
# AMS-Hilfsraums (0.72 statt 0.30). Bauen mit:
#   cmake --build build --config Release --target test_amg_anisotropy
add_executable(test_amg_anisotropy
    tests/test_amg_anisotropy.cu
    src/DiscreteGradient.cpp
    src/AlgebraicMultigrid.cpp
    src/Backend.cu
)
target_include_directories(test_amg_anisotropy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_amg_anisotropy PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 AMG loest den 3D-Laplace (Bezugswert fuer den AMS-Konvergenzfaktor)
# Bauen mit:
#   cmake --build build --config Release --target test_amg_convergence
add_executable(test_amg_convergence
    tests/test_amg_convergence.cu
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/Backend.cu
)
target_include_directories(test_amg_convergence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_amg_convergence PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 AMS: skalarer Hilfsraum bei rundungskleinem A_G
# Haelt fest, dass das Ausblenden der informationslosen Zeilen die Verstaerkung
# von 1e+12 auf das Niveau des gesunden Falls zurueckholt. Bauen mit:
#   cmake --build build --config Release --target test_ams_scalar_masking
add_executable(test_ams_scalar_masking
    tests/test_ams_scalar_masking.cu
    src/AlgebraicMultigrid.cpp
    src/AmsPreconditioner.cpp
    src/DiscreteGradient.cpp
    src/Backend.cu
)
target_include_directories(test_ams_scalar_masking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_ams_scalar_masking PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 Genauigkeit der Sauter-Schwab-Quadratur (4-Punkt-Regel)
# Haelt fest, dass die Produktionsordnung die konvergierten Werte (N=24) auf
# 2e-3 trifft -- insbesondere fuer den Doppelschichtkern bei gemeinsamer Kante,
# dem kritischsten Fall. Reiner Host-Code. Bauen mit:
#   cmake --build build --config Release --target test_singular_quadrature
add_executable(test_singular_quadrature
    tests/test_singular_quadrature.cu
)
target_include_directories(test_singular_quadrature PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_singular_quadrature PRIVATE CUDA::cudart)

# 🧪 bf::dgemv darf y bei beta == 0 nicht lesen (BLAS-Semantik)
# Bauen mit:
#   cmake --build build --config Release --target test_dgemv_beta_zero
add_executable(test_dgemv_beta_zero
    tests/test_dgemv_beta_zero.cu
    src/HostLinalg.cu
    src/Backend.cu
)
target_include_directories(test_dgemv_beta_zero PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_dgemv_beta_zero PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 NaN-/Inf-Propagation in bf::dnrm2
# Haelt fest, dass ein Vektor aus lauter NaN die Norm NaN liefert und nicht 0.0.
# Bauen mit:
#   cmake --build build --config Release --target test_norm_nan_propagation
add_executable(test_norm_nan_propagation
    tests/test_norm_nan_propagation.cu
    src/HostLinalg.cu
    src/Backend.cu
)
target_include_directories(test_norm_nan_propagation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_norm_nan_propagation PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 Konsistenz des blocktriangulaeren Vorkonditionierers (--blocktri)
# Baut den Operator im CPU-Modus mit Eichung UND Kohomologie-Projektoren auf und
# haelt fest, dass beide Vorkonditionierer-Zweige bei r_K == 0 exakt dasselbe
# liefern muessen. Bauen mit:
#   cmake --build build --config Release --target test_block_triangular_consistency
add_executable(test_block_triangular_consistency
    tests/test_block_triangular_consistency.cu
    src/CoupledSolver.cu
    src/Backend.cu
    src/HostLinalg.cu
    src/AmsPreconditioner.cpp
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/RitzSpectrum.cpp
)
target_include_directories(test_block_triangular_consistency PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_block_triangular_consistency PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# 🧪 Regressionstest fuer das is_moving()-Gate (kraftgetriebene Objekte)
add_executable(test_free_motion_gate
    tests/test_free_motion_gate.cpp
    src/Backend.cu              # bf::-Symbole fuer DeviceArray
    src/PhysicalObject.cpp
)
target_include_directories(test_free_motion_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_free_motion_gate PRIVATE CUDA::cudart)

# 🧪 Eichungstest
add_executable(test_tree_cotree_gauge
    tests/test_tree_cotree_gauge.cu
    src/Backend.cu              # bf::-Symbole fuer DeviceArray
    src/VtkReader.cpp
    src/TopologyExtractor.cpp
    src/TreeCotreeGauge.cpp
)
target_include_directories(test_tree_cotree_gauge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_tree_cotree_gauge PRIVATE CUDA::cudart)

# 🧪 Kohomologie-Eichungstest (diskrete Hodge-Zerlegung)
add_executable(test_cohomology_gauge
    tests/test_cohomology_gauge.cu
    src/Backend.cu              # bf::-Symbole fuer DeviceArray
    src/VtkReader.cpp
    src/TopologyExtractor.cpp
    src/TreeCotreeGauge.cpp
    src/CohomologyGauge.cpp
)
target_include_directories(test_cohomology_gauge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cohomology_gauge PRIVATE CUDA::cudart)

# 🔬 Determinismus-Fingerprint: lokalisiert plattformabhaengige Divergenzen
add_executable(test_determinism_fingerprint
    tests/test_determinism_fingerprint.cu
    src/Backend.cu              # bf::-Symbole fuer DeviceArray
    src/VtkReader.cpp
    src/TopologyExtractor.cpp
    src/ClusterTree.cpp
)
target_include_directories(test_determinism_fingerprint PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_determinism_fingerprint PRIVATE CUDA::cudart)

# 🧪 Abnahmepruefung des CPU-Backends (--cpu)
# Prueft die neu geschriebene CPU-Linearalgebra aus src/HostLinalg.cu, die im
# CPU-Modus cuBLAS und cuSPARSE ersetzt: CSR-SpMV, dgemv (beide
# Transpositionen), ILU(0) samt der definierenden Eigenschaft
# (L*U)_ij == A_ij auf dem Besetzungsmuster, die Dreieckssubstitutionen und
# die ueberlaufsichere Norm. Braucht KEINE GPU -- der Test stellt bf::g_backend
# selbst auf CPU. Bauen mit:
#   cmake --build build --config Release --target test_cpu_backend
add_executable(test_cpu_backend
    tests/test_cpu_backend.cu
    src/Backend.cu
    src/HostLinalg.cu
)
target_include_directories(test_cpu_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cpu_backend PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
if (FEMBEM_OPENMP_AVAILABLE)
    target_link_libraries(test_cpu_backend PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(test_cpu_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${FEMBEM_OMP_NVCC_FLAGS}>)
endif()

# Abnahmepruefung des Referenzloesers (--reference)
add_executable(test_reference_solver
    tests/test_reference_solver.cu
    src/Backend.cu
    src/HostLinalg.cu
    src/DenseLinalg.cu
)
target_include_directories(test_reference_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_reference_solver PRIVATE
    CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)
if (FEMBEM_OPENMP_AVAILABLE)
    target_link_libraries(test_reference_solver PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(test_reference_solver PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${FEMBEM_OMP_NVCC_FLAGS}>)
endif()

# Unsymmetrie-Schaetzer
add_executable(test_asymmetry_estimator tests/test_asymmetry_estimator.cu)
target_include_directories(test_asymmetry_estimator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_asymmetry_estimator PRIVATE CUDA::cudart)

# Ritz-Spektrum: Eigenwerte der Hessenberg-Matrix
add_executable(test_ritz_spectrum tests/test_ritz_spectrum.cu src/RitzSpectrum.cpp)
target_include_directories(test_ritz_spectrum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_ritz_spectrum PRIVATE CUDA::cudart)

# Sauter-Schwab: Deckung des Integrationsgebiets
add_executable(test_sauter_schwab_coverage tests/test_sauter_schwab_coverage.cu)
target_include_directories(test_sauter_schwab_coverage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_sauter_schwab_coverage PRIVATE CUDA::cudart)

# Quadratur benachbarter BEM-Panelpaare
add_executable(test_adjacent_bem tests/test_adjacent_bem.cu)
target_include_directories(test_adjacent_bem PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_adjacent_bem PRIVATE CUDA::cudart)

# AMG: netzunabhaengige Konvergenz auf dem 3D-Knoten-Laplace
add_executable(test_amg
    tests/test_amg.cu
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/Backend.cu
)
target_include_directories(test_amg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_amg PRIVATE CUDA::cudart)

# AMS-Vorkonditionierer: Iterationszahlen gegen Jacobi und ILU(0)
add_executable(test_ams_preconditioner
    tests/test_ams_preconditioner.cu
    src/AmsPreconditioner.cpp
    src/AlgebraicMultigrid.cpp
    src/DiscreteGradient.cpp
    src/HostLinalg.cu
    src/Backend.cu
)
target_include_directories(test_ams_preconditioner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_ams_preconditioner PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

# Grundbausteine des Hiptmair-Vorkonditionierers
add_executable(test_discrete_gradient
    tests/test_discrete_gradient.cu
    src/DiscreteGradient.cpp
    src/Backend.cu
)
target_include_directories(test_discrete_gradient PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_discrete_gradient PRIVATE CUDA::cudart)

# Validierung der analytischen Biot-Savart-Referenz (--verify-field)
add_executable(test_biot_savart
    tests/test_biot_savart.cu
    src/BiotSavart.cpp
)
target_include_directories(test_biot_savart PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_biot_savart PRIVATE CUDA::cudart)

# Abnahmepruefung der Netzverfeinerung (--refine=N)
add_executable(test_mesh_refinement
    tests/test_mesh_refinement.cu
    src/Backend.cu
    src/MeshRefinement.cu
)
target_include_directories(test_mesh_refinement PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_mesh_refinement PRIVATE CUDA::cudart)

endif()  # FEMBEM_BUILD_TESTS (Testziele)


# =============================================================================
# Diagnose und Kompatibilitaetshinweis
# =============================================================================
message(STATUS "FEMBEM: System      ${CMAKE_SYSTEM_NAME}")
message(STATUS "FEMBEM: CXX         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "FEMBEM: CUDA        ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "FEMBEM: CUDA-Host   ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "FEMBEM: GPU-Arch    sm_${FEMBEM_CUDA_ARCH}")
message(STATUS "FEMBEM: Build-Typ   ${CMAKE_BUILD_TYPE}${CMAKE_CONFIGURATION_TYPES}")
if (FEMBEM_OPENMP_AVAILABLE)
    message(STATUS "FEMBEM: OpenMP      aktiv (${OpenMP_CXX_FLAGS}) -- CPU-Modus parallel")
else()
    message(STATUS "FEMBEM: OpenMP      inaktiv -- CPU-Modus (--cpu) laeuft seriell")
endif()

# Grobe Obergrenze der von nvcc akzeptierten GCC-Hauptversion. Die Tabelle ist
# aus den CUDA-Release-Notes und bewusst konservativ; sie fuehrt daher nur zu
# einer Warnung, nicht zu einem Abbruch.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_max_gcc 0)
    if     (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)  # 12.0 - 12.3
        set(_max_gcc 12)
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6)  # 12.4 - 12.5
        set(_max_gcc 13)
    else()                                                  # 12.6 und neuer
        set(_max_gcc 14)
    endif()
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "${_max_gcc}.99")
        message(WARNING
            "GCC ${CMAKE_CXX_COMPILER_VERSION} ist fuer CUDA "
            "${CMAKE_CUDA_COMPILER_VERSION} voraussichtlich zu neu (erwartet "
            "maximal GCC ${_max_gcc}). Falls nvcc mit "
            "\"unsupported GNU version\" abbricht, einen aelteren "
            "Hostcompiler waehlen:\n"
            "  cmake -B build -DFEMBEM_CUDA_HOST_COMPILER=/usr/bin/g++-${_max_gcc}\n"
            "Unter RHEL: sudo dnf install gcc-toolset-${_max_gcc}, dann\n"
            "  -DFEMBEM_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${_max_gcc}/root/usr/bin/g++")
    endif()
endif()

# ============================================================================
# CTest-Registrierung
# ============================================================================
#
#   cmake -B build
#   cmake --build build --target host_tests          # nur die GPU-freien Tests
#   ctest --test-dir build -L host --output-on-failure
#
# Die Marke "host" kennzeichnet Tests, die OHNE GPU auskommen. Sie bauen ihre
# Netze und Matrizen selbst und lesen keine Datei von der Platte, laufen also
# auch in einem leeren Arbeitsverzeichnis und in einer CI ohne Grafikkarte.
# Die uebrigen Tests brauchen teils eine GPU, teils das Netz team28.vtk
# (liegt im Build-Verzeichnis); sie laufen mit "ctest" ohne -L.
if (FEMBEM_BUILD_TESTS)
enable_testing()

foreach(_t test_nedelec_orientation test_edge_sign_convention test_curl_gradient_nullspace)
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES LABELS "host" TIMEOUT 120)
endforeach()

add_test(NAME test_coupling_sign_convention COMMAND test_coupling_sign_convention)
set_tests_properties(test_coupling_sign_convention PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_block_triangular_consistency COMMAND test_block_triangular_consistency)
set_tests_properties(test_block_triangular_consistency PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_norm_nan_propagation COMMAND test_norm_nan_propagation)
set_tests_properties(test_norm_nan_propagation PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_dgemv_beta_zero COMMAND test_dgemv_beta_zero)
set_tests_properties(test_dgemv_beta_zero PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_singular_quadrature COMMAND test_singular_quadrature)
set_tests_properties(test_singular_quadrature PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_ams_scalar_masking COMMAND test_ams_scalar_masking)
set_tests_properties(test_ams_scalar_masking PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_gbem_symmetry COMMAND test_gbem_symmetry)
set_tests_properties(test_gbem_symmetry PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_amg_convergence COMMAND test_amg_convergence)
set_tests_properties(test_amg_convergence PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_alpha_poisson COMMAND test_alpha_poisson)
set_tests_properties(test_alpha_poisson PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_amg_anisotropy COMMAND test_amg_anisotropy)
set_tests_properties(test_amg_anisotropy PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_preconditioner_end_to_end COMMAND test_preconditioner_end_to_end)
set_tests_properties(test_preconditioner_end_to_end PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_solver_nonfinite_abort COMMAND test_solver_nonfinite_abort)
set_tests_properties(test_solver_nonfinite_abort PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_harmonic_inner_direct COMMAND test_harmonic_inner_direct)
set_tests_properties(test_harmonic_inner_direct PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_apply_many COMMAND test_apply_many)
set_tests_properties(test_apply_many PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_dense_direct COMMAND test_dense_direct)
set_tests_properties(test_dense_direct PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_harmonic_phase_average COMMAND test_harmonic_phase_average)
set_tests_properties(test_harmonic_phase_average PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_block_preconditioner_study COMMAND test_block_preconditioner_study)
set_tests_properties(test_block_preconditioner_study PROPERTIES LABELS "host" TIMEOUT 600)
add_test(NAME test_nedelec_prism COMMAND test_nedelec_prism)
set_tests_properties(test_nedelec_prism PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_nedelec_tet COMMAND test_nedelec_tet)
set_tests_properties(test_nedelec_tet PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_mixed_topology COMMAND test_mixed_topology)
set_tests_properties(test_mixed_topology PROPERTIES LABELS "host" TIMEOUT 120)

# Laeuft bis N = 16, also 24576 Tetraeder, dreimal assembliert. Reiner
# Host-Code, aber nicht in Sekundenbruchteilen durch wie test_nedelec_tet.
add_test(NAME test_gauge_on_tets COMMAND test_gauge_on_tets)
set_tests_properties(test_gauge_on_tets PROPERTIES LABELS "host" TIMEOUT 120)

add_test(NAME test_source_projection COMMAND test_source_projection)
set_tests_properties(test_source_projection PROPERTIES LABELS "host" TIMEOUT 120)

add_test(NAME test_patch_assembly COMMAND test_patch_assembly)
set_tests_properties(test_patch_assembly PROPERTIES LABELS "host" TIMEOUT 300)
add_test(NAME test_dirichlet_bc COMMAND test_dirichlet_bc)
set_tests_properties(test_dirichlet_bc PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_element_dispatch COMMAND test_element_dispatch)
set_tests_properties(test_element_dispatch PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_piola_production COMMAND test_piola_production)
set_tests_properties(test_piola_production PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_block_system COMMAND test_block_system)
add_test(NAME test_harmonic_dirichlet_split COMMAND test_harmonic_dirichlet_split)
set_tests_properties(test_harmonic_dirichlet_split PROPERTIES LABELS "host" TIMEOUT 120)
set_tests_properties(test_block_system PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_sparse_direct COMMAND test_sparse_direct)
set_tests_properties(test_sparse_direct PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME test_fem_direct_preflight COMMAND test_fem_direct_preflight)
set_tests_properties(test_fem_direct_preflight PROPERTIES LABELS "host" TIMEOUT 120)

# Die uebrigen Tests: teils GPU-abhaengig, teils auf Netzdateien im
# Arbeitsverzeichnis angewiesen. Registriert, aber ohne die Marke "host" --
# wer sie laufen lassen will, ruft ctest ohne -L auf.
add_test(NAME test_bem_index_mapping COMMAND test_bem_index_mapping)
set_tests_properties(test_bem_index_mapping PROPERTIES TIMEOUT 900)
add_test(NAME test_free_motion_gate COMMAND test_free_motion_gate)
set_tests_properties(test_free_motion_gate PROPERTIES TIMEOUT 900)
add_test(NAME test_tree_cotree_gauge COMMAND test_tree_cotree_gauge)
set_tests_properties(test_tree_cotree_gauge PROPERTIES TIMEOUT 900)
add_test(NAME test_cohomology_gauge COMMAND test_cohomology_gauge)
set_tests_properties(test_cohomology_gauge PROPERTIES TIMEOUT 900)
add_test(NAME test_determinism_fingerprint COMMAND test_determinism_fingerprint)
set_tests_properties(test_determinism_fingerprint PROPERTIES TIMEOUT 900)
add_test(NAME test_cpu_backend COMMAND test_cpu_backend)
set_tests_properties(test_cpu_backend PROPERTIES TIMEOUT 900)
add_test(NAME test_reference_solver COMMAND test_reference_solver)
set_tests_properties(test_reference_solver PROPERTIES TIMEOUT 900)
add_test(NAME test_asymmetry_estimator COMMAND test_asymmetry_estimator)
set_tests_properties(test_asymmetry_estimator PROPERTIES TIMEOUT 900)
add_test(NAME test_ritz_spectrum COMMAND test_ritz_spectrum)
set_tests_properties(test_ritz_spectrum PROPERTIES TIMEOUT 900)
add_test(NAME test_sauter_schwab_coverage COMMAND test_sauter_schwab_coverage)
set_tests_properties(test_sauter_schwab_coverage PROPERTIES TIMEOUT 900)
add_test(NAME test_adjacent_bem COMMAND test_adjacent_bem)
set_tests_properties(test_adjacent_bem PROPERTIES TIMEOUT 900)
add_test(NAME test_amg COMMAND test_amg)
set_tests_properties(test_amg PROPERTIES TIMEOUT 900)
add_test(NAME test_ams_preconditioner COMMAND test_ams_preconditioner)
set_tests_properties(test_ams_preconditioner PROPERTIES LABELS "host" TIMEOUT 900)
add_test(NAME test_discrete_gradient COMMAND test_discrete_gradient)
set_tests_properties(test_discrete_gradient PROPERTIES TIMEOUT 900)
add_test(NAME test_biot_savart COMMAND test_biot_savart)
set_tests_properties(test_biot_savart PROPERTIES TIMEOUT 900)
add_test(NAME test_mesh_refinement COMMAND test_mesh_refinement)
set_tests_properties(test_mesh_refinement PROPERTIES TIMEOUT 900)

# Baut genau die Tests mit der Marke "host".
add_custom_target(host_tests DEPENDS
    test_harmonic_dirichlet_split
    test_ams_preconditioner
    test_nedelec_orientation
    test_edge_sign_convention
    test_curl_gradient_nullspace
    test_coupling_sign_convention
    test_block_triangular_consistency
    test_norm_nan_propagation
    test_dgemv_beta_zero
    test_singular_quadrature
    test_ams_scalar_masking
    test_gbem_symmetry
    test_amg_convergence
    test_alpha_poisson
    test_amg_anisotropy
    test_preconditioner_end_to_end
    test_solver_nonfinite_abort
    test_harmonic_inner_direct
    test_apply_many
    test_dense_direct
    test_harmonic_phase_average
    test_block_preconditioner_study
    test_nedelec_prism
    test_nedelec_tet
    test_mixed_topology
    test_gauge_on_tets
    test_source_projection
    test_patch_assembly
    test_dirichlet_bc
    test_element_dispatch
    test_piola_production
    test_block_system
    test_sparse_direct
    test_fem_direct_preflight
)


# ----------------------------------------------------------------------------
# Python-Werkzeuge (tools/): nur registrieren, wenn Python und die Pakete
# vorhanden sind. Ohne sie bleiben die C++-Tests unveraendert.
#
#   ctest --test-dir build -L tools     # Pruefmodule (braucht numpy)
#   ctest --test-dir build -L gmsh      # Netzgeneratoren und Import (braucht gmsh)
#
# Ein anderes Python (z. B. eine virtuelle Umgebung mit gmsh) waehlt
#   -DPython3_EXECUTABLE=~/venv-gmsh/bin/python
# ----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
set(_fembem_py_numpy 1)
set(_fembem_py_gmsh 1)
if (Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                    RESULT_VARIABLE _fembem_py_numpy OUTPUT_QUIET ERROR_QUIET)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import gmsh"
                    RESULT_VARIABLE _fembem_py_gmsh OUTPUT_QUIET ERROR_QUIET)
endif()

set(_tools ${CMAKE_CURRENT_SOURCE_DIR}/tools)
if (_fembem_py_numpy EQUAL 0)
    add_test(NAME tools_meshcheck_tests
             COMMAND ${Python3_EXECUTABLE} ${_tools}/tests/test_meshcheck.py)
    add_test(NAME tools_check_map_tests
             COMMAND ${Python3_EXECUTABLE} ${_tools}/tests/test_check_map.py)
    add_test(NAME tools_standard_mesh
             COMMAND ${Python3_EXECUTABLE} ${_tools}/meshcheck.py
                     ${CMAKE_CURRENT_SOURCE_DIR}/data/team28/team28.vtk
                     --require-hex --separate-bodies)
    add_test(NAME tools_team28_axisymmetric_tests
             COMMAND ${Python3_EXECUTABLE} ${_tools}/tests/test_team28_axisymmetric.py)
    set_tests_properties(tools_meshcheck_tests tools_check_map_tests tools_standard_mesh
                         tools_team28_axisymmetric_tests
                         PROPERTIES LABELS "tools" TIMEOUT 120)
    message(STATUS "FEMBEM: Tests tools  aktiv (${Python3_EXECUTABLE})")
else()
    message(STATUS "FEMBEM: Tests tools  aus (Python mit numpy nicht gefunden)")
endif()

if (_fembem_py_numpy EQUAL 0 AND _fembem_py_gmsh EQUAL 0)
    # Jeder Lauf prueft das erzeugte Netz selbst und endet bei einem Fehler
    # mit 1 (tools/gmsh/team28_hex.py).
    add_test(NAME gmsh_team28_hex
             COMMAND ${Python3_EXECUTABLE} ${_tools}/gmsh/team28_hex.py -o gmsh_team28_hex.vtk)
    add_test(NAME gmsh_team28_hex_coarse
             COMMAND ${Python3_EXECUTABLE} ${_tools}/gmsh/team28_hex.py --scale 0.5
                     -o gmsh_team28_hex_coarse.vtk)
    add_test(NAME gmsh_team28_hex_variant
             COMMAND ${Python3_EXECUTABLE} ${_tools}/gmsh/team28_hex.py --n-quarter 9
                     --plate-grading 0.85 --plate-z0 9.8 -o gmsh_team28_hex_variant.vtk)
    add_test(NAME gmsh_team28_fem_air_coarse
             COMMAND ${Python3_EXECUTABLE} ${_tools}/gmsh/team28_fem_air.py --scale 0.5
                     -o gmsh_team28_fem_air_coarse.vtk)
    add_test(NAME gmsh_team28_fem_air_truncation2
             COMMAND ${Python3_EXECUTABLE} ${_tools}/gmsh/team28_fem_air.py --scale 0.5
                     --truncation 2 --plate-z0 11.3 -o gmsh_team28_fem_air_t2.vtk)
    # Import eigener .msh/.geo (tools/gmsh/import_msh.py): Rundreise, Fehlerfaelle,
    # Prismen-Orientierung mit Kontrollfall ohne Drehung.
    add_test(NAME gmsh_import_msh_tests
             COMMAND ${Python3_EXECUTABLE} ${_tools}/tests/test_import_msh.py)
    set_tests_properties(gmsh_team28_hex gmsh_team28_hex_coarse gmsh_team28_hex_variant
                         gmsh_team28_fem_air_coarse gmsh_team28_fem_air_truncation2
                         gmsh_import_msh_tests
                         PROPERTIES LABELS "gmsh" TIMEOUT 300)
    message(STATUS "FEMBEM: Tests gmsh   aktiv")
else()
    message(STATUS "FEMBEM: Tests gmsh   aus (Python-Paket gmsh nicht gefunden; "
                   "-DPython3_EXECUTABLE=<python mit gmsh>)")
endif()

endif()  # FEMBEM_BUILD_TESTS (CTest)
