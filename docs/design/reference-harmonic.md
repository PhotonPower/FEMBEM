# Referenzlöser im Frequenzbereich

Stand 2026-11-02. Gerechnet auf einer NVIDIA GeForce RTX 3090 (sm_86, 24 GB),
Windows 11, CUDA 13.4. Programm: `main` 0c7c8c1 plus die Änderungen dieses
Berichts (noch nicht eingecheckt). Logs in `build/refharm/` (nicht im Repo).

## Frage

Der Referenzlöser (`--reference`: Q1-Knotenelemente, Strafterm-Eichung,
vollbesetzte BEM) integriert transient mit implizitem Euler, dt = 0,1 ms. Er
liegt bei 11,3 mm etwa 3 % unter dem Ringmodell
(`tools/team28_axisymmetric.py`). Gesucht war ein Pfad ohne Zeitfehler, der
dieselbe Gleichung löst wie FEM-direkt und der harmonische BEM-FEM-Pfad.

## Methode

Aufruf: `fembem --reference --harmonic[=HZ] --fixed-plate [--plate-z=MM]`.

**Gleichung.** Mit A(t) = Re(Â e^{jωt}), Â = A_r + jA_i, wird aus
(K+G)A + C dA/dt = b(t) der reelle 2n-Block

    [ K+G   −ωC  ] [A_r]   [b]
    [ ωC    K+G  ] [A_i] = [0]

G = −(1/µ₀)·T·D ist der DtN-Block aus `ReferenceOperator::apply`, C = σ·M die
Leitermassenmatrix. K und W = K + ωC kommen aus `reference_assemble` mit
dt = 1e30 bzw. dt = 1/ω auf demselben CSR-Muster; C = (W − K)/ω. Die
atomare Streuung auf der GPU summiert in nicht festgelegter Reihenfolge,
W − K enthält deshalb außerhalb der Leiter Rundungsrauschen (gemessen
≤ 2,3e-14 gegen max|C| ≈ 2). Einträge mit |W − K| ≤ 1e-12·|K| werden auf
null gesetzt und gezählt; echte Leitwerteinträge liegen zehn
Größenordnungen darüber.

**Quelle.** Spulen bei t = T/4 ihrer eigenen Frequenz (50 Hz), also im
Scheitelwert 20 A, reelle rechte Seite, b_i = 0. Das entspricht
i(t) = I cos(ωt) und legt nur die globale Phase fest. Die Amplitude hängt
nicht von `--harmonic=HZ` ab.

**Löser.** FGMRES(30), rechtsvorkonditioniert, Toleranz 1e-8 am echten
Residuum, höchstens 200 Iterationen. Vorkonditionierer diag(M, M) mit
M = K + G + ωC; das ist genau die Matrix eines transienten Schritts mit
dt = 1/ω, M⁻¹ deshalb über den vorhandenen `reference_gmres` mit ILU(0)
(innere Toleranz 1e-6). `reference_gmres` hat dafür drei optionale
Parameter (Toleranz, Deckel, Ausgabe) bekommen; die Vorgaben sind die
bisherigen Konstanten, der transiente Pfad ruft unverändert auf. Protokoll
je äußerer Iteration: geschätztes Residuum und innere Iterationen beider
Blöcke, je Restart das echte Residuum.

Zum Spektrum: für symmetrisches K+G > 0 und C ≥ 0 hat P⁻¹A die Eigenwerte
(1 ± jµ)/(1 + µ), µ die verallgemeinerten Eigenwerte von ωC v = µ(K+G) v.
Sie liegen auf den Strecken Re + |Im| = 1 von 1 nach ±j. Netzunabhängig,
aber kein Haufen: mit wachsendem ωτ reicht das Spektrum näher an ±j.

**Kraft.** `nodal_lorentz_harmonic` (NodalMaxwell.cuh): je Hexaeder und
Gauß-Punkt f = ½σω(A_i × rot A_r − A_r × rot A_i), Kraft und Moment um den
Schwerpunkt, Quadraturordnung 2 und 3. Beitragen nur Elemente mit κ > 0 und
gesetzter Plattenmaske. Dazu die Wirbelstromdiagnose aus `nodal_eddy_diag`:
Verlustleistung ½(P(A_r) + P(A_i)) und Q_eff = |m_z(A_i)|/|m_z(A_r)| (Moment
im Strommaximum gegen Nulldurchgang, wie `[Ref-Q]` der Zeitschleife).

**Abgrenzung.** Kein Code aus `HarmonicSolver` oder `HarmonicBlockSystem`.
Optionen des Nédélec-Pfads (`--harmonic-outer-restart`, `-inner-tol`,
`-outer-cap`, `-checks`, `-inner-direct`, `-field-dump`) sowie
`--fem-direct*`, `--direct`, `--pc-study`, `--harmonic-study` werden mit
`--reference --harmonic` abgewiesen (Exit-Code 1), ebenso ein fehlendes
`--fixed-plate`.

## Tests

`tests/test_reference_harmonic.cu` (CTest-Marke `host`, < 1 s):

1. Kraftkernel analytisch (uniformes B_r, konstantes A_i): exakt auf 1e-16.
2. Kraftkernel gegen das Zyklusmittel des transienten `nodal_lorentz` auf
   einem verzerrten Hexaeder mit Zufallsfeldern, Kraft und Moment,
   Ordnung 2 und 3: 7e-8 (Restfehler des Differenzenquotienten).
   Gegenprobe: vertauschte A_r/A_i weicht um 200 % ab.
3. Phaseninvarianz (Kraft und Moment auf 4e-16), A_i ∥ A_r ergibt null,
   Maske und κ = 0. Gegenprobe: die Variante mit falschem Vorzeichen ist
   nicht phaseninvariant (126 %).
4. Blockoperator gegen das komplexe Produkt (2e-16). Gegenprobe: ω → −ω
   wird erkannt.
5. FGMRES gegen die komplexe Direktlösung (n = 40, |A_i|/|A_r| = 0,47):
   exakter innerer Löser 28 Iterationen gegen 77 ohne Vorkonditionierer;
   wechselnder ungenauer innerer Löser (1 bis 3 Jacobi-Schritte) konvergiert
   ebenfalls auf 6e-12. σ = 0: eine Iteration, A_i exakt null.
   Gegenprobe: falsches Kopplungsvorzeichen liefert eine andere Lösung (94 %).

`ctest -L host`: 35 von 35 bestanden. `test_reference_solver` (ohne Marke)
besteht ebenfalls.

## Messungen

Standardnetz `team28.vtk` (4805 Hexaeder, 21 039 Unbekannte, 4138
Randknoten) und grobes Netz `gmsh_team28_hex_coarse.vtk` (696 Hexaeder,
3681 Unbekannte, 938 Randknoten).

### a) σ = 0

`--reference --harmonic --fixed-plate --verify-field`, team28.vtk bei
3,8 mm (Netzlage), wie der transiente Feldtest in
[bemfem-field-check.md](bemfem-field-check.md), Befund 5:

| z [mm] | B_z harmonisch, σ = 0 [T] | B_z transient [T] | Biot-Savart [T] |
|---|---|---|---|
| 4,3 | 9,377921e-02 | 9,377921e-02 | 9,383920e-02 |
| 5,3 | 8,972183e-02 | 8,972183e-02 | 8,984550e-02 |
| 6,3 | 8,570678e-02 | 8,570678e-02 | 8,589093e-02 |

max|A_i| = 0 exakt, C = 0 (104 Rauschanteile ≤ 1,2e-14 verworfen).
FGMRES 2 Iterationen bis 1,7e-12 (die erste endet bei 1,8e-6, der
inneren Toleranz), 16 s. Die transiente Gleichstromprobe braucht dafür
1000 Zeitschritte. Die Felder stimmen auf alle sieben gedruckten Stellen
überein. Auf dem groben Netz ebenso (B_z = 9,381946e-02 und 8,748338e-02 T
in beiden Pfaden, +1,06 % bzw. +0,70 % gegen Biot-Savart); der transiente
`--reference --verify-field` läuft dort unverändert durch (589 s).

### b) σ klein: F_z ∝ σ²

Im System und in der Kraft stehen σ und ω nur als Produkt σω, eine kleinere
Frequenz ist also exakt eine kleinere Leitfähigkeit. team28.vtk, 11,3 mm:

| f | entspricht | F_z | Iterationen außen / innen | Zeit |
|---|---|---|---|---|
| 1 Hz | σ/50 | 4,43468e-04 N | 4 / 196 | 35 s |
| 0,1 Hz | σ/500 | 4,43499e-06 N | 3 / 140 | 26 s |

Verhältnis 99,993 (Soll 100; das Ringmodell prüft dasselbe mit
Toleranz ±1). Q_eff 0,0139 bzw. 0,0014, also ebenfalls linear in σω.

### c) Transient gegen harmonisch

Grobes Netz, 11,3 mm, `--fixed-plate`, je drei Perioden; Zyklusmittel über
die letzte Periode (nach der zweiten Periode auf vier Stellen stabil).

| | dt | F_z | Abstand zu harmonisch | Q_eff | Schritte | Zeit |
|---|---|---|---|---|---|---|
| transient | 0,1 ms | 0,8844 N | +0,0311 N (+3,6 %) | 0,8284 | 600 | 296 s |
| transient | 0,05 ms | 0,8690 N | +0,0157 N (+1,8 %) | 0,8176 | 1200 | 571 s |
| transient | 0,025 ms | 0,8612 N | +0,0079 N (+0,9 %) | 0,8122 | 2400 | 1146 s |
| harmonisch | — | 0,8533 N | — | 0,8069 | 9 / 288 | 5 s |

Der Abstand halbiert sich mit dt (Faktor 1,98 und 1,99): implizites Euler,
erste Ordnung, und der Grenzwert ist der harmonische Wert. Richardson aus
dt/2 und dt/4: 2·0,8612 − 0,8690 = 0,8534 N gegen 0,8533 N. Q_eff verhält
sich genauso (Abstände 0,0215 / 0,0107 / 0,0053).

Der Zeitfehler des transienten Pfads hebt die Kraft also an: bei
dt = 0,1 ms um 3,6 %.

### d) Vergleich mit dem Ringmodell

team28.vtk, 50 Hz. Ringmodell: Grenzwert aus
[gmsh-phase9-report.md](gmsh-phase9-report.md).

| Höhe | harmonisch F_z | F_z/(m·g) | Ringmodell | Abweichung | außen / innen | Zeit |
|---|---|---|---|---|---|---|
| 11,3 mm | 0,9436 N | 0,8989 | 1,00684 N | −6,3 % | 11 / 518 | 86 s |
| 3,8 mm | 3,1277 N | 2,9797 | 3,28965 N | −4,9 % | 11 / 638 | 110 s |

Weitere Ausgaben: Quadratur 2×2×2 gegen 3×3×3 gleich auf 0,000 %, F_x und
F_y ≤ 1,6e-4 N, Moment ≤ 2,9e-3 Nm. Q_eff 0,784 (11,3 mm) und 0,588
(3,8 mm); die Zeitschleife meldete früher 0,80 und 0,60.

Der transiente Referenzpfad liegt bei 11,3 mm mit dt = 0,1 ms bei
0,930·m·g = 0,976 N (Wert aus dem Quelltextkommentar der Zeitschleife,
in dieser Arbeit nicht neu gerechnet), also 3,4 % über dem harmonischen
Wert -- dieselbe Richtung und Größe wie auf dem groben Netz unter c).

## Laufzeiten

FGMRES ist der ganze Aufwand: 81 s von 86 s bei 11,3 mm. Eine innere
Iteration kostet rund 0,15 s, weil die ILU-Substitution auf dem Host läuft
(siehe ReferenceSolver.cuh). BEM 1,6 s, LU von V 2,8 s, Assemblierung 0,3 s.
Zum Vergleich der transiente Pfad: rund 9 s je Zeitschritt, also etwa
1,5 Stunden für drei Perioden.

## Offene Punkte

- **Der Abstand zum Ringmodell wird größer, nicht kleiner.** Ohne Zeitfehler
  liegt der Referenzpfad 6,3 % (11,3 mm) bzw. 4,9 % (3,8 mm) unter dem
  Ringmodell. Die 3 % des transienten Pfads waren also zum Teil
  Zeitfehler, der in die günstige Richtung zeigte. Was bleibt, ist ein
  Diskretisierungsfehler des Referenzpfads (stetige Neumann-Spur an
  Kanten, dokumentiert in ReferenceSolver.cuh; Netz; Strafterm-Eichung in
  der Platte). Eine Netzfolge mit `--refine=1` (16 542 Randknoten,
  6,1 GiB dichte Algebra) ist nicht gerechnet.
- Der transiente Wert auf dem Standardnetz bei dt und dt/2 ist nicht neu
  gerechnet (etwa 1,5 und 3 Stunden). Befehl für den Nutzer:
  `fembem 600 --reference --fixed-plate --plate-z=11.3 --dt=1e-4` und
  `fembem 1200 --reference --fixed-plate --plate-z=11.3 --dt=5e-5`.
- Der Maxwell-Spannungstensor wird im harmonischen Pfad nicht ausgewertet
  (er taugt für den Gleichanteil nicht, siehe Zeitschleife).
- diag(M, M) bleibt bei großem ωτ nicht beschränkt in der Iterationszahl
  (Spektrum nahe ±j). Bei 50 Hz genügen 11 äußere Iterationen; eine
  Alternative mit reellem Spektrum in [½, 1] wäre PRESB.
