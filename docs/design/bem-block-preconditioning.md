# Vorkonditionierung des BEM-Blocks — Entscheidungsgrundlage und Bauplan

`docs/reports/convergence-findings.tex` nennt den BEM-Block den „Elefanten im Raum": `G_BEM` geht mit
der Identität in den Löser, und die blockweise Residuumsaufschlüsselung zeigt
ihn mit `1,8e-04` gegen `1,3e-04` des FEM-Blocks als den *größeren* Anteil.
Daraus wurde der Vorschlag abgeleitet, einen Calderón-Vorkonditionierer oder
eine H-Matrix-Näherungsinverse für `G_BEM` zu bauen.

Dieses Dokument hält fest, warum vor dem Bau eine Messung steht, wie sie
aussieht, und was — je nach Ergebnis — danach zu tun ist.

---

## 1. Warum nicht direkt bauen

### 1.1 Die Begründung misst die falsche Größe

Die blockweise Residuumsaufschlüsselung beantwortet die Frage **wo sitzt das
Residuum**. Die Frage, die über den Vorkonditionierer entscheidet, ist aber
**welcher Block verursacht die Kondition**. Das sind zwei verschiedene Größen,
und sie fallen nicht zusammen: ein perfekt konditionierter Block trägt den
größeren Residuumsanteil, wenn die rechte Seite dort liegt.

Aus `1,8e-04 > 1,3e-04` folgt also nichts über `κ₂(G_BEM)`.

### 1.2 Die vorhandenen Daten sprechen bereits dagegen

Die Δt-Tabelle aus `convergence-findings.tex`, Abschnitt „Konditionszahl über die
Zeitschrittweite":

| Δt | σ_max | σ_min | κ₂ |
|---|---|---|---|
| 1e-6 | 155 | 7,56e-05 | 2,05e+06 |
| 1e-5 | 155 | 1,50e-04 | 1,03e+06 |
| 1e-4 | 155 | 1,45e-04 | 1,07e+06 |
| 1e-3 | 155 | 5,89e-05 | 2,63e+06 |
| 1e-2 | 155 | 7,25e-06 | 2,14e+07 |

`σ_max` ist über fünf Dekaden in Δt **konstant**. `σ_min` bewegt sich um
Faktor 20.

Der äquilibrierte BEM-Block ist aber **Δt-invariant**: `D_K` wird aus
`diag(μ₀ G)` gebildet, und `G` enthält keinen Zeitschritt. Der Massenterm
`σ/Δt` steht ausschließlich im FEM-Block `A_sys = K + C/Δt`.

Eine Größe, die sich mit Δt um Faktor 20 bewegt, kann damit kein Singulärwert
des reinen BEM-Blocks sein. Sie kommt aus dem FEM-Block oder aus der Kopplung
(deren Skalierung über `D_A` Δt-abhängig ist) — beides nicht aus `A22`.

Das ist keine neue Messung. Es ist eine Schlussfolgerung aus einer Tabelle,
die seit dem Δt-Durchlauf im Bericht steht.

### 1.3 Die Größenordnung passt nicht

Für den Single-Layer-Operator auf einer geschlossenen Fläche wächst die
Konditionszahl der Galerkin-Matrix nach Jacobi-Skalierung wie `O(h⁻¹)`, also
`O(√N)` in 3D. Bei `N = 16 544` BEM-Freiheitsgraden sind das
Größenordnung `10²`, großzügig gerechnet `10³`.

Gemessen wurde für das Gesamtsystem `κ₂ = 1,07e+06`. Zwischen beiden liegen
drei bis vier Dekaden. Selbst wenn ein Calderón-Vorkonditionierer `κ₂(A22)`
auf `O(1)` drückte — was er in der Theorie tut, denn `W V = ¼I − K'²` ist ein
Operator zweiter Art —, nähme er dem Gesamtsystem nur das weg, was in `A22`
überhaupt steckt.

### 1.4 Der Aufwand ist erheblich

**Calderón (Operator-Preconditioning).** Die Calderón-Identität
`W V = ¼I − K'²` ist die richtige Grundlage, aber die diskrete Umsetzung ist
nicht der hypersinguläre Operator auf demselben Netz. `V` wird auf
stückweise konstanten Funktionen diskretisiert, `W` braucht stetige; auf
demselben Netz ist die Paarung nicht stabil (inf-sup verletzt). Notwendig ist
ein **duales Netz** — das barizentrisch verfeinerte Oberflächennetz — mit
**Buffa-Christiansen-Basisfunktionen**, plus die Massenmatrix, die zwischen
Primal- und Dualraum vermittelt. Das ist ein zweiter vollständiger
BEM-Assemblierungspfad: Netzverfeinerung auf der Oberfläche, neue
Basisfunktionen, neuer singulärer Quadraturpfad für den hypersingulären Kern,
neue ACA-Blockstruktur.

**H-Matrix-Näherungsinverse.** Braucht Addition, Multiplikation, LU und
Inversion auf der Blockstruktur des Clusterbaums, jeweils mit
Rangkürzung. `src/ClusterTree.cpp` und `src/AcaPlus.cpp` liefern die
Blockstruktur, aber keine H-Arithmetik.

Beides sind Wochen. Nach der Regel, die der Bericht selbst aufstellt — *jede
Vermutung zieht eine Messung nach sich, die sie widerlegen kann* — steht davor
eine Messung.

---

## 2. Die Messung: `--pc-study`

    ./fembem --cpu --mesh=team28_coarse_12.vtk --pc-study

`--pc-study` schaltet `--direct` mit ein und arbeitet auf der Matrix, die
`--direct` ohnehin schon spaltenweise abtastet. Es entsteht **kein zweiter
Abtastlauf**.

### 2.1 Was ausgegeben wird

**Erstens** `κ₂` der beiden Diagonalblöcke **getrennt**, neben `κ₂` des
Gesamtsystems. Das beantwortet 1.1 direkt.

**Zweitens** GMRES-Iterationszahlen unter fünf idealisierten
Vorkonditionierern (`n_A` = FEM-Zeilen, `n_K` = BEM-Zeilen):

| # | `M` | wozu |
|---|---|---|
| 0 | `I` | Bezugsgröße |
| 1 | `blockdiag(A11, I)` | Decke für AMS/ILU auf dem FEM-Block |
| 2 | `blockdiag(I, A22)` | **Decke für Calderón / H-Matrix** |
| 3 | `blockdiag(A11, A22)` | beide Blöcke exakt |
| 4 | `[[A11, A12],[0, S]]`, `S = A22 − A21 A11⁻¹ A12` | exaktes Schur-Komplement |

Und, sofern `n ≤ 6000`, zusätzlich `κ₂(M⁻¹A)` je Variante. Beide Kennzahlen,
weil sie verschiedene Fehler machen: `κ₂` sagt für unsymmetrische Matrizen nur
grob etwas über die Krylov-Konvergenz (die hängt am Spektrum, nicht an den
Singulärwerten), und die Iterationszahl hängt an der rechten Seite. Wo beide
dasselbe sagen, ist der Befund belastbar.

### 2.2 Die entscheidende Zahl

Variante 2 gegen Variante 0 ist die **Decke**:

> Calderón, H-Matrix-Näherungsinverse, Nahfeld-ILU, Block-Jacobi — sie alle
> sind Näherungen an `A22⁻¹`. Keine kann besser sein als `A22⁻¹` selbst.

Bringt die exakte Inverse nichts, ist der gesamte Ansatz widerlegt, endgültig,
ohne dass eine Zeile Calderón-Code geschrieben werden muss. Bringt sie viel,
ist der Aufwand belegt gerechtfertigt — und man kennt sogar den Zielwert, den
man anstreben kann.

### 2.3 Die zweite Frage: war die Schur-Idee falsch oder nur die Näherung?

Variante 4 gegen Variante 3. Der Bericht hat `S̃ = μ₀ G_BEM` widerlegt (inneres
CG divergiert, mehr Iterationen verschlechtern das Ergebnis). Offen bleibt
dabei, ob die **Schur-Idee** falsch war oder nur die **Näherung**, die den Term
`A21 A11⁻¹ A12` vollständig weglässt. Das sind sehr verschiedene Befunde: der
erste beendet den Ansatz, der zweite verlagert ihn auf die Frage, wie dieser
Term zu nähern ist — und dann wäre `A22` gar nicht der Ort, an dem gearbeitet
werden muss.

---

## 3. Selbstprüfung des Werkzeugs

`tests/test_block_preconditioner_study.cpp`, Marke `host`, ohne CUDA baubar:

    cmake --build build --target test_block_preconditioner_study
    ctest --test-dir build -R test_block_preconditioner_study --output-on-failure

Oder ganz ohne CMake:

    g++ -std=c++17 -O2 -Iinclude tests/test_block_preconditioner_study.cpp \
        src/BlockPreconditionerStudy.cpp -o t && ./t

`convergence-findings.tex` hält fest, dass sechs Diagnosekennzahlen des Projekts
miskalibriert waren und die Suche zeitweise in die Irre geführt haben. Ein
auffälliges Messergebnis ist ohne Gegenprobe nicht von einem Werkzeugfehler zu
unterscheiden. Geprüft wird deshalb in **beide** Richtungen:

| Test | prüft | Stand |
|---|---|---|
| 1 | `HostDenseLU`: `solve()` und `solve_transposed()` gegen Referenz | `4,4e-16` / `5,1e-16` |
| 1 | Pivot `1e-300` wird als singulär erkannt (nicht „ungleich null") | grün |
| 1 | `NaN` in der Matrix wird erkannt (Zählen statt Vergleichen) | grün |
| 2 | `κ₂` an Matrizen vorgegebener Kondition `1e3`/`1e6`/`1e9` | auf `~1e-10 %` |
| 3 | schlechte Kondition **bewusst im FEM-Block** → Decke muss klein sein | gemeldet: `1,10` |
| 4 | Gegenprobe, schlechte Kondition **im BEM-Block** → Decke muss groß sein | gemeldet: `5,11` |
| 5 | exaktes Schur-Komplement löst in wenigen Iterationen | **2** Iterationen |

Test 4 ist der Punkt: **ohne ihn bestünde auch ein Werkzeug, das grundsätzlich
„bringt nichts" sagt, den Test 3.** Genau diese Sorte Gegenprobe fehlte bei den
sechs miskalibrierten Kennzahlen.

Test 5 misst zugleich das GMRES mit: rechts-vorkonditioniertes GMRES mit
exaktem Schur-Komplement terminiert theoretisch nach 2 Iterationen, weil das
Minimalpolynom von `A M⁻¹` Grad 2 hat. Gemessen: 2. Ein GMRES mit einem Fehler
in der Arnoldi-Rekursion oder in der Rückwärtssubstitution träfe diese Zahl
nicht.

### Warum reiner Host-Code

`src/BlockPreconditionerStudy.cpp` hat **keine CUDA-, cuBLAS- oder
BLAS-Abhängigkeit**. Das ist Absicht: die Frage, ob sich der Aufwand für
Calderón lohnt, soll beantwortbar sein, ohne erst eine GPU-Umgebung
aufzusetzen. Zugleich ist die Datei damit direkt testbar, und der Eingriff in
`fem.cu` beschränkt sich auf den Aufruf.

Die Fallen aus `docs/design/nan-and-thresholds.md` sind eingearbeitet: die
Singularitätsprüfung ist **relativ zur größten Matrixnorm** und schließt
Endlichkeit ein; die Norm **zählt** nicht endliche Einträge, statt sie zu
vergleichen; die GMRES-Rückwärtssubstitution hat eine relative Pivotschranke.

---

## 3a. Ergebnis der Messung (team28_coarse_12, 4511 x 4511)

**Calderón und H-Matrix-Näherungsinverse sind widerlegt.**

| | σ_max | σ_min | κ₂ |
|---|---|---|---|
| System | 155 | 1,45e-4 | 1,07e+06 |
| A11 (FEM) | 4,16 | 1,56e-4 | 2,66e+04 |
| A22 (BEM) | 60,6 | 6,35e-3 | **9,55e+03** |

Der BEM-Block ist der **besser konditionierte von beiden**. Die exakte Inverse
von A22 macht den Löser sogar schlechter (0,0341 gegen 0,0204). Da Calderón und
H-Matrix-Inverse nur Näherungen an genau diese exakte Inverse sind, ist der
Ansatz erledigt. Abschnitt 1.2 hat das aus der Δt-Tabelle vorhergesagt; die
Messung bestätigt es.

### Der eigentliche Befund: die Kopplung

σ_max des Systems (155) ist **größer als σ_max beider Diagonalblöcke** (4,16
und 60,6). Die Norm des Operators kommt aus den Kopplungsblöcken. Entsprechend:

| Variante | rel. Res nach 4000 Iter | gegen `M = I` |
|---|---|---|
| 3 blockdiag(A11,A22) | 0,173 | 0,12× — schlechter |
| 1 blockdiag(A11,I) | 0,0442 | 0,46× — schlechter |
| 2 blockdiag(I,A22) | 0,0341 | 0,60× — schlechter |
| 0 M = I | 0,0204 | 1× |
| 5 blocktri, S̃ = A22 | 0,0101 | 2× |
| 6 blocktri, S_diag | 8,73e-4 | 23× |
| 4 blocktri, S exakt | 1,09e-11 | **2 Iterationen** |

**Blockdiagonal ist durchweg schlechter als gar nichts, blocktriangulär
durchweg besser, monoton in der Güte von S.** Das erklärt Befund 5 aus
`convergence-findings.tex`: vier Verbesserungen des FEM-Block-Vorkonditionierers bewegten den
äußeren Löser nicht — sie konnten nicht.

### Warum S̃ = μ₀G_BEM scheitern musste

- `‖A21 A11⁻¹ A12‖_F / ‖S‖_F = 1` — der weggelassene Term *ist* S. A22 (Norm
  60,6) ist gegen S (Norm ~5,3e4) um **Faktor 871** zu klein.
- S ist **indefinit**: Eigenwerte des symmetrischen Anteils
  [−5,28e+04, +2,34e+03]. **CG war dort nie zulässig**, unabhängig von der
  Näherung.
- κ₂(S) = 2e+08, also 200× schlechter als das System selbst.

### Die Wurzel: σ_min(A11)

σ_min(A11) = 1,56e-4 *ist* σ_min des Systems (1,45e-4), und A11⁻¹ trägt diesen
Fastnullraum in S hinein. Die 12 kleinsten Singulärwerte von A11:

    1,56e-4  1,78e-4  2,24e-4  2,56e-4  6,45e-4  7,06e-4
    9,36e-4  1,40e-3  1,46e-3  1,59e-3  2,01e-3  2,30e-3

Kein Wert setzt sich um mehr als Faktor 2,5 ab, Spanne 14,7 über 12 Werte —
ein **Kontinuum, kein abgesetzter Einzelmodus**. Die Eichung ist damit *nicht*
das Problem; der Fastnullraum ist physikalisch (nichtleitende Gebiete ohne
σ/Δt-Term). Er lässt sich nicht wegeichen, er muss behandelt werden.

### Was daraus folgt

Der blocktriangulare Zweig ist der richtige Weg, und drei Dinge daran sind
jetzt konkret:

1. `apply_schur_tilde()` muss den Kopplungsterm bekommen:
   `S v = A22 v − A21 (Ã11⁻¹ (A12 v))`. Alle drei Bausteine existieren.
2. Das innere CG muss weg — S ist gemessen indefinit. GMRES oder BiCGStab.
3. Offen bleibt allein, **wie genau Ã11⁻¹ sein muss**. Dafür die
   Genauigkeitsreihe (Varianten 7.x).

---

## 3b. Vollständige Messkette (team28_coarse_12)

### Der Strukturbefund

Blockdiagonal ist **durchweg schlechter als gar nichts**, blocktriangulär
durchweg besser, monoton in der Güte des Schur-Komplements. σ_max des Systems
(155) ist größer als σ_max beider Diagonalblöcke (4,16 und 60,6) — die Norm des
Operators kommt aus der **Kopplung**, und die sieht ein blockdiagonaler
Vorkonditionierer bauartbedingt nicht.

### Beide Eliminationsrichtungen

| | κ₂ des Schur-Komplements | definit? |
|---|---|---|
| über A11 → S | 2,0e+08 | indefinit |
| über A22 → S_A (DtN) | 1,03e+08 | indefinit |

DtN gewinnt **alle fünf** Genauigkeitsstufen, exakt gelöst in **einer**
Iteration. Bei ε = 1e-3: 29 Iterationen gegen 1181.

### Genauigkeitsanforderung

| | trägt ab | äußere Iterationen |
|---|---|---|
| nur A22⁻¹ ungenau | 1e-3 | 29 |
| nur S_A-Löser ungenau | 1e-4 | 127 |
| **beide ungenau** | **1e-4** | **127** |

Reihe 11 folgt Reihe 10 Zeile für Zeile — der **S_A-Löser ist der Flaschenhals**,
A22⁻¹ ist daneben praktisch gratis.

### Der Preis

Unvorkonditionierte Krylov-Schritte bis 1e-4: **S_A 1132, A22 469**. Eine
S_A-Anwendung kostet eine A22-Lösung, also

    127 × 1132 × 469 = 6,74e+07 BEM-Matvecs je Zeitschritt

gegen 4000 des heutigen Lösers — **Faktor 1,69e+04**.

Break-even verlangt, dass beide inneren Löser zusammen diesen Faktor drücken,
also je rund **Faktor 130**: S_A von 1132 auf 9 Schritte, A22 von 469 auf 4.
Das ist mit keinem bekannten Vorkonditionierer zu haben.

Realistisch:

| innere Löser | Matvecs | gegen heute |
|---|---|---|
| AMS auf S_A ~40, Calderón/H-LU auf A22 ~25 | 1,27e+05 | 32× |
| sehr gut: 20 / 15 | 3,81e+04 | 10× |

Der Handel lautet also: **10- bis 30-facher Aufwand für eine konvergierte
Lösung statt einer bei 2,6e-2 stagnierenden.** Ob das ein guter Handel ist,
entscheidet sich nicht am Löser, sondern an der geforderten Genauigkeit der
Kraftwerte.

### Grenze dieser Zahl: das Störungsmodell

Die Reihen 7 bis 11 modellieren einen ungenauen inneren Löser als **zufällig
gestörte Matrix**. Das ist linear (und damit für GMRES zulässig, siehe
Abschnitt 5), aber es ist ein pessimistisches und womöglich unrepräsentatives
Modell:

Ein echter Krylov-Löser lässt keinen zufälligen Restfehler zurück, sondern
einen **strukturierten** — er erfasst die dominanten Spektralanteile zuerst und
lässt gerade das untere Spektralende stehen. Zufälliges Rauschen trifft alle
Richtungen gleichmäßig, also auch die, die ein Krylov-Verfahren gut trifft.

Die Anforderung „1e-4" — und damit die 127 äußeren Iterationen und der Faktor
1,69e+04 — **könnte ein Artefakt dieses Modells sein**. Ein faithfulleres
Modell wäre ein fester Spektralfilter (Abschneiden der kleinsten
Singulärrichtungen von S_A): ebenfalls linear, aber genau das, was ein früh
abgebrochenes Krylov-Verfahren tatsächlich tut. Solange das nicht gemessen ist,
ist der Faktor 1,69e+04 eine **obere Schranke mit Modellunsicherheit**, kein
belegtes Nein.

---

## 3c. Ergebnis: der harmonische Weg trägt

TEAM 28 ist vollständig linear, und gesucht ist die Schwebehöhe — also die
zeitgemittelte Kraft im eingeschwungenen Zustand bei ruhender Platte. Der
harmonische Ansatz ist damit **keine Näherung**, sondern derselbe Zustand,
direkt gerechnet.

Die Aufspaltung A(Δt) = K + C/Δt ist über drei Abtastungen exakt möglich
(Vorhersagefehler der Kontrollabtastung 4,63e-18, ‖C‖ außerhalb des FEM-Blocks
exakt 0). Daraus das reelle Blocksystem `[[K, -ωC],[ωC, K]]`.

| | κ₂ | GMRES |
|---|---|---|
| transient (Δt = 1e-4) | 1,07e+06 | 4000, stagniert bei 0,0264 |
| harmonisch | 2,07e+06 | 4000, stagniert |
| harmonisch + C-to-R | — | **38 Iterationen auf 7,5e-09** |

κ **steigt**, und die Iterationszahl fällt um Faktor 105. Nicht die Kondition
entscheidet, sondern die Blockstruktur.

### Der innere Löser (W+T) = K + ωC

C-to-R braucht zwei Lösungen mit (W+T) je äußerer Iteration. Genauigkeitsreihe:

| rel. Fehler in (W+T)⁻¹ | äußere Iterationen | innere Schritte | Arbeit |
|---|---|---|---|
| 1e-1, 1e-2 | trägt nicht | — | — |
| 1e-3 | 81 | 2899 | 4,70e+06 |
| **1e-4** | **41** | **2914** | **2,39e+06** |

Die feinere Stufe ist billiger — eine gröbere Toleranz spart innere Schritte,
kostet aber äußere, und das Produkt entscheidet.

**Gesamtarbeit: 2,39e+06 gegen 4,00e+06 des transienten Laufs, also Faktor
1,67 besser — ohne jeden inneren Vorkonditionierer.** Und der transiente
Vergleichswert ist großzügig gerechnet: seine 4000 Iterationen sind
vorkonditioniert und erreichen die Toleranz trotzdem nicht, während die
harmonische Rechnung unvorkonditioniert konvergiert.

### Blockvorkonditionierung von (W+T) lohnt NICHT

Die Deckenmessung auf (W+T) wiederholt alle Befunde des transienten Systems
(blockdiagonal schlechter als nichts, DtN exakt in einer Iteration), aber die
Genauigkeitsanforderung steigt von 1e-4 auf **1e-6**, und damit die Kosten:

    DtN auf (W+T):   8 x 1205 x 639 = 6,16e+06 Matvecs je Lösung
    schlicht GMRES:                    2914     Matvecs je Lösung

**Unvorkonditioniert ist 2100-mal billiger.** Für den inneren Löser ist die
einfachste Variante die richtige — hier ist keine Blockstruktur zu holen.

### Auf dem Hexaedernetz trägt das innere GMRES nicht mehr

Die Zahlen oben stammen von team28_coarse_12 (4511 Unbekannte). Auf dem
Gmsh-Hexaedernetz `team28_hex.py --scale 1` (4944 Hexaeder, 18 651 FEM- und
14 784 BEM-Freiheitsgrade, nd = 33 435) erreicht das innere GMRES mit ILU die
Toleranz 1e-4 nicht: Residuum 9,7e-3 nach 4000 Schritten, 0,02 nach einer
Stunde (`docs/design/gmsh-phase9-report.md`). Mit so ungenauen inneren
Lösungen trägt C-to-R nach der Genauigkeitsreihe oben nicht.

`--harmonic-inner-direct` umgeht das für kleine Netze: (W+T) wird einmal
spaltenweise abgetastet (`sample_operator_dense`, A e_j im Tilde-Raum
einschließlich des Nullraum-Terms, also regulär) und mit cuSOLVER zerlegt.
Jede innere Lösung ist dann exakt, und für C-to-R gilt wieder der Fall
„Fehler in (W+T)⁻¹ → 0“.

Kosten auf hex_s1: 8,3 GiB für die Matrix; das Abtasten braucht nd
Operatoranwendungen zu je etwa 0,3 s, also rund 3 Stunden. Die Zerlegung
selbst dauert Minuten. Damit ist der Weg auf nd ≲ 35 000 begrenzt (Speicher)
und schon dort durch das Abtasten teuer.

Für feinere Netze ist der nächste Schritt eine Schur-Zerlegung: den dünn
besetzten FEM-Block mit cuDSS zerlegen und nur das BEM-Schur-Komplement
(n_BEM² statt nd², auf hex_s1 1,6 statt 8,3 GiB) dicht aufbauen. Eichung und
Kohomologie-Projektion müssen dabei im FEM-Block mitgeführt werden.

---

## 4. Wie es weitergeht — je nach Ergebnis

### Fall A: Decke < ~1,5

Der Ansatz ist widerlegt. Kein Calderón, keine H-Matrix-Näherungsinverse. Was
dann noch offensteht:

- Variante 4 ansehen. Trägt das exakte Schur-Komplement, liegt der Hebel im
  Term `A21 A11⁻¹ A12`, nicht in `A22`.
- Ansonsten gilt die Empfehlung aus `convergence-findings.tex` unverändert: erst klären,
  **welche Genauigkeit die Kraftwerte tatsächlich brauchen**. Bei `κ₂ ≈ 1e6`
  ist unterhalb von `1e-6` ohnehin die Grenze der doppelten Genauigkeit
  erreicht.

### Fall B: Decke > ~2

Dann lohnt sich Arbeit an `A22`, und zwar in dieser Reihenfolge — von billig
nach teuer, jede Stufe mit eigener Messung:

**Stufe 1 — Nahfeld-Blockinverse (Tage, nicht Wochen).**
Die billigste sinnvolle Form der H-Matrix-Näherungsinverse. `dense_blocks`
enthält bereits die Nahfeldblöcke des Clusterbaums; die Diagonalblöcke `(I,I)`
werden mit dem vorhandenen `bf::DenseLU` faktorisiert und als
Block-Jacobi-Vorkonditionierer auf den K-Block angewandt. Kein neuer
Assemblierungspfad, keine H-Arithmetik.

Achtung bei der Äquilibrierung: `diag(D_K μ₀G D_K) = 1` per Konstruktion,
punktweises Jacobi ist auf dem K-Block also bereits ein No-Op. Die
Blockversion ist es nicht — sie nimmt die Kopplung *innerhalb* eines Clusters
mit, und genau dort sitzt beim Single-Layer-Operator der Hauptanteil.

Erwartung: nimmt einen Teil der Decke, nicht die ganze. Die Messung sagt,
wieviel — und ob Stufe 2 noch nötig ist.

**Stufe 2 — H-LU auf dem Clusterbaum.**
Approximative LU-Zerlegung mit Rangkürzung über die vorhandene Blockstruktur.
Braucht H-Addition und H-Multiplikation. Deutlich stärker als Stufe 1, aber
auch deutlich mehr Code.

**Stufe 3 — Calderón / Operator-Preconditioning.**
`W V = ¼I − K'²`. Erforderlich:

1. barizentrisch verfeinertes Oberflächennetz (`src/MeshRefinement.cu` als
   Ausgangspunkt, aber Oberfläche statt Volumen)
2. Buffa-Christiansen-Basisfunktionen auf dem Dualnetz
3. Assemblierung des hypersingulären Operators `W` — neuer singulärer
   Quadraturpfad; `src/SauterSchwab.cu` deckt den Kern nicht ab
4. Massenmatrix zwischen Primal- und Dualraum, plus deren Inverse
   (gut konditioniert, CG genügt)

Der Gewinn ist netzunabhängige Kondition auf dem K-Block. Das ist genau dann
sein Geld wert, wenn die Netze deutlich feiner werden sollen — bei fester
Netzgröße bringt Stufe 1 oder 2 einen großen Teil davon zu einem Bruchteil des
Aufwands.

---

## 5. Grenzen dieses Stands

- Die Änderungen an `src/fem.cu`, `src/Backend.cu` und
  `include/Backend.cuh` sind **nicht übersetzt** worden — die Arbeit entstand
  ohne CUDA-Toolkit. Geprüft sind Klammerbilanz und Sichtbarkeit der
  verwendeten Bezeichner, nicht mehr. Dieselbe Grenze wie im methodischen
  Nachtrag von `convergence-findings.tex`.
- `src/BlockPreconditionerStudy.cpp` und der zugehörige Test sind übersetzt
  und ausgeführt, alle Prüfungen grün.
- Der Aufwand der Studie ist **kubisch** in `n`. Gedacht für die groben
  Ersatznetze aus `tools/make_coarse_team28.py` (`n = 1791` bis `9139`), nicht
  für das Originalnetz. Für `n = 4511` liegt eine Zerlegung bei rund `3e10`
  Flops, und die Studie braucht mehrere davon plus fünf GMRES-Läufe.
- `κ₂(M⁻¹A)` je Variante wird ab `n > 6000` automatisch abgeschaltet. Die
  Iterationszahlen bleiben.


---

# Anhang: der harmonische Loeser und die offene Kraftfrage

## Was funktioniert

Der harmonische Loeser konvergiert. Auf `team28_coarse_12`:

    19 aeussere Iterationen, rel. Residuum 9,27e-07
    38 innere (W+T)-Loesungen, zusammen 103520 Iterationen

Gegen 4e+06 Iterationen des transienten Laufs, der bei 2,6e-02 stagniert.
Faktor 39 weniger Arbeit, und er liefert eine Loesung.

Geprueft und bestaetigt:

- Arnoldi-Relation `|H Z_k - sum V_i h_ik| / |h|` = 5e-17
- Schaetzung gegen echtes Residuum beim Restart: Abweichung 5e-15
- Linearitaet des Vorkonditionierers: 3,4e-10
- Phasenmittelung: die acht Einzelkraefte sind paarweise identisch, also
  periodisch mit pi statt 2pi -- die doppelte Frequenz, die eine
  quadratische Groesse haben muss
- Amplitudenkorrektur 1/sin(omega*dt): Faktor 1013 exakt wie vorhergesagt

## Was nicht funktioniert, und was davon belegt ist

**Die Kraft stimmt nicht.** Bei 11,3 mm, wo m*g = 1,0497 N stehen muss:

    Maxwell (Oberflaeche) : (0,0120,  0,00286, -0,0702) N
    Lorentz (Volumen)     : (0,00037, 8,92e-05, -5,77e-06) N

**BELEGT: beide Kraftpfade verletzen die Achsensymmetrie.** TEAM 28 ist
achsensymmetrisch, F_x und F_y muessen praktisch null sein. Beim
Volumenintegral ist F_x 64-mal groesser als F_z. Beide Pfade werten
dasselbe Feld aus -- der Fehler liegt also im FELD, nicht in den
Kraftpfaden.

**BELEGT: omega*C koppelt rund 30-mal zu schwach.**

    ||wC_i||/||K_i||, zeilenweise ueber 687 leitende Kanten:
      Mittel 0,0378, min 0,00162, max 0,14
    erwartet aus w*kappa*mu0*h^2 bei h = 8 mm:  ~0,86

Das gemessene Verhaeltnis entspricht einer Elementgroesse von 1,68 mm;
die Plattenelemente sind rund 8 mm. Faktor 4,8 in h, also 23 in h^2.

Das erklaert unmittelbar ||A_i||/||A_r|| = 0,0285 und damit die
verschwindende Lorentzkraft, die LINEAR in A_i ist.

**BELEGT: der Maxwell-Oberflaechenpfad ist hier zusaetzlich untauglich.**
Fuer den Realteil allein liefert er -0,122 N. A_r ist im Wesentlichen das
Spulenfeld; bei mu_r = 1 und stromfreiem Inneren muss das Integral ueber
eine geschlossene Flaeche dafuer exakt null sein. Der Ausloeschungsfehler
auf 200 Flaechen ist groesser als das gesuchte Ergebnis.

## Geprueft und AUSGESCHLOSSEN

- Elementmatrizen: `C_e = Int kappa W.W`, `K_e = Int nu curlW.curlW`
  (fem.cu:177-184). Korrekt.
- mm-nach-m-Skalierung: erfasst beide Knotenarrays, sauber (fem.cu:620-645).
- `compute_maxwell_loads`: nimmt hex_edges/hex_edge_signs, ist also fuer
  Nedelec gebaut; Kantenindizierung mit Vorzeichen korrekt; 4-Punkt-Gauss
  mit Gewicht 1,0 korrekt; Spannungstensor in Standardform.
- Flaechenorientierung: aus `local_face_to_nodes` der eigenen Komponente
  abgeleitet, zeigt nach aussen. Sorgfaeltig geloest.

## Offen

Warum verhaelt sich C, als waeren die Elemente fuenfmal kleiner? Die
Elementmatrix ist korrekt, die Skalierung auch. Der naechste Verdaechtige
ist die Gewinnung von C aus zwei Assemblierungen:

    C = ( A(dt=1/omega) - A(dt=1e30) ) / omega

Zu pruefen waere, ob `A(dt=1e30)` wirklich K ist -- bei dt = 1e30 ist
C/dt = 1e-30*C, was gegenueber K voellig verschwindet, aber die Frage ist,
ob `build_implicit_euler_system` mit einem so grossen dt ueberhaupt das
tut, was man erwartet.

Ein zweiter offener Punkt: `VtkExporter.cu` transformiert B SPALTENweise
und begruendet das mit der Aufbaukonvention von `compute_jacobian`
(J[i][j] = d x_j / d xi_i); der Maxwell-Kernel in `TransientSolver.cu`
transformiert ZEILENweise. Beide koennen nicht recht haben. Das koennte
sowohl die Symmetrieverletzung als auch den Faktor erklaeren und waere
vor allem anderen zu klaeren.

> **Nachtrag (31.10.2026): geklaert, der zweite Punkt ist erledigt.**
> `compute_jacobian` (und ebenso `compute_jacobian_tet`/`_prism`) baut
> J[i][j] = d x_j / d xi_i, also J = J_std^T und invJ = J_std^-T. Richtig
> ist damit W_phys = invJ * W_ref (ZEILENweise ueber invJ) und
> rot W_phys = J^T * rot W_ref / detJ (SPALTENweise ueber J), so wie
> `VtkExporter.cu` es begruendet. Der Maxwell-Kernel in `TransientSolver.cu`
> war tatsaechlich falsch (zeilenweise) und ist laengst auf die spaltenweise
> Form korrigiert; der Kommentar dort beschreibt die Wirkung (F_x = 17 % von
> F_z bei achsensymmetrischer Anordnung). Der obige Absatz beschreibt also
> einen alten Stand. Alle Produktivstellen (Assemblierung und
> `verify_bz_nedelec_body` in `fem.cu`, Maxwell-Kernel, `VtkExporter`,
> `HarmonicLorentz.cuh`, `HarmonicMaxwell.cuh`, `NonlinearSolver.cu`)
> rufen inzwischen `piola_covariant`/`piola_contravariant` aus
> `NedelecDispatch.cuh` auf, statt die Formel abzuschreiben, und
> `tests/test_piola_production.cu` prueft die Kraft- und Ausgabekoerper auf
> gedrehten und gescherten Elementen gegen ein homogenes B, mit einem
> Kontrollfall, der die transponierte Formel verwerfen muss. Als Erklaerung
> fuer den Faktor 5 in C scheidet die Transposition damit aus: sie betraf
> nur den Kraftpfad, nicht die Assemblierung von K und C.


## Nachtrag: Code-Verfolgung der schwachen Kopplung

Vollstaendig im Code verfolgt, ohne Simulationslauf.

### Geprueft und korrekt

- `build_implicit_euler_system`: `A_sys = K + C/dt`, wie erwartet
  (TransientSolver.cu:38).
- Elementmatrizen: `K_e += nu * dot_curl * weight`,
  `C_e += kappa * dot_W * weight` (fem.cu:181-182). `nu` und `kappa`
  kommen direkt aus `mat` (fem.cu:44-45).
- Differenzbildung `C = (A(1/omega) - A(1e30))/omega`: bei K ~ 6,4e+03 und
  wC ~ 5,5e+03 sind beide von derselben Groessenordnung -- keine
  Ausloeschung, relativer Fehler 0.
- Meine Abschaetzung aus den Elementmatrizen ergibt wC/K ~ 0,85, also
  genau den erwarteten Wert. Bis zur CSR-Ebene stimmt alles.

### Gefunden: die Aequilibrierung passt nicht zum Operator

`split.K` und `split.C` stammen aus der dicht abgetasteten Matrix, also
aus dem TILDE-Raum. Dort ist `K~ = D K D` und `wC~ = D (wC) D`. Im
zeilenweisen Verhaeltnis kuerzt sich `D_i` heraus, `D_j` aber nicht --
und weil `C_ij` nur auf den leitenden Kanten von null verschieden ist,
geht dort ein anderes `D_j` ein als bei `K`.

Entscheidend: **D wurde bei dt = 1e-4 bestimmt, der harmonische Operator
benutzt dt = 1/omega = 3,18e-03.**

    C/dt bei dt = 1e-4 :  1,0e+04
    omega*C            :  3,14e+02
    Verhaeltnis        :  31,8

Der Massenterm ist in der Aequilibrierung also 31,8-mal staerker
gewichtet als im Operator, `D` auf den leitenden Kanten entsprechend
sqrt(31,8) = 5,6-mal zu klein. Das gemessene Verhaeltnis erscheint um
denselben Faktor gedrueckt:

    0,0378 * 5,6 = 0,21    gegen erwartete ~0,85

**Faktor 5,6 von 23 sind damit erklaert.**

### Und der Rest: meine Erwartung war falsch

Der verbleibende Faktor 4 steckte nicht im Code, sondern in meiner
Abschaetzung. `w*kappa*mu0*h^2 ~ 0,86` unterstellt `|curlW| ~ |W|/h`. Fuer
Nedelec-Kantenelemente auf einem Quader ist das zu grob: die drei
Richtungen mischen sich, und die Zeilennormen von K und C haengen anders
von den Kantenlaengen ab als h^2.

Nachgerechnet mit DENSELBEN Basisfunktionen und DERSELBEN Quadratur wie
im Code (Basis und curl vorher gegen zentrale Differenzen geprueft,
Abweichung 1,3e-11):

    Quader             ||wC_e_i|| / ||K_e_i||, Mittel ueber 12 Kanten
    8 x 8 x 10 mm      0,125
    8 x 8 x  5 mm      0,075
    15 x 15 x 10 mm    0,279
    4 x 4 x  5 mm      0,031

Meine Handabschaetzung war also um Faktor 7 zu hoch. Damit schliesst sich
die Bilanz:

    gemessen im Tilde-Raum              0,0378
    x 5,6 (Aequilibrierung)             0,213
    exakt erwartet (8x8x10 mm)          0,125
    verbleibend                         Faktor 0,58

Das liegt innerhalb der Unsicherheit ueber die tatsaechlichen
Elementgroessen der Platte.

**BEFUND: die Kopplung ist NICHT zu schwach. Sie hat die Groesse, die
dieses Netz hergibt.** Ein Imaginaerteil von 2,8 Prozent des Realteils ist
bei diesen Elementgroessen zu erwarten, und eine Kraft, die linear in A_i
ist, ist entsprechend klein.

Damit ist die Erklaerung fuer die fehlende Levitationskraft nicht ein
Fehler, sondern die NETZAUFLOESUNG. Bei 348 Hexaedern und einer Skintiefe
von 12,2 mm ist die Platte mit einem bis zwei Elementen ueber die Dicke
diskretisiert -- viel zu grob, um Wirbelstroeme aufzuloesen.

### Was daraus folgt

Die Aequilibrierung wird im harmonischen Pfad bewusst nicht neu
aufgesetzt -- das war noetig, damit die Aufspaltung A(dt) = K + C/dt im
selben Raum gilt (siehe Abschnitt oben). Sie passt damit aber zum
TRANSIENTEN Operator, nicht zum harmonischen.

Ob das nur die MESSUNG verzerrt oder auch den LOESER, ist nicht geklaert.
Fuer den Loeser ist eine unpassende Aequilibrierung kein Fehler, sondern
nur eine schlechtere Vorkonditionierung -- die Loesung selbst bleibt
richtig. Fuer die Kopplungsmessung ist sie ein systematischer Fehler.

Der verbleibende Faktor 4 ist nicht erklaert. Er koennte in meiner groben
Abschaetzung h = 8 mm stecken (h geht quadratisch ein, 8 gegen 4 mm waere
bereits Faktor 4), oder eine weitere Ursache haben.

### Naechster Schritt, falls jemand weitermacht

Die Kopplungsmessung im UNSKALIERTEN Raum wiederholen, also `split.K` und
`split.C` vor der D-Skalierung vergleichen. Dann faellt die
Aequilibrierungsfrage weg und man sieht, ob der Faktor 4 real ist. Dafuer
muesste `--harmonic-study` die Abtastung ohne Aequilibrierung anbieten --
oder D aus dem Operator holen und herausrechnen.
