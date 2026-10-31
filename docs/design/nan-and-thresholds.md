# Vergleiche, die NaN nicht sehen — eine Durchsicht des gesamten Codes

Bei der Suche nach den NaN-Kräften des TEAM-28-Laufs sind nacheinander vier
Fehler aufgetaucht, die alle dieselbe Bauart haben. Das war Anlass, den
gesamten Code gezielt danach abzusuchen. Dieses Dokument hält fest, was
gefunden wurde, was behoben ist und was bewusst stehen bleibt.

## Das Muster

Zwei verwandte Formen:

**Form 1 — der Vergleich sieht NaN nicht.** Jeder Vergleich mit NaN ist
falsch. `a < b`, `a > b`, `a == b` sind für NaN alle unwahr. Damit gilt:

```cpp
if (v > max) max = v;          // NaN wird nie zum Maximum
if (scale < a) scale = a;      // NaN setzt scale nicht
if (d != 0.0) inv = 1.0/d;     // NaN ist "ungleich null" -> inv = NaN
if (res <= tol) return;        // NaN gilt nie als konvergiert
```

Eine min/max-Statistik kann NaN also grundsätzlich nicht anzeigen. Wer sie zur
Absicherung benutzt, sichert nichts ab.

**Form 2 — die Prüfung auf exakte Gleichheit greift zu kurz.** `d != 0.0`
schützt vor der Division durch die exakte Null, nicht aber vor der Division
durch `1e-300`. Bei physikalischen Größen ist die exakte Null der seltene
Fall; die rundungskleine Größe ist der häufige.

In beiden Formen gilt: die richtige Prüfung ist **relativ zu einem Maßstab**
und schließt Endlichkeit ausdrücklich ein.

## Befunde und Stand

### Behoben

| Ort | Mechanismus | Wirkung |
|---|---|---|
| `bf::dnrm2` (`HostLinalg.cu`) | `scale < a` ist für NaN falsch, `scale` bleibt 0, Rückgabe `(scale==0) ? 0.0 : ...` | Ein Vektor aus **lauter** NaN lieferte die Norm **0.0**. GMRES meldete „Converged, Rel. Res: 0" und gab NaN-Kräfte aus. Betraf jedes Abbruchkriterium und jede Norm-Diagnose im CPU-Pfad. |
| `bf::dgemv` (`HostLinalg.cu`) | `y[i] = alpha*acc + beta*y[i]` las `y` auch bei `beta == 0` | BLAS verlangt ausdrücklich, dass `y` bei `beta == 0` nicht gelesen wird. `d_temp_z` war uninitialisiert, `0 * NaN = NaN` machte **16544 von 16544** BEM-Einträgen zu NaN. **Das war die Ursache der NaN-Kräfte.** |
| `AcaPlus::compress` | `std::abs(val) > max_val_v` wählt ein NaN nie als Pivot — speichert es aber trotzdem in `U`/`V` | Ein NaN aus dem BEM-Kern wäre unbemerkt in die komprimierten Blöcke gewandert. Bricht jetzt mit Angabe der Matrixposition `(i, j)` ab. |
| `AlgebraicMultigrid` (3 Stellen) | `(d != 0.0) ? 1.0/d : 0.0` | Rundungskleine Diagonalen ergaben `inv_diag = 1e+20`. Ersetzt durch `fill_inv_diag()` mit relativer Schranke. |
| `AmsPreconditioner` | dieselbe Prüfung im Kanten-Jacobi; `A_G = G^T A G` außerhalb der Leiter komplett rundungsklein | `||z_A|| = 8.6e+14` bei `||r_A|| = 1`. Die betroffenen Zeilen werden jetzt ausgeblendet. |
| `DeviceArray::allocate` | lieferte uninitialisierten Speicher | 62 Aufrufstellen, keine einzige mit folgendem `fill_zero()`. Nullt jetzt bei jeder Neuallokation. |
| GMRES-Rückwärtssubstitution (`CoupledSolver.cu`) | `y[i] /= H[i][i]` ohne jede Prüfung | Bei Stagnation oder „happy breakdown" wird `H[i][i]` winzig, `y[i]` unendlich und `x += Z*y` komplett NaN. Jetzt relative Pivotschranke, unbestimmbare Anteile werden weggelassen und gemeldet. |
| `DenseLU::factorize`, AMG-Grobgitter-LU | `if (best == 0.0)` als Singularitätsprüfung | Ein Pivot von `1e-300` kam durch. Ein NaN ebenfalls: `v > best` ist für NaN falsch, das NaN wird nie als *besseres* Pivot gewählt, steht aber als Startwert `best = |A(k,k)|` schon da und ist „ungleich null". Jetzt `isfinite` plus relative Schranke. |
| `evaluate_quad_normal` (`BemIntegration.cuh`) | `scale3(1.0/len, n)` ohne Prüfung | Ein entartetes Quad lieferte eine Normale aus Inf/NaN, die über die ACA-Blöcke in den ganzen Operator wandert. Liefert jetzt die Nullnormale — für ein flächenloses Element ist `r·n' = 0` richtig. |
| `D_scale`-Diagnose (`CoupledSolver.cu`) | `std::min`/`std::max` über Gleitkommadaten | Die Ausgabe „min = 1.14e-05, max = 0.000131" schloss NaN **nicht** aus. Ergänzt um `audit_finite()`, das zählt statt vergleicht. |
| `build_auxiliary_nodal_matrix` | Parameter `out_num_tiny_diagonal` existierte, wurde mit `nullptr` aufgerufen | Die aussagekräftige Kennzahl wurde berechnet und weggeworfen. Jetzt angeschlossen. |

### Geprüft und in Ordnung

- `NedelecHexahedron.cuh`, `NodalHexahedron.cuh`: `1.0/detJ` ist mit
  `detJ > 1e-14 || detJ < -1e-14` bzw. `detJ != 0.0` abgesichert. Die erste
  Form ist richtig, die zweite grenzwertig, aber `detJ` ist dort das Ergebnis
  einer Geometrieauswertung mit bekannter Größenordnung.
- `GMRESSolver`: `if (h_next > 1e-14)` vor `1.0/h_next` — korrekt und relativ
  gedacht.
- `CohomologyGauge.cpp:70`: `if (b_norm < 1e-300) return;` — für NaN ist das
  falsch, der Löser läuft dann durch. Das ist die **sichere** Richtung
  (keine falsche Konvergenzmeldung).

### Bewusst stehen gelassen

- **Abbruchkriterien der Form `if (res/b_norm <= tol) return;`** (GMRES,
  BiCGStab, CohomologyGauge). Für NaN sind sie falsch, der Löser meldet also
  **keine** Konvergenz und läuft in die Iterationsgrenze. Das ist die sichere
  Richtung. Eine ausdrückliche Endlichkeitsprüfung wäre trotzdem besser — sie
  würde sofort abbrechen statt 1000 Iterationen zu verbrennen.
  **Erledigt (31.10.2026)**, siehe den Nachtrag unten.
- **Bounding-Box-Berechnung** (`fem.cu:630`): `fmin`/`fmax` über
  Knotenkoordinaten. NaN-Koordinaten fielen durch. Risiko gering, weil der
  VTK-Leser die Koordinaten liefert; ein Audit direkt nach dem Einlesen wäre
  der bessere Ort.
- **Die relative Schranke ist selbst eine Annahme.** `1e-10` trennt beim
  vorliegenden Problem sauber (das Rauschen liegt bei `~1e-16`). Zeilen, die
  nur um `1e-6` kleiner sind als der Rest, werden **nicht** erfasst und
  verstärken weiterhin — `tests/test_ams_scalar_masking.cu` prüft und
  dokumentiert genau diesen Fall. Verschöben sich die Materialskalen, müsste
  die Schranke mitgeführt werden.

## Faustregeln fürs Weiterarbeiten

1. **Zählen statt vergleichen.** Wer wissen will, ob Daten in Ordnung sind,
   zählt nicht endliche Einträge. min/max beantwortet die Frage nicht.
2. **Keine Prüfung auf exakte Null vor einer Division.** Immer eine Schranke
   relativ zu einem Maßstab, und `isfinite` dazu.
3. **Kein uninitialisierter Speicher.** Seit `DeviceArray::allocate` nullt, ist
   die Klasse hier erledigt; bei rohem `malloc`/`new` gilt es weiter.
4. **Diagnosen müssen ausschlagen können.** Eine Kennzahl, die im Fehlerfall
   harmlos aussieht, ist schlimmer als keine — sie schafft falsches Vertrauen.
   „888 mit leerer Diagonale" hat die eigentliche Zahl (mehrere tausend
   rundungskleine) jahrelang verdeckt.


## Nachtrag: Grenzen der Prüfumgebung ohne CUDA-Toolkit

Ein Teil dieser Arbeit ist auf einer Maschine ohne `nvcc` entstanden. Geprüft
wurde mit `g++` gegen Stub-Header für `cuda_runtime`, `cublas`, `cusparse` und
`cusolver`, wobei die `<<<...>>>`-Launchsyntax vorher aus dem
vorverarbeiteten Code entfernt wird. Das fängt sehr viel — Typfehler,
Namensauflösung, falsche Signaturen — aber nicht alles:

- **`nvcc` ist strenger bei der Typherleitung.** Eine Initialisierungsliste mit
  gemischt getippten Elementen (`{std::pair<bool,double>{a,b}, {c,d}}`) übersetzt
  `g++` anstandslos, `nvcc` meldet *cannot deduce type of range in range-based
  for loop*. Ausgeschriebene Felder statt solcher Listen vermeiden das.
- **Gerätecode wird gar nicht geprüft.** Alles innerhalb von `__global__`
  Kerneln läuft im Stub-Aufbau als Host-Code; Beschränkungen des Gerätecodes
  fallen erst beim echten Build auf.
- **Veraltungswarnungen der CUDA-Bibliotheken** erscheinen nicht.

Wer ohne Toolkit arbeitet, sollte deshalb `-pedantic -Wall` mitgeben und damit
rechnen, dass ein Build auf der Zielmaschine noch Fehler findet.

## Nachtrag 31.10.2026: Endlichkeitsprüfung in den Krylov-Lösern

Die Abbruchkriterien sind geblieben, dazu kommt eine ausdrückliche Prüfung:

- `GMRESSolver` (`src/CoupledSolver.cu`) prüft `||b||`, das Residuum am
  Zyklusbeginn und das über die Givens-Drehungen mitgeführte Residuum. Ist
  eines nicht endlich, bricht er sofort ab, **bevor x angefasst wird**, und
  `last_relative_residual()` ist NaN. Nebenbei setzt er den Wert jetzt auch,
  wenn er schon am Zyklusbeginn konvergiert; vorher blieb dort der Wert eines
  früheren Aufrufs stehen.
- `BiCGStabSolver` prüft `||b||`, das Startresiduum, `||s||` und das
  Residuum; beim Abbruch gibt er das beste bisher gefundene x zurück.
- Der harmonische GMRES (`src/HarmonicSolver.cu`) prüft den neuen
  Arnoldi-Vektor. Vorher fiel ein NaN dort erst in `!(den > 0)` auf, und das
  führte in die Aktualisierung von x — x wurde NaN.
- `CohomologyGauge.cpp` (`solve_cg`) bricht bei nicht endlichem Residuum ab.

Bei der Durchsicht fanden sich außerdem **drei Aufrufer mit Form 1 in der
gefährlichen Richtung**: `fem.cu` prüfte zweimal `last_relative_residual() >
tol` als „nicht konvergiert“, und `warn_if_inner_failed` in
`HarmonicSolver.cu` kehrte bei `!(rel > …)` früh zurück. Ein NaN-Residuum
galt damit als konvergiert bzw. löste keine Warnung aus. Alle drei lauten jetzt
`!(res <= tol)` bzw. `rel <= …`.

Test: `tests/test_solver_nonfinite_abort.cu` (NaN in b, NaN in der Matrix,
Überlauf erst im Arnoldi-Schritt; GMRES und BiCGStab). Gegen den alten Löser
fallen 8 von 17 Prüfungen durch: GMRES läuft bis `max_iter` (200) und gibt ein
NaN-x zurück, BiCGStab ebenso.
