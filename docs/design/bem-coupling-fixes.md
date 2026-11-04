# Umsetzungsfehler der BEM-Kopplung: vier Korrekturen

Stand: 04.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), Linux (nicht RTX 3090
/ Windows). Basis: `main` b4813c8, Zweig `fix-bem-coupling`, ein Commit je Schritt. Netze aus
`tools/gmsh/team28_hex.py` (`--scale 1.0` und `--scale 0.5`), 2 Hz für die Zeilenmessung
(`--bem-row-check`), 50 Hz für die Abschlussmessung, Platte bei 11,3 mm (`--plate-z=11.3`).
Logs, Netze, Dumps in `build/fix/` und `build/rowcheck/` (nicht im Repo).

## Auftrag und Ergebnis in Kürze

`bem-row-check.md` hat vier Umsetzungsbefunde gemessen. Dieser Zweig behebt **nur diese vier**; die
Formulierungsfrage K1/K2 bleibt offen. Jede Aussage des Auftrags wurde am Code geprüft; alle vier
treffen zu (Abschnitt „Prüfung der Befunde“).

**Gemessen** (Skalierung 1,0; Tabellen unten)

1. **Schritt 1** (T-Geometrie in BEM-Reihenfolge): ||T k − g||/||g|| 104,8 % → 101,3 %. Die Korrektur ist
   richtig (Test gegen dichte Host-Referenz, Gegenprobe), senkt den Messwert aber nur wenig, weil
   Vorzeichen und Basis noch falsch sind.
2. **Schritt 2** (Vorzeichen von T): der Messwert **steigt** auf 183,9 % (K1) / 164,8 % (K2). Das war
   zu erwarten: richtige Vorzeichen auf einer falschen Basis. Ohne Schritt 3 ist Schritt 2 allein keine
   Verbesserung.
3. **Schritt 3** (kovariante Spurfunktionen): **0,00 %** (unter 0,005 %). T k gibt das exakte
   Randintegral wieder (Galerkin-Eigenschaft). Zeile 1 verhält sich damit wie das Referenzintegral:
   mit +K2 7,4 % (Skalierung 0,5: 16,2 %), mit +K1 29,4 %. Die Randspur-Rekonstruktion aus den
   Kantenwerten trifft A_tan auf 0,8 % (Projektion 0,995, vorher 3,57).
4. **Schritt 4** (K-Zeile als Außenraumform): Zeile 2 mit +K1 **4,9 %** (Skalierung 0,5: 7,7 %),
   vorher 74,2 % (Skalierung 0,5: 74,4 %). Hypothese vorab geprüft (Würfeltest ohne FEM, Außenform 9,6 %, Innenform
   371 % / 181 %).
5. **Zielwerte** des Auftrags (Zeile 1 mit K2 etwa 7 %, Zeile 2 etwa 3 %): Zeile 1 **erreicht** (7,4 %).
   Zeile 2 **nicht erreicht** (4,9 % statt etwa 3 %). Der Wert 2,65 % aus dem Bericht stammt aus der
   Messung mit der alten Basis; mit der kovarianten Basis lag er in derselben Messung bei 4,7 %
   (Schritt 3, konsistente Darstellung). Das ist ein Befund, ich habe nicht weitergedreht.
6. **Abschluss, 50 Hz, 11,3 mm, Skalierung 1,0** (BEM-FEM, `--harmonic-inner-direct`):

| | vorher (`main`) | nachher | Referenzpfad |
|---|---|---|---|
| F_z (Zeitmittel) | −25,2 N (F_z/(m g) = −24) | **+0,545 N** (F_z/(m g) = 0,519) | 0,9665 N |
| äußere / innere Iterationen | 24 / 48 (7,71e-07) | 19 / 38 (8,49e-07) | – |
| Projektion auf Biot-Savart, Re, Platte | 0,019 | **1,083** | – |
| Projektion auf Biot-Savart, Re, Innenspule | 0,893 | 0,871 | – |
| Projektion auf Biot-Savart, Re, Außenspule | 0,521 | 0,788 | – |

**Frage: erklären die vier Fixes das falsche Feld?** *Teilweise, nicht ganz.* Sie erklären den größten
Teil des Fehlers: F_z geht von −25,2 N auf +0,545 N, die Projektion der Plattenfelder auf Biot-Savart
von 0,02 auf 1,08, der L2-Fehler gegen den Referenzpfad in der Platte von 2,36 auf 0,34. Das Feld ist
aber noch nicht richtig: F_z liegt 44 % unter dem Referenzpfad, der Imaginärteil in den Spulen ist
im Betrag 1,5- (Innenspule) bis 5,0-fach (Außenspule) zu groß (Referenz: klein), B_φ/B liegt bei 0,06 bis 0,10 (Referenz 4e-4)
und Zeile 2 steht bei 4,9 %. *Interpretation (nicht geprüft):* Der Rest passt zur offenen K1/K2-Frage,
denn Zeile 1 verlangt K2, Zeile 2 K1, und beide unterscheiden sich auf den Spulenseiten um 27 bis 42 %.

## Prüfung der Befunde im Code (alle vier treffen zu)

- **b)** `apply_T_matrix_kernel_body` (`CouplingKernels.cuh`) liest `faces[face_idx]`;
  `bem_to_fem[pos*4+e]` gehört zu Quad `perm[pos]` (`BemCoupling.cpp`); `setup_bem_mesh` bekam
  `mesh.boundary_faces` unpermutiert (`fem.cu`); G und D benutzen `perm[...]`. Gemessen:
  `perm[pos] != pos` für 4136 von 4136 Randquads auf team28.vtk, 3696 von 3696 auf hex_s1.
- **c)** Der Kernel trug `sgn[r]*sgn[c]` in beiden Zeilen; `x_K` ist roh (G wirkt ohne Vorzeichen),
  Gather `a_bem = sign*a`. Abgeleitet: Zeile 1 nur `sign[r]`, Zeile 2 nur `sign[c]` (Herleitung im
  Kommentar des Kernels).
- **a)** `evaluate_rooftop_basis`: W = c(u,v)·∂r/∂u dividiert durch detJ = kontravariante Abbildung,
  Kantenintegral 2a/b. Im Host-Test: 2,0 / 4,0 / 1,0 für a/b = 1 / 2 / 0,5; die kovariante Basis 1,0.
  **Es gibt keine Duplikate:** `SauterSchwab.cu`, `SingularQuadPair.cuh`, `AdjacentIntegration.cu`,
  `fem.cu` (G, D) und `integrate_boundary_mass_matrix` rufen alle `evaluate_rooftop_basis`; eine
  Umstellung an dieser einen Stelle gilt für alle. (`MaxwellLoads.cuh` hat ein eigenes `dr_du`, aber
  für die Kraftintegration, nicht für die Basis.)
- **d)** Hypothese vorab geprüft: Würfelnetz (6 × 3 × 3 Quads) mit Punktquelle im Inneren,
  u = e/|x − x0|, dichte G, D, T aus denselben Integratoren wie der Produktionspfad
  (`tests/test_bem_calderon.cu`): Außenform (+½T − D)a + G k mit k = +∂u/∂n **9,6 %**, mit k = −∂u/∂n
  190,6 %; Innenform (−½T − D)a + G k 371,0 % (k = +) bzw. 181,4 % (k = −). Die Hypothese des
  Auftrags trifft zu: nur das Vorzeichen des ½T-Terms ändert sich. Die Handrechnung an der
  Einheitskugel (D·1 = −½, G ∂u/∂n = −1) stimmt mit dem Kommentar im Kernel überein.
  *Grenze:* das Restresiduum 9,6 % kommt von der Darstellung (D wirkt nur auf die Tangentialspur; der
  Normalanteil der Quelle fehlt an den Würfelkanten), nicht von den Vorzeichen.

## Die Schritte

Jeder Schritt: Änderung, Tests, Messung (`--bem-row-check`, 2 Hz, 11,3 mm; Skalierung 1,0 und 0,5).

1. `fix-bem-coupling 1`: `bem_ordered_boundary_faces` (neu, `BemCoupling.h/.cpp`) liefert die Quads
   in BEM-Reihenfolge; `fem.cu` übergibt sie an `setup_bem_mesh` (Kopie `d_bem_faces`). Alle Verwender
   von T teilen die Geometrie: `apply_physical`, `apply_coupling_T_tilde` und der
   Vorkonditionierer-Anteil (`apply_T_to_A_kernel`, gleicher Zeiger), `row_check_parts`. Test
   `test_bem_coupling_geometry` auf team28.vtk (perm != id für alle Quads) gegen eine unabhängig
   aufgebaute Host-Referenz: 1e-16 (Zeile 1) bzw. 0 (Zeile 2); Gegenprobe alte Indexierung: 0,71 bzw.
   0,71.
2. `fix-bem-coupling 2`: Vorzeichen (Zeile 1 nur `sign[r]`, Zeile 2 nur `sign[c]`), beide Kernel
   (`apply_T_matrix_kernel_body`, `apply_T_to_A_kernel_body`). `test_coupling_sign_convention`
   **musste angepasst werden**: die Prüfung „globale Umkehr aller sign[] ist wirkungslos“ galt für
   `sign[r]*sign[c]` und gilt jetzt nicht mehr (EIN Vorzeichen: die Umkehr kehrt y_A und y_K um); neu
   mit Gegenprobe. Referenztest: Gegenprobe altes Muster 1,54 bzw. 1,51.
3. `fix-bem-coupling 3`: `evaluate_rooftop_basis` liefert W_0 = ¼(1−v) g^u, W_1 = ¼(1+u) g^v,
   W_2 = −¼(1+v) g^u, W_3 = −¼(1−u) g^v mit der reziproken Basis (g^u = (∂r/∂v × N)/|N|²,
   g^v = (N × ∂r/∂u)/|N|², N = ∂r/∂u × ∂r/∂v). Tests: Rekonstruktion eines konstanten Felds aus den
   Kantenintegralen = 1,000000 für a/b = 1, 2, 0,5 (Gegenprobe alte Funktion 2,0 / 4,0 / 1,0 = 2a/b);
   Kantenintegral der Basisfunktion i auf Kante j = Kronecker-Delta auf einem verzerrten Quad (2e-10).
   **`test_singular_quadrature` musste angepasst werden:** seine festen Referenzblöcke (gerechnet mit
   der alten Basis) waren damit ungültig; die Referenz (N = 16, lokale Kopie der Sauter-Schwab-
   Zerlegung mit freier Ordnung) wird jetzt zur Laufzeit gerechnet. Ergebnis 3,8e-4 / 5,0e-5 / 1,0e-3 /
   6,2e-5, wie im Kopf der Datei dokumentiert.
4. `fix-bem-coupling 4`: Vorzeichen des ½T-Terms in Zeile 2 (`val_K += 0.5 * ...`); D, G und
   Vorkonditionierer unverändert, Kommentare der Blockstruktur nachgezogen (S~ = µ0 G, blocktriangulär
   mit −T, keine Vorzeichenfolge darin). Tests: Calderón-Außenraumtest mit Gegenproben (Innenform,
   falsches Vorzeichen von k), Vorzeichen des ½T-Terms im Kernel- und Referenztest.

### Messwerte je Schritt (Zeile „alle“ der jeweiligen Tabellen)

| Schritt | Skalierung | T-Test K1 / K2 | Zeile 1 (+K1 / −K1 / +K2 / −K2) | Zeile 2 (+K1 / −K1 / +K2 / −K2) | Randspur-Projektion |
|---|---|---|---|---|---|
| vor (`main`) | 1,0 | 104.76 % / 104.53 % | 103.88 % / 103.26 % / 104.65 % / 104.08 % | 74.16 % / 132.37 % / 74.57 % / 131.89 % | 3.5732 |
| vor (`main`) | 0,5 | 116.15 % / 114.81 % | 113.30 % / 111.88 % / 115.31 % / 114.04 % | 74.44 % / 132.03 % / 74.86 % / 131.51 % | 3.5222 |
| 1 T-Geometrie | 1,0 | 101.25 % / 101.22 % | 101.00 % / 101.19 % / 101.25 % / 101.47 % | 82.89 % / 123.44 % / 83.15 % / 123.08 % | 3.5732 |
| 1 T-Geometrie | 0,5 | 102.70 % / 102.68 % | 102.19 % / 102.44 % / 102.77 % / 103.06 % | 83.51 % / 122.74 % / 83.76 % / 122.36 % | 3.5222 |
| 2 T-Vorzeichen | 1,0 | 183.88 % / 164.80 % | 169.38 % / 330.37 % / 167.52 % / 340.80 % | 173.01 % / 136.78 % / 170.47 % / 137.73 % | 3.5732 |
| 2 T-Vorzeichen | 0,5 | 184.32 % / 164.61 % | 171.38 % / 330.85 % / 169.53 % / 341.48 % | 158.59 % / 127.10 % / 156.23 % / 127.90 % | 3.5222 |
| 3 Basis | 1,0 | 0.00 % / 0.00 % | 29.40 % / 187.85 % / 7.44 % / 200.94 % | 574.87 % / 459.21 % / 617.26 % / 483.19 % | 0.9949 |
| 3 Basis | 0,5 | 0.00 % / 0.00 % | 31.58 % / 187.74 % / 16.19 % / 201.04 % | 575.37 % / 464.39 % / 613.35 % / 485.15 % | 0.9795 |
| 4 K-Zeile | 1,0 | 0.00 % / 0.00 % | 29.40 % / 187.85 % / 7.44 % / 200.94 % | 4.89 % / 200.82 % / 20.18 % / 206.88 % | 0.9949 |
| 4 K-Zeile | 0,5 | 0.00 % / 0.00 % | 31.58 % / 187.74 % / 16.19 % / 201.04 % | 7.66 % / 200.40 % / 18.93 % / 205.79 % | 0.9795 |

Spalten: T-Test = ||T k − g||/||g||, g = exaktes Randintegral mit der Nédélec-Spur (nur Zeile „alle“,
V0 = Operator). Zeile 1 / Zeile 2 mit den Operatoren **wie sie im Code stehen**, vier Fälle ±K1, ±K2
(„+K1 | −K1 | +K2 | −K2“). Randspur: Projektion der Rooftop-Rekonstruktion aus a_ex auf A_tan, Soll 1.
„vor“ ist der Stand von `main` (Messung des Berichts `bem-row-check.md`).

## Endzustand: Tabellen (Skalierung 1,0, 2 Hz, 11,3 mm)

Zeile 1: ||K a − T k − b|| / ||K a − b||

| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | eben | 2448 | 0.60 % | 200.47 % | 0.60 % | 200.47 % |
| Platte | Seite | 48 | 100.00 % | 100.00 % | 100.00 % | 100.00 % |
| Platte | Kante | 288 | 13.13 % | 186.94 % | 0.36 % | 199.93 % |
| Platte | alle | 2784 | 1.58 % | 200.31 % | 0.60 % | 200.47 % |
| Innenspule | eben | 672 | 9.83 % | 201.78 % | 9.83 % | 201.78 % |
| Innenspule | Seite | 1056 | 35.62 % | 181.70 % | 6.50 % | 200.94 % |
| Innenspule | Kante | 576 | 30.36 % | 185.56 % | 8.92 % | 201.58 % |
| Innenspule | alle | 2304 | 31.53 % | 186.00 % | 7.77 % | 201.24 % |
| Aussenspule | eben | 672 | 11.76 % | 200.12 % | 11.76 % | 200.12 % |
| Aussenspule | Seite | 1056 | 26.33 % | 189.47 % | 5.53 % | 200.01 % |
| Aussenspule | Kante | 576 | 25.72 % | 191.72 % | 7.96 % | 201.12 % |
| Aussenspule | alle | 2304 | 25.28 % | 190.97 % | 6.94 % | 200.31 % |
| alle | alle | 7392 | 29.40 % | 187.85 % | 7.44 % | 200.94 % |

Zeile 2: ||(+0,5 T − D) a + µ0 G k|| / ||(+0,5 T − D) a||

| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | eben | 4992 | 19.62 % | 200.85 % | 30.70 % | 221.35 % |
| Platte | Seite | 192 | 18.49 % | 185.40 % | 20.34 % | 182.96 % |
| Platte | Kante | 384 | 14.43 % | 194.62 % | 15.17 % | 192.02 % |
| Platte | alle | 5568 | 19.59 % | 200.74 % | 30.58 % | 221.01 % |
| Innenspule | eben | 1536 | 8.95 % | 202.33 % | 19.79 % | 215.70 % |
| Innenspule | Seite | 2304 | 6.75 % | 204.35 % | 44.38 % | 233.89 % |
| Innenspule | Kante | 768 | 9.56 % | 204.35 % | 37.52 % | 230.73 % |
| Innenspule | alle | 4608 | 7.92 % | 203.68 % | 37.07 % | 227.60 % |
| Aussenspule | eben | 1536 | 5.87 % | 199.84 % | 6.71 % | 198.33 % |
| Aussenspule | Seite | 2304 | 1.10 % | 199.74 % | 7.26 % | 198.70 % |
| Aussenspule | Kante | 768 | 1.01 % | 199.86 % | 5.87 % | 198.48 % |
| Aussenspule | alle | 4608 | 2.26 % | 199.77 % | 6.97 % | 198.62 % |
| alle | alle | 14784 | 4.89 % | 200.82 % | 20.18 % | 206.88 % |

T gegen das exakte Randintegral:

| Variante | Koerper | Klasse | n | K1 | K2 |
|---|---|---|---:|---:|---:|
| V0 | Platte | eben | 2448 | 0.00 % | 0.00 % |
| V0 | Platte | Seite | 48 | 0.00 % | 0.00 % |
| V0 | Platte | Kante | 288 | 0.00 % | 0.00 % |
| V0 | Platte | alle | 2784 | 0.00 % | 0.00 % |
| V0 | Innenspule | eben | 672 | 0.00 % | 0.00 % |
| V0 | Innenspule | Seite | 1056 | 0.00 % | 0.00 % |
| V0 | Innenspule | Kante | 576 | 0.00 % | 0.00 % |
| V0 | Innenspule | alle | 2304 | 0.00 % | 0.00 % |
| V0 | Aussenspule | eben | 672 | 0.00 % | 0.00 % |
| V0 | Aussenspule | Seite | 1056 | 0.00 % | 0.00 % |
| V0 | Aussenspule | Kante | 576 | 0.00 % | 0.00 % |
| V0 | Aussenspule | alle | 2304 | 0.00 % | 0.00 % |
| V0 | alle | alle | 7392 | 0.00 % | 0.00 % |

Randspur (Rooftop-Rekonstruktion aus a_ex gegen A_tan):

| Koerper | Klasse | Quads | relativer L2-Fehler | Projektion <rec, A_t>/<A_t, A_t> | Betragsverhaeltnis |
|---|---|---:|---:|---:|---:|
| Platte | eben | 1248 | 0.0171 | 0.9941 | 0.9942 |
| Platte | Seite | 48 | 0.0037 | 0.9999 | 1.0000 |
| Platte | Kante | 96 | 0.0037 | 0.9999 | 1.0000 |
| Innenspule | eben | 384 | 0.0250 | 0.9792 | 0.9793 |
| Innenspule | Seite | 576 | 0.0068 | 0.9949 | 0.9949 |
| Innenspule | Kante | 192 | 0.0129 | 0.9894 | 0.9895 |
| Aussenspule | eben | 384 | 0.0049 | 0.9956 | 0.9956 |
| Aussenspule | Seite | 576 | 0.0044 | 0.9967 | 0.9967 |
| Aussenspule | Kante | 192 | 0.0076 | 0.9934 | 0.9934 |
| alle | alle | 3696 | 0.0079 | 0.9949 | 0.9949 |

Skalierung 0,5 gegen 1,0, Zeilen „alle“ je Körper (zuerst 1,0, dann 0,5):

Zeile 1:

| Koerper | Skalierung | n | +K1 | −K1 | +K2 | −K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | 1,0 | 2784 | 1.58 % | 200.31 % | 0.60 % | 200.47 % |
| Platte | 0,5 | 720 | 2.63 % | 201.74 % | 2.31 % | 201.85 % |
| Innenspule | 1,0 | 2304 | 31.53 % | 186.00 % | 7.77 % | 201.24 % |
| Innenspule | 0,5 | 576 | 32.54 % | 186.26 % | 16.89 % | 201.76 % |
| Aussenspule | 1,0 | 2304 | 25.28 % | 190.97 % | 6.94 % | 200.31 % |
| Aussenspule | 0,5 | 576 | 30.51 % | 189.94 % | 15.15 % | 199.45 % |
| alle | 1,0 | 7392 | 29.40 % | 187.85 % | 7.44 % | 200.94 % |
| alle | 0,5 | 1872 | 31.58 % | 187.74 % | 16.19 % | 201.04 % |

Zeile 2:

| Koerper | Skalierung | n | +K1 | −K1 | +K2 | −K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | 1,0 | 5568 | 19.59 % | 200.74 % | 30.58 % | 221.01 % |
| Platte | 0,5 | 1440 | 45.41 % | 198.08 % | 44.91 % | 215.52 % |
| Innenspule | 1,0 | 4608 | 7.92 % | 203.68 % | 37.07 % | 227.60 % |
| Innenspule | 0,5 | 1152 | 9.32 % | 202.83 % | 33.68 % | 225.51 % |
| Aussenspule | 1,0 | 4608 | 2.26 % | 199.77 % | 6.97 % | 198.62 % |
| Aussenspule | 0,5 | 1152 | 4.42 % | 199.58 % | 8.06 % | 198.35 % |
| alle | 1,0 | 14784 | 4.89 % | 200.82 % | 20.18 % | 206.88 % |
| alle | 0,5 | 3744 | 7.66 % | 200.40 % | 18.93 % | 205.79 % |

**Messung:** Zeile 1 (die Operatoren stimmen jetzt mit dem exakten Randintegral überein) wird mit +K2 erfüllt,
nicht mit K1: bei Skalierung 1,0 Platte 0,6 %, Innenspule 7,8 %, Außenspule 6,9 % (gesamt 7,4 %); mit +K1
1,6 % / 31,5 % / 25,3 % (gesamt 29,4 %). Bei Skalierung 0,5 sind es mit +K2 2,3 % / 16,9 % / 15,2 %
(gesamt 16,2 %). Zeile 2 wird mit +K1 erfüllt: Außenspule 2,3 %, Innenspule 7,9 %, Platte 19,6 %
(gesamt 4,9 %); mit +K2 20,2 % gesamt (Innenspule 37,1 %). Die Platte ist in Zeile 2 mit 19,6 % (0,5:
45,4 %) der Ausreißer; ein Zusammenhang mit den Panelmaßen wurde nicht untersucht. Zeile 1 will also K2,
Zeile 2 K1: der K1/K2-Widerspruch, der offen bleibt.

## Abschlussmessung (50 Hz, 11,3 mm, Skalierung 1,0)

Gleiche Befehle vor und nach den Fixes (`fembem_before` aus `main`, `fembem_after` aus diesem
Zweig; Referenzpfad `--reference --harmonic=50` auf demselben Netz). Projektion auf Biot-Savart
(+ Ringmodell-Wirbelströme), Zeile „alle“, `tools/compare_fields.py`:

| Koerper | Phase | proj vorher | proj nachher | L2 vorher | L2 nachher | B/Bref vorher | B/Bref nachher | B_phi/B vorher | B_phi/B nachher |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Platte | Re | 0.0194 | 1.0833 | 2.3509 | 0.3363 | 2.1367 | 1.1313 | 0.1742 | 0.0960 |
| Platte | Im | 0.0229 | 1.2892 | 6.8481 | 0.4930 | 6.7781 | 1.3496 | 0.1872 | 0.1683 |
| Innenspule | Re | 0.8927 | 0.8706 | 1.1994 | 0.4072 | 1.4913 | 0.9524 | 0.3582 | 0.0563 |
| Innenspule | Im | 0.0567 | 1.4810 | 119.9220 | 0.6446 | 119.9183 | 1.5420 | 0.5959 | 0.1128 |
| Aussenspule | Re | 0.5213 | 0.7879 | 2.5552 | 0.4862 | 2.5635 | 0.9013 | 0.3186 | 0.0990 |
| Aussenspule | Im | 0.4557 | 3.7302 | 838.7985 | 3.9176 | 838.7985 | 4.6699 | 0.8429 | 0.3986 |

Vergleich BEM-FEM gegen Referenzpfad (`compare_fields.py TEST REF`), Zeile „alle“:

| Koerper | Phase | proj vorher | proj nachher | L2 vorher | L2 nachher | B/Bref vorher | B/Bref nachher | B_phi/B vorher | B_phi/B nachher |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Platte | Re | 0.0195 | 1.0891 | 2.3620 | 0.3410 | 2.1490 | 1.1377 | 0.1742 | 0.0960 |
| Platte | Im | 0.0250 | 1.3072 | 6.9565 | 0.5161 | 6.8878 | 1.3714 | 0.1872 | 0.1683 |
| Innenspule | Re | 0.8669 | 0.8482 | 1.1653 | 0.3958 | 1.4463 | 0.9236 | 0.3582 | 0.0563 |
| Innenspule | Im | 0.0500 | 1.5397 | 124.6461 | 0.6995 | 124.6424 | 1.6027 | 0.5959 | 0.1128 |
| Aussenspule | Re | 0.5235 | 0.7876 | 2.5545 | 0.4870 | 2.5637 | 0.9013 | 0.3186 | 0.0990 |
| Aussenspule | Im | 0.7006 | 3.9280 | 892.5371 | 4.2233 | 892.5373 | 4.9691 | 0.8429 | 0.3986 |

**F_z:** vorher −25,2 N, nachher +0,545 N, Referenzpfad 0,9665 N (Ringmodell 1,00684 N). Die Kraft je
Phase (nachher): 1,14 / −2,58 / −0,046 / 3,67 N (vorher −36,8 / −23 / −13,5 / −27,3 N); der Mittelwert
über vier Phasen ist 0,545 N. **Iterationen:** äußere 24 → 19, innere 48 → 38.

## Transienter Pfad (derselbe Operator, Verhalten ändert sich absichtlich)

Erster Zeitschritt, `fembem 5 --fixed-plate --plate-z=11.3 --mesh=hex_s0p5.vtk`, äußeres GMRES,
relatives Residuum nach n Iterationen (vorher: `main`; nachher: dieser Zweig; beide Läufe liefen in
ein Zeitlimit, gemessen sind die ausgegebenen Iterationen):

| Iterationen | vorher | nachher |
|---:|---:|---:|
| 100 | 0,2 | 5,3e-3 |
| 200 | 0,106 | 1,27e-3 |
| 300 | 0,0829 | 5,9e-4 |
| 400 | 0,0748 | 3,6e-4 |
| 690 | 0,0641 | 1,04e-4 |

Der transiente Lauf wurde nicht weiter untersucht und nicht repariert; beide Läufe erreichen die
Toleranz im ersten Schritt innerhalb des Zeitlimits nicht (Zeitlimit ~400 s).

## Tests

`ctest -L host`: vorher (PR #22, `main`) 37 von 37, **nachher 39 von 39** (neu: `test_bem_coupling_geometry`,
`test_bem_calderon`). Vollständiges `ctest`: **72 von 72** grün (Fingerprint-Tests auf team28.vtk
unverändert, keine Erwartung geändert). `ctest -L tools`: 11 von 11.

Neue Tests: `test_bem_coupling_geometry` (T gegen dichte Host-Referenz, je Quad mit der Geometrie von
`perm[pos]`, auf team28.vtk; Gegenproben: alte Indexierung, altes Vorzeichenmuster, altes Vorzeichen
des ½T-Terms), Basis-Tests in `test_bem_row_check` (konstantes Feld, drei Seitenverhältnisse,
Gegenprobe alte Funktion), `test_bem_calderon` (Würfel, Punktquelle; Gegenprobe Innenform).
Angepasst: `test_coupling_sign_convention` (siehe Schritt 2), `test_singular_quadrature` (Schritt 3).

## Offene Punkte

- **Zielwert Zeile 2 nicht erreicht:** 4,9 % statt etwa 3 %; Platte 19,6 %.
- **Das Feld ist noch nicht richtig** (F_z −44 % gegen den Referenzpfad, Imaginärteile der Spulen zu
  groß, B_φ/B 0,06 bis 0,10). Die K1/K2-Formulierungsfrage ist nicht Teil dieses Auftrags und
  unverändert; ob sie den Rest erklärt, ist nicht geprüft.
- Der Calderón-Würfeltest erreicht 9,6 % (Außenform), nicht „wenige Prozent“: die Grenze der
  Tangentialdarstellung an Kanten (D kennt A·n nicht). Auf zylinderartigen Flächen (Spulen) sind es
  2 bis 8 %.
- Die K-Zeile und der Vorkonditionierer: blocktriangulär mit −T und S~ = µ0 G ist von den
  Vorzeichenänderungen nicht betroffen (S~ enthält kein ½T); `test_block_triangular_consistency` ist
  grün. Die Konvergenz des Vorkonditionierers bei 50 Hz ist nicht gesondert untersucht.
- Nach Schritt 2 allein (ohne 3) ist der Operator schlechter als vorher; die Schritte sind nur
  zusammen sinnvoll (jeder wurde einzeln committet, damit die Messung je Schritt nachvollziehbar ist).
- Die Annahmen des Auftrags waren nicht zu korrigieren; alle vier Befunde trafen zu.
- Nicht geprüft: 3,8 mm, andere Netze als hex_s1 und hex_s0p5 für die Abschlussmessung, der Imaginärteil
  der Zeilenmessung (2 Hz, Realteil).

## Befehle

Aus `build/`:

```
python3 ../tools/gmsh/team28_hex.py --scale 1.0 -o rowcheck/hex_s1.vtk
python3 ../tools/gmsh/team28_hex.py --scale 0.5 -o rowcheck/hex_s0p5.vtk
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=rowcheck/hex_s1.vtk --bem-row-check
./fembem --reference --harmonic=50 --fixed-plate --plate-z=11.3 --mesh=rowcheck/hex_s1.vtk \
    --harmonic-field-dump=fix/ref_s1_50.csv
./fembem --harmonic=50 --harmonic-inner-direct --fixed-plate --plate-z=11.3 \
    --mesh=rowcheck/hex_s1.vtk --harmonic-field-dump=fix/after_s1_50.csv
python3 ../tools/compare_fields.py fix/after_s1_50.csv fix/ref_s1_50.csv --mesh rowcheck/hex_s1.vtk \
    --biot-savart --freq 50 --plate-z0 11.3 --markdown
./test_bem_coupling_geometry ../data/team28/team28.vtk
./test_bem_calderon
```
