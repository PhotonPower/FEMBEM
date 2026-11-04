# Gegenprobe 2: das exakte Biot-Savart-Paar in den Blockzeilen des harmonischen BEM-FEM-Pfads

Stand: 04.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), Linux
(nicht RTX 3090 / Windows). Programm: Zweig `bem-row-check`, Basis `main` 7f17630. Netze aus
`tools/gmsh/team28_hex.py` (`--scale 1.0` und `--scale 0.5`), 2 Hz, Platte bei 11,3 mm
(`--plate-z=11.3`). Logs und Netze in `build/rowcheck/` (nicht im Repo). **Nur Diagnose:
kein Reparaturversuch, keine Änderung an Formulierung, Operatoren, Matrizen oder Lösung.**

## Frage

`bemfem-field-check.md` und `field-comparison-reference-vs-bemfem.md`: Das Feld des
harmonischen BEM-FEM-Pfads ist schon bei 2 Hz falsch, FEM-Matrix und Quellterm stimmen auf den
Zeilen ohne BEM. Vermutet war, dass die BEM-Zeilen µ0·K mit ∂A/∂n identifizieren, die
FEM-Randterme aber n × H liefern, und beide sich um ∇_t(A·n) unterscheiden. Gegenprobe 2 setzt
das exakte Paar (a_ex, K) in **beide** Blockzeilen des produktiven Pfads ein.

## Ergebnis in Kürze

**Gemessen** (Zahlen in den Abschnitten darunter; „Skalierung 1,0“ falls nicht anders genannt)

1. **Die Vermutung zu K1 und K2 trifft zu, aber in anderer Form.** Zeile 1 (FEM) wird vom
   exakten Paar mit **K2** (µ0 K2 = −n × B) erfüllt, nicht mit K1 und nicht mit −K2: sobald das
   Randintegral mit der echten Nédélec-Spur der FEM-Testfunktionen gebildet wird (ohne T und
   ohne Rooftop-Darstellung), liegt das Residuum bei 7,4 % (Skalierung 0,5: 16,2 %); auf ebenen
   Plattenquads bei 0,6 %. Mit K1: 29,4 %, mit −K2: 200,9 %. Zeile 2 (BEM) wird mit **K1**
   (µ0 K1 = (∂A/∂n)_tan) erfüllt, auf den Seitenquads der Spulen mit 6,9 % (innen) und 1,3 %
   (außen), mit K2 dort 32,1 % bzw. 7,5 % – **aber nur mit umgekehrtem Vorzeichen des D-Terms**
   und mit konsistenter Darstellung der Randspur (siehe 3 und 4). Auf ebenen Quads sind K1 und
   K2 gleich (||k1 − k2||/||k1|| = 0,0000), auf Seitenquads der Spulen 0,42 (innen) bzw. 0,27
   (außen), an der Plattenwand 0,15.
2. **Die Operatoren, wie sie im Code stehen, erfüllen keine der beiden Zeilen mit dem
   exakten Paar**: Residuen 72 % und mehr in allen Körpern und Klassen, bei jedem K und
   jedem Vorzeichen von K, **auch auf ebenen Quads, wo K1 = K2 ist**. Das ist kein
   Formulierungseffekt (∇_t(A·n) = 0 dort), sondern ein Umsetzungsbefund.
3. **Vier Umsetzungsbefunde, jeder einzeln gemessen** (Ursachen im Code im Abschnitt „Was die
   Zeilen wirklich sind“):
   a. Die Randspur: aus den Kantenwerten a_ex rekonstruiert die Rooftop-Basis nicht
      A_tan, sondern ein Vielfaches (Projektion 1,4 (Innenspule), 3,5 bis 6,1 (Außenspule), 2,3
      (Platte), 17 an der Plattenwand; im Mittel 3,57; die Vorhersage 2a/b für eine
      Rechteckzelle wird im Host-Test auf 1e-10 getroffen).
   b. Der T-Kernel kombiniert die Geometrie `faces[pos]` (Netzreihenfolge) mit den
      Freiheitsgraden der Position `pos` (Baumreihenfolge, `perm[pos]`). Bei dieser Gegenprobe
      ist `perm[pos] != pos` für 3696 von 3696 Randquads. Mit der Geometrie des Quads
      `perm[pos]` ändert sich T k um 94 % (Zeile 1) bzw. 91 % (Zeile 2).
   c. Das Vorzeichenmuster der T-Anwendung (`sign[r]·sign[c]` in beiden Zeilen) passt nicht
      zu dem der G- und D-Blöcke (roh); die Variante mit konsistenten Vorzeichen ändert T k um
      das 9-fache (Zeile 1) bzw. 1,8-fache (Zeile 2).
   d. Die D-Zeile ist **die Innenraum-Identität** (½ + D)u = G ∂u/∂n mit nach außen zeigender
      Normale. Für das Luftgebiet (Außenraum) gilt (½ − D)u = −G ∂u/∂n. Mit dem
      D-Vorzeichen umgekehrt und konsistenter Darstellung (Galerkin-Projektion von A_tan und K
      in den Rooftop-Raum) verschwindet Zeile 2 mit −K1 auf 2,7 % (Skalierung 0,5: 4,1 %); mit
      dem Vorzeichen des Codes sind es 945 % bis 1060 % (Skalierung 1,0, alle Fälle).
4. Nicht belegt, nur kontrolliert: das exakte Feld selbst (A_ex, B_ex, ∂A/∂n) stimmt (Tests
   unten); das Randintegral der FEM-Zeile mit K2 stimmt, d. h. die FEM-Seite und die exakten
   Felder sind miteinander verträglich.

**Interpretation (nicht bewiesen):**

- Die Reihenfolge der Befunde 3a bis 3d macht die BEM-Kopplung unabhängig von der
  Formulierungsfrage (K1 gegen K2) inkonsistent. Ob 3a bis 3d zusammen das falsche Feld
  erklären, ist hier nicht geprüft; es wurde nichts repariert und kein Lauf mit geänderten
  Operatoren gemacht.
- Zeile 1 verlangt K2, Zeile 2 (richtig geschrieben) K1. Auf Seitenquads unterscheiden sie sich
  um ∇_t(A·n)/µ0; das passt zur Vermutung des Berichts und ist hier das einzige Mal, dass sie
  messbar wird: auf ebenen Quads fällt der Unterschied weg.

## Was die Zeilen wirklich sind (Aufgabe 1)

Der Code gilt, wo die Annahmen des Auftrags abweichen (dort jeweils vermerkt).

**Blockstruktur** (`CoupledSolver.cu`, `apply_physical`, `apply_bem_blocks`): mit x = (a, k)

    Zeile 1 (FEM-Kanten):   y_A = P·A_sys·P·a + (I−P)·a  −  T k
    Zeile 2 (BEM, je Quad):  y_K = −½ T a − D_BEM a + µ0 G_BEM k

Die ganze K-Gleichung ist mit −1 multipliziert (rechte Seite null). Die Zeile 2 des Auftrags
`(−0,5 T − D_BEM) a_ex + µ0 G_BEM k` ist damit wörtlich die des Codes.

- **A_sys im harmonischen Pfad ist W = K + ωC** (`csr_matrix`), nicht K. Die Gegenprobe
  rechnet Zeile 1 in der K-Form W a − ωC a − b (C = `csr_C`, auf der Platte relevant) und
  druckt die W-Form zusätzlich (Tabelle unten; Unterschied der Platte gegen die K-Form 0,02 % bei Skalierung 1,0, 0,1 % bei 0,5).
- **Eichung:** `apply` projiziert a mit P (Baum-Kobaum-Eichung) vor FEM-SpMV, T und D. Das
  exakte a_ex ist nicht baumgeeicht. Die Gegenprobe wendet die Operatoren deshalb **ohne**
  Eichprojektion an (`row_check_parts`), also auf das nackte exakte Paar.
- **Testfunktionen.** Zeile 1: die Nédélec-Kantenfunktion des Hexaeders (Kanten-Freiheitsgrad =
  Kantenintegral, global orientiert). Der Randterm steht aber **nicht** als ∮ W_Nédélec · K im
  Code, sondern als T k mit der **Rooftop**-Massenmatrix T_ij = ∫ W_i · W_j dS
  (`integrate_boundary_mass_matrix`), d. h. mit der Rooftop-Funktion als Testfunktion. Zeile
  2: die Rooftop-Funktion des Quads, Kernel Vektor-Einfachschicht 1/(4πR) bzw. DLP
  (r_p − r_q)·n_q/(4πR³) auf dem Skalarprodukt W_i · W_j, **nur Tangentialkomponente** von A
  und K (der Normalanteil A·n der Randspur kommt in D nicht vor).
- **Bedeutung von T:** die Rooftop-Massenmatrix des Quads; blockdiagonal, wird je Quad
  ausgewertet (`CouplingKernels.cuh`). Der Auftrag nimmt „T = Massenmatrix“ an; das stimmt,
  aber die Funktionen sind die Rooftop-, nicht die Nédélec-Spur.
- **Rooftop-Basis** (`evaluate_rooftop_basis`, `BemIntegration.cuh`): W_i = c_i(u,v)·∂r/∂u
  bzw. ∂r/∂v dividiert durch detJ, also die **kontravariante** (Fluss-)Piola-Abbildung, nicht
  die kovariante. Für ein Rechteck mit Halbseiten a (u) und b (v) hat das Kantenintegral
  der u-Kantenfunktion den Wert 2a/b (statt 1), das der v-Kantenfunktion 2b/a. Gemessen im
  Host-Test: Rekonstruktion eines konstanten Felds A_x = 1 aus den Kantenintegralen liefert
  2,0 / 4,0 / 1,0 für a/b = 1 / 2 / 0,5.
- **Raum von K:** 4 Koeffizienten je Quad, quad-lokal und über Quadgrenzen unstetig, in
  **Baumreihenfolge** (`perm`, `BemCoupling.h`); K(x) = Σ c_i W_i(x) mit c_i zur lokalen Kante
  {i, i+1}. Gegen a: globale Kantenwerte, gather `x_bem[pos·4+e] = sign·a[bem_to_fem]`; der
  Operator identifiziert damit den Rooftop-Koeffizienten mit dem (vorzeichenkorrigierten)
  Kantenintegral.
- **Normalenorientierung:** `evaluate_quad_normal` = normiertes ∂r/∂u × ∂r/∂v. Gemessen an der
  Lage zum Elternhexaeder: **0 von 936 (Skalierung 0,5) bzw. 0 von 3696 (1,0) Quads zeigen nach
  innen**, die Normale ist also die äußere des Leiters. Die Gegenprobe benutzt n = diese äußere
  Normale für K1 und K2.
- **µ0:** `G_BEM` wird mit µ0 multipliziert (`m_mu_0`), T und D nicht. Die
  `apply_T_matrix_kernel`-Beiträge tragen `sign[r]·sign[c]·T_loc[r][c]` in beiden Zeilen.
- **Geometrie des T-Kernels:** `apply_T_matrix_kernel_body` liest `faces[face_idx]` mit
  `face_idx` = Position in der BEM-Zählung; `setup_bem_mesh` bekommt
  `mesh.boundary_faces.data()` in Netzreihenfolge, `bem_to_fem`/`sign` aber sind
  permutiert (`fem.cu`, `d_bem_to_fem[i] = bem_map.bem_to_fem[i]`). Die G- und D-Blöcke nehmen
  dagegen `perm[…]` (`fem.cu`, Blockschleife).

## Exakte Felder (Aufgabe 2)

`include/BemRowCheck.h`: Vektorpotential und Induktion einer Kreisschleife in geschlossener Form
(vollständige elliptische Integrale über das AGM), TEAM-28-Spulen als Summe über
Querschnittsfäden wie `tools/team28_field_check.py`: Innenspule r 27 bis 55 mm, 960 Windungen,
Außenspule r 80 bis 95 mm, 576 Windungen, gegensinnig, z −52 bis 0 mm, 20 A Scheitelstrom;
Standard-Fadenabstand 0,5 mm (8944 Fäden). (n·∇)A ist **analytisch**: aus A_φ, B_r, B_z mit
∂A_φ/∂z = −B_r und ∂A_φ/∂r = B_z − A_φ/r, kartesisch
(n·∇)A = (n·∇A_φ) e_φ − (A_φ/r)(n·e_φ) e_r.

Prüfungen (`test_bem_row_check`, Host):

| Prüfung | Messwert |
|---|---|
| K(0,5), E(0,5) | 2e-16 / 1e-16 |
| Schleife: geschlossene Form gegen Umfangsintegral (A_φ, B_r, B_z) | 9e-15 |
| rot A_ex = B_ex (zentrale Differenz, h = 1e-6) | 1,2e-7 |
| (n·∇)A analytisch gegen zentrale Differenz | 4,4e-8 |
| B_z auf der Achse gegen `BiotSavart.cuh` (homogene Stromdichte): Faden 1 mm / 0,5 mm | 8,3e-4 / 2,1e-4 |

Gegenproben im Test: falscher Radius (2,1e-2), ohne Krümmungsterm in (n·∇)A (0,19).

**Fehler des Fadenmodells.** Es ist eine Mittelpunktsregel der Stromverteilung. Auf der Achse
2,1e-4 (0,5 mm). Auf der Spulenoberfläche, wo K1 und K2 ausgewertet werden, liegt die nächste
Fadenreihe nur 0,25 mm entfernt; gemessen als Streuung der Kennzahlen mit dem Fadenabstand
(Skalierung 0,5, alle Körper):

| Fadenabstand | Zeile 1 mit exakten Randintegralen, +K2 | Zeile 2 konsistent, D umgekehrt, −K1 | −K2 |
|---|---|---|---|
| 1,0 mm | 16,00 % | 7,88 % | 12,66 % |
| 0,5 mm | 16,19 % | 4,08 % | 9,31 % |
| 0,25 mm | 15,98 % | 6,79 % | 11,79 % |

Zeile 1 ist dagegen unempfindlich. Bei Zeile 2 beträgt die Streuung 3 bis 4 Prozentpunkte, nicht
monoton. **Die Zeile-2-Werte (K1, K2) sind damit auf etwa ±3 Prozentpunkte unsicher.** Die
Schlüsse unten hängen nicht daran (200 % gegen 3 bis 10 %).

## Diskrete Größen (Aufgabe 3)

- a_ex: Kantenintegral von A_ex über jede der Kanten des Netzes (nicht nur Randkanten), 4-Punkt-Gauß,
  Richtung wie `mesh.edges` (v_min → v_max).
- k1, k2: je Randquad Galerkin-Projektion in den Rooftop-Raum, T c = f mit der Operator-Massenmatrix
  (`integrate_boundary_mass_matrix`) und f_i = ∫ W_i · F dS (16-Punkt-Gauß); F1 = [(n·∇)A]_tan/µ0,
  F2 = −n × B/µ0. Test: Rundlauf Koeffizienten → Feld → Projektion 4e-16; ohne Massenmatrix 0,39.
  Alle Quads haben eine reguläre Massenmatrix.
- Zusätzlich (Messung, nicht Teil des Auftrags): `cA` = Galerkin-Projektion der tangentialen
  Randspur P_t A_ex in den Rooftop-Raum (für die konsistente Darstellung), und ein
  **Referenz-Randterm** g_v[r] = ∫ W_Nédélec,r · K_v dS mit der kovarianten Spurfunktion
  (W_0 = ¼(1−v) g^u, W_1 = ¼(1+u) g^v, W_2 = −¼(1+v) g^u, W_3 = −¼(1−u) g^v mit der reziproken
  Basis g^u, g^v der Fläche), mal Kantenvorzeichen.

## Aufbau der Diagnose (Aufgabe 4)

`fembem --harmonic[=HZ] --fixed-plate --plate-z=H --mesh=NETZ --bem-row-check[=MM]`, nur mit
`--harmonic` (ohne `--reference`; sonst Exit 1). MM ist der Fadenabstand (Standard 0,5). Die
Diagnose läuft nach dem Aufbau von Operator und rechter Seite, **vor** dem Lösen (die rechte
Seite wird wie im Felddump in den physikalischen Raum zurückgerechnet), und endet mit Exit 0
ohne Lösung. Ohne die Option bleibt der Pfad unverändert (Gegenprobe unten).

**Eine Änderung am Operatorcode, rein additiv, gemeldet:** `CoupledMVOperator::row_check_parts`
(`CoupledSolver.cuh/.cu`) ruft die bestehenden Kernel und Blöcke einzeln auf ein vorgegebenes
Paar (a, k) und liefert −T k, −½ T a, −D a, +µ0 G k getrennt, ohne Eichprojektion, ohne
Äquilibrierung. Kein bestehender Operator wurde angefasst. Wer das für eine Änderung am
Operatorcode im Sinn des Auftrags hält, kann den Commit
(`bem-row-check 2`) einzeln zurücknehmen; die Tabellen oben entstehen dann nicht mehr.

Zeile 1 wird auf den Randkanten ausgewertet, Bezug ||K a − b||; Zeile 2 je BEM-Freiheitsgrad,
Bezug ||(−½ T − D) a||. Kantenklassen für Zeile 1: beide angrenzenden Quads dieselbe Klasse →
diese, sonst „Kante“.
**Quadklassen:** eben |n_z| > 0,99; Kante: nicht eben und eine der vier Kanten wird mit einem
Quad geteilt, dessen Normale um mehr als etwa 26° abweicht (|n·n'| < 0,9); Seite: Rest. Die
Facetten des 48-Ecks (7,5°) zählen nicht als Kante.

Kontrollen im Lauf:

Skalierung 1,0:

```
3696 Randquads, 18651 FEM-Kanten, 14784 BEM-Freiheitsgrade (7392 mit Vorzeichen -1)
Normalenorientierung: 0 von 3696 Quads haben eine nach INNEN zeigende evaluate_quad_normal()-Normale
BEM-Reihenfolge: 3696 von 3696 Positionen mit perm[pos] != pos
  Platte      Quads: eben 1248, Seite 48, Kante 96
  Innenspule  Quads: eben 384, Seite 576, Kante 192
  Aussenspule Quads: eben 384, Seite 576, Kante 192
Quelle: 8944 Faeden (Abstand 0.5 mm), Scheitelstrom 20 A
innere Zeilen Platte      ||K a_ex - b|| = 2.360e+00 (||b|| = 0)
innere Zeilen Innenspule  ||K a_ex - b|| / ||b|| = 24.40 % (Bezug: nur Zeilen ohne BEM; b = 0 in der Platte); ||K a_ex|| / ||b|| = 1.0259, bester Faktor <K a_ex, b>/<b, b> = 0.9964, Kosinus = 0.9713
innere Zeilen Aussenspule ||K a_ex - b|| / ||b|| = 24.27 % (Bezug: nur Zeilen ohne BEM; b = 0 in der Platte); ||K a_ex|| / ||b|| = 1.0258, bester Faktor <K a_ex, b>/<b, b> = 0.9967, Kosinus = 0.9716
T-Kontrolle (Operator gegen Host, k = k1): Zeile 1 5.23e-16, Zeile 2 8.77e-17 (Soll ~1e-12: die Nachrechnung gibt den Operator wieder)
T-Varianten gegen Operator: V1 (Geometrie perm[pos]) Zeile 1 9.39e-01, Zeile 2 9.07e-01; V2 (zusaetzlich konsistente Vorzeichen) Zeile 1 9.07e+00, Zeile 2 1.79e+00
Normen (V0, nur Randkanten): ||K a - b|| = 3.3358e+04, ||T k1|| = 8.9942e+03, ||T k2|| = 9.9625e+03, ||W a - b|| = 3.3358e+04
Normen (konsistent): ||(-0.5 T - D) cA|| = 1.3822e-04, ||mu0 G k1|| = 1.3879e-03, ||mu0 G k2|| = 1.3730e-03
Normen (D umgekehrt): ||(-0.5 T + D) cA|| = 1.3929e-03
```

Skalierung 0,5:

```
936 Randquads, 3086 FEM-Kanten, 3744 BEM-Freiheitsgrade (1872 mit Vorzeichen -1)
Normalenorientierung: 0 von 936 Quads haben eine nach INNEN zeigende evaluate_quad_normal()-Normale
BEM-Reihenfolge: 936 von 936 Positionen mit perm[pos] != pos
  Platte      Quads: eben 312, Seite 0, Kante 48
  Innenspule  Quads: eben 96, Seite 96, Kante 96
  Aussenspule Quads: eben 96, Seite 96, Kante 96
innere Zeilen Platte      ||K a_ex - b|| = 8.348e+00 (||b|| = 0)
innere Zeilen Innenspule  ||K a_ex - b|| / ||b|| = 68.96 % (Bezug: nur Zeilen ohne BEM; b = 0 in der Platte); ||K a_ex|| / ||b|| = 1.2044, bester Faktor <K a_ex, b>/<b, b> = 0.9875, Kosinus = 0.8199
innere Zeilen Aussenspule ||K a_ex - b|| / ||b|| = 68.76 % (Bezug: nur Zeilen ohne BEM; b = 0 in der Platte); ||K a_ex|| / ||b|| = 1.2046, bester Faktor <K a_ex, b>/<b, b> = 0.9891, Kosinus = 0.8211
T-Kontrolle (Operator gegen Host, k = k1): Zeile 1 2.61e-16, Zeile 2 8.26e-17 (Soll ~1e-12: die Nachrechnung gibt den Operator wieder)
T-Varianten gegen Operator: V1 (Geometrie perm[pos]) Zeile 1 9.22e-01, Zeile 2 9.53e-01; V2 (zusaetzlich konsistente Vorzeichen) Zeile 1 4.83e+00, Zeile 2 1.81e+00
Normen (V0, nur Randkanten): ||K a - b|| = 3.2652e+04, ||T k1|| = 1.6893e+04, ||T k2|| = 1.8327e+04, ||W a - b|| = 3.2653e+04
Normen (konsistent): ||(-0.5 T - D) cA|| = 1.4088e-04, ||mu0 G k1|| = 1.3605e-03, ||mu0 G k2|| = 1.3433e-03
Normen (D umgekehrt): ||(-0.5 T + D) cA|| = 1.3900e-03
```

Die T-Nachrechnung auf dem Host (gleiche Indexierung wie der Operator) gibt den Operator auf
5e-16 wieder; die beiden Varianten V1 und V2 sind danach echte Messungen und keine
Rechenfehler der Nachrechnung.

**Innere Zeilen.** ||K a_ex − b||/||b|| auf den Zeilen ohne BEM: 24,4 % (Innenspule) und
24,3 % (Außenspule) bei Skalierung 1,0, 69,0 % und 68,8 % bei 0,5; Kosinus 0,971 bzw. 0,82,
bester Faktor 0,996 bzw. 0,99. Das ist die Konsistenz von Kantenfunktion und Quellterm bei
sehr groben Zellquerschnitten und fällt mit dem Netz. **Der Wert 2,8 % / 1,6 % aus
`bemfem-field-check.md` ist hier nicht reproduziert** (anderes Netz `team28.vtk`, andere
Zeilenauswahl; nicht untersucht, siehe offene Punkte).

## Ergebnisse, Skalierung 1,0, 2 Hz, Platte bei 11,3 mm

Spalten: +K1, −K1, +K2, −K2 sind die vier Fälle k = ±k1 und ±k2; „Residuum“ ist die relative
Norm in Prozent.

### Die Operatoren, wie sie sind (V0)

Zeile 1: ||K a − T k − b|| / ||K a − b||

| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | eben | 2448 | 119.86 % | 118.89 % | 119.86 % | 118.89 % |
| Platte | Seite | 48 | 4474.59 % | 4459.34 % | 12103.85 % | 12066.25 % |
| Platte | Kante | 288 | 145.11 % | 194.74 % | 160.40 % | 213.78 % |
| Platte | alle | 2784 | 120.20 % | 120.12 % | 120.44 % | 120.52 % |
| Innenspule | eben | 672 | 106.30 % | 104.59 % | 106.30 % | 104.59 % |
| Innenspule | Seite | 1056 | 101.18 % | 102.06 % | 101.53 % | 103.00 % |
| Innenspule | Kante | 576 | 102.95 % | 102.18 % | 103.67 % | 102.69 % |
| Innenspule | alle | 2304 | 102.46 % | 102.49 % | 102.85 % | 103.17 % |
| Aussenspule | eben | 672 | 126.22 % | 126.93 % | 126.22 % | 126.93 % |
| Aussenspule | Seite | 1056 | 102.93 % | 101.10 % | 104.20 % | 101.64 % |
| Aussenspule | Kante | 576 | 105.65 % | 101.94 % | 108.58 % | 105.07 % |
| Aussenspule | alle | 2304 | 105.75 % | 103.68 % | 107.34 % | 104.85 % |
| alle | alle | 7392 | 103.88 % | 103.26 % | 104.65 % | 104.08 % |

Zeile 2: ||(−½ T − D) a + µ0 G k|| / ||(−½ T − D) a||

| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | eben | 4992 | 92.07 % | 112.26 % | 93.06 % | 113.09 % |
| Platte | Seite | 192 | 85.77 % | 117.63 % | 85.99 % | 117.34 % |
| Platte | Kante | 384 | 83.58 % | 124.81 % | 83.65 % | 124.56 % |
| Platte | alle | 5568 | 90.31 % | 114.52 % | 91.11 % | 115.06 % |
| Innenspule | eben | 1536 | 106.18 % | 123.07 % | 108.51 % | 126.70 % |
| Innenspule | Seite | 2304 | 110.38 % | 118.54 % | 112.87 % | 122.58 % |
| Innenspule | Kante | 768 | 110.54 % | 128.93 % | 114.16 % | 134.72 % |
| Innenspule | alle | 4608 | 109.10 % | 120.90 % | 111.64 % | 124.97 % |
| Aussenspule | eben | 1536 | 71.81 % | 134.85 % | 72.16 % | 134.27 % |
| Aussenspule | Seite | 2304 | 73.18 % | 132.62 % | 73.45 % | 132.02 % |
| Aussenspule | Kante | 768 | 72.55 % | 131.04 % | 73.02 % | 130.39 % |
| Aussenspule | alle | 4608 | 72.73 % | 132.91 % | 73.05 % | 132.31 % |
| alle | alle | 14784 | 74.16 % | 132.37 % | 74.57 % | 131.89 % |

Zeile 1 mit W statt K (ωC·a_ex mitgerechnet; relevant nur für die Platte):

| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | eben | 2448 | 119.85 % | 118.86 % | 119.85 % | 118.86 % |
| Platte | Seite | 48 | 4474.59 % | 4459.34 % | 12103.85 % | 12066.25 % |
| Platte | Kante | 288 | 144.24 % | 193.73 % | 159.33 % | 212.59 % |
| Platte | alle | 2784 | 120.18 % | 120.09 % | 120.42 % | 120.49 % |
| Innenspule | eben | 672 | 106.30 % | 104.59 % | 106.30 % | 104.59 % |
| Innenspule | Seite | 1056 | 101.18 % | 102.06 % | 101.53 % | 103.00 % |
| Innenspule | Kante | 576 | 102.95 % | 102.18 % | 103.67 % | 102.69 % |
| Innenspule | alle | 2304 | 102.46 % | 102.49 % | 102.85 % | 103.17 % |
| Aussenspule | eben | 672 | 126.22 % | 126.93 % | 126.22 % | 126.93 % |
| Aussenspule | Seite | 1056 | 102.93 % | 101.10 % | 104.20 % | 101.64 % |
| Aussenspule | Kante | 576 | 105.65 % | 101.94 % | 108.58 % | 105.07 % |
| Aussenspule | alle | 2304 | 105.75 % | 103.68 % | 107.34 % | 104.85 % |
| alle | alle | 7392 | 103.88 % | 103.26 % | 104.65 % | 104.08 % |

**Messung:** Zeile 1: Platte 120 %, Innenspule 102 bis 103 %, Außenspule 104 bis 107 %; an der
Plattenwand (48 Mittelkanten) 4 500 % bis 12 100 %. Zeile 2: 73 % bis 133 %; kleinster Wert 71,8 %
(Außenspule eben, +K1). Kein Vorzeichen und kein K senkt eine der Zeilen unter 70 %. Die ebenen Quads (K1 = K2) haben dieselben Werte in K1- und K2-Spalten (Zeile 1: Platte 120 %,
Innenspule 106 %, Außenspule 126 %) und sind **nicht besser** als Seiten und Kanten: ein Umsetzungsfehler, keine Formulierungsfrage.
Die Vorzeichenfrage K2 (und a_ex) ist damit **offen und nicht entschieden**; keines der Vorzeichen
wurde als „richtig“ gewählt.

### Zeile 1 mit exakten Randintegralen (ohne T, ohne Rooftop)

||K a − g − b|| / ||K a − b|| mit g = ∫ W_Nédélec · K dS; die Spalten wie oben.

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

**Messung:** +K2 erfüllt Zeile 1 (7,4 % gesamt, Innenspule/Außenspule Seiten 6,5 % / 5,5 %,
Plattenquads eben 0,6 %); −K2 und −K1 liegen bei rund 200 %, +K1 bei 29 %. Auf ebenen Quads
sind K1 und K2 gleich (9,8 % / 11,8 % an den Spulendeckeln, 0,6 % an der Platte). Die
Zeile „Platte, Seite“ (48 Kanten, Mittelkanten der Plattenwand) zeigt 100 % in allen Spalten;
dort ist g ≈ 0, während K a − b nicht verschwindet (nicht weiter untersucht).

### T gegen das exakte Randintegral

||T k − g|| / ||g||, V0 = Operator, V1 = Geometrie perm[pos], V2 = V1 plus konsistente
Vorzeichen (nur Zeilen „alle“).

| Variante | Koerper | Klasse | n | K1 | K2 |
|---|---|---|---:|---:|---:|
| V0 | Platte | alle | 2784 | 120.13 % | 120.26 % |
| V0 | Innenspule | alle | 2304 | 103.27 % | 102.75 % |
| V0 | Aussenspule | alle | 2304 | 106.36 % | 107.24 % |
| V0 | alle | alle | 7392 | 104.76 % | 104.53 % |
| V1 | Platte | alle | 2784 | 113.91 % | 117.54 % |
| V1 | Innenspule | alle | 2304 | 100.59 % | 100.46 % |
| V1 | Aussenspule | alle | 2304 | 101.46 % | 101.71 % |
| V1 | alle | alle | 7392 | 101.25 % | 101.22 % |
| V2 | Platte | alle | 2784 | 203.12 % | 220.93 % |
| V2 | Innenspule | alle | 2304 | 94.85 % | 74.76 % |
| V2 | Aussenspule | alle | 2304 | 287.31 % | 271.34 % |
| V2 | alle | alle | 7392 | 183.88 % | 164.80 % |

### Zeile 2 mit konsistenter Darstellung

Hier ist die Randspur P_t A_ex ebenso wie K per Galerkin in den Rooftop-Raum projiziert
(`cA`, `k`); T mit Geometrie `perm[pos]` und roher Testfunktion; D und G die bestehenden Blöcke.
Vorzeichen von D wie im Code (−½ T − D):

| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | eben | 4992 | 340.79 % | 197.01 % | 384.75 % | 230.73 % |
| Platte | Seite | 192 | 11848.14 % | 12004.52 % | 11701.40 % | 11858.27 % |
| Platte | Kante | 384 | 1139.57 % | 941.47 % | 1129.01 % | 931.52 % |
| Platte | alle | 5568 | 394.14 % | 265.77 % | 430.88 % | 289.82 % |
| Innenspule | eben | 1536 | 589.80 % | 414.48 % | 643.42 % | 463.69 % |
| Innenspule | Seite | 2304 | 413.01 % | 302.91 % | 467.14 % | 321.92 % |
| Innenspule | Kante | 768 | 508.40 % | 473.36 % | 584.29 % | 513.44 % |
| Innenspule | alle | 4608 | 459.50 % | 342.62 % | 514.85 % | 370.67 % |
| Aussenspule | eben | 1536 | 4024.26 % | 4164.45 % | 3957.93 % | 4101.08 % |
| Aussenspule | Seite | 2304 | 1086.74 % | 917.07 % | 1062.67 % | 900.64 % |
| Aussenspule | Kante | 768 | 2206.14 % | 2153.53 % | 2155.38 % | 2112.99 % |
| Aussenspule | alle | 4608 | 1336.24 % | 1221.75 % | 1308.41 % | 1200.76 % |
| alle | alle | 14784 | 1059.58 % | 955.92 % | 1049.33 % | 944.57 % |

Vorzeichen des D-Terms **umgekehrt** (−½ T + D) (Messung, nicht der Code):

| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | eben | 4992 | 199.71 % | 16.46 % | 215.24 % | 27.31 % |
| Platte | Seite | 192 | 186.55 % | 16.43 % | 185.31 % | 19.02 % |
| Platte | Kante | 384 | 196.26 % | 10.96 % | 195.13 % | 14.06 % |
| Platte | alle | 5568 | 196.95 % | 15.29 % | 206.16 % | 23.52 % |
| Innenspule | eben | 1536 | 200.26 % | 9.50 % | 210.52 % | 17.10 % |
| Innenspule | Seite | 2304 | 201.73 % | 6.94 % | 211.68 % | 32.08 % |
| Innenspule | Kante | 768 | 201.42 % | 6.08 % | 212.89 % | 27.27 % |
| Innenspule | alle | 4608 | 201.23 % | 7.74 % | 211.46 % | 27.66 % |
| Aussenspule | eben | 1536 | 199.58 % | 1.93 % | 197.98 % | 3.96 % |
| Aussenspule | Seite | 2304 | 199.52 % | 1.31 % | 197.35 % | 7.53 % |
| Aussenspule | Kante | 768 | 199.63 % | 1.68 % | 197.46 % | 6.30 % |
| Aussenspule | alle | 4608 | 199.55 % | 1.57 % | 197.54 % | 6.51 % |
| alle | alle | 14784 | 199.62 % | 2.65 % | 198.36 % | 9.16 % |

**Messung:** Mit dem Vorzeichen des Codes sind die Residuen 190 % bis über 10 000 %; ||(−½ T − D) cA||
ist dort 1,4e-4 gegen ||µ0 G k|| = 1,4e-3, die beiden Terme heben sich nicht in der Größenordnung auf.
Mit umgekehrtem D-Vorzeichen: −K1 erfüllt Zeile 2 (2,7 % gesamt; Außenspule 1,6 %, Innenspule 7,7 %,
Platte 15,3 %); +K1, +K2 liegen bei 200 %; −K2 bei 9,2 %, auf Seitenquads der Innenspule 32,1 %.
Die Platte ist mit 15 % schlechter als die Spulen (Kantenlängen und Seitenverhältnis der Panels
sind dort anders; ein Zusammenhang wurde nicht untersucht).

### K1 gegen K2

| Koerper | Klasse | n | ||k1-k2|| / ||k1|| |
|---|---|---:|---:|
| Platte | eben | 4992 | 0.0000 |
| Platte | Seite | 192 | 0.1495 |
| Platte | Kante | 384 | 0.1490 |
| Platte | alle | 5568 | 0.0156 |
| Innenspule | eben | 1536 | 0.0000 |
| Innenspule | Seite | 2304 | 0.4233 |
| Innenspule | Kante | 768 | 0.4063 |
| Innenspule | alle | 4608 | 0.3619 |
| Aussenspule | eben | 1536 | 0.0000 |
| Aussenspule | Seite | 2304 | 0.2719 |
| Aussenspule | Kante | 768 | 0.2944 |
| Aussenspule | alle | 4608 | 0.2581 |

### Randspur: Rooftop-Rekonstruktion gegen A_tan

c_i = sign_i · a_ex (Gather des Operators), Rekonstruktion Σ c_i W_i gegen den tangentialen Anteil
von A_ex, Gauß 4×4 je Quad:

| Koerper | Klasse | Quads | relativer L2-Fehler | Projektion <rec, A_t>/<A_t, A_t> | Betragsverhaeltnis |
|---|---|---:|---:|---:|---:|
| Platte | eben | 1248 | 1.4674 | 2.2587 | 2.3813 |
| Platte | Seite | 48 | 16.0040 | 17.0039 | 17.0040 |
| Platte | Kante | 96 | 16.0040 | 17.0039 | 17.0040 |
| Innenspule | eben | 384 | 0.5115 | 1.4478 | 1.4688 |
| Innenspule | Seite | 576 | 0.6833 | 1.4408 | 1.5325 |
| Innenspule | Kante | 192 | 0.6158 | 1.3752 | 1.4593 |
| Aussenspule | eben | 384 | 5.1320 | 6.1185 | 6.1298 |
| Aussenspule | Seite | 576 | 2.5918 | 3.5750 | 3.5871 |
| Aussenspule | Kante | 192 | 2.5823 | 3.5656 | 3.5776 |
| alle | alle | 3696 | 2.9111 | 3.5732 | 3.8237 |

**Messung:** die Projektion <rec, A_t>/<A_t, A_t> und das Betragsverhältnis liegen zwischen 1,4 (Innenspule)
und 6,1 (Außenspule ebene Quads) sowie 17 an den Seitenwänden der dünnen Platte; Soll 1. Das folgt dem
Seitenverhältnis der Quads (Host-Test: 2a/b).

## Skalierung 0,5 gegen 1,0

Nur die Zeilen „alle“ je Körper (zuerst 1,0, dann 0,5):

Zeile 1, Operatoren wie sie sind (V0):

| Koerper | Skalierung | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | 1,0 | 2784 | 120.20 % | 120.12 % | 120.44 % | 120.52 % |
| Platte | 0,5 | 720 | 162.87 % | 165.50 % | 163.71 % | 166.55 % |
| Innenspule | 1,0 | 2304 | 102.46 % | 102.49 % | 102.85 % | 103.17 % |
| Innenspule | 0,5 | 576 | 106.85 % | 104.64 % | 107.47 % | 105.30 % |
| Aussenspule | 1,0 | 2304 | 105.75 % | 103.68 % | 107.34 % | 104.85 % |
| Aussenspule | 0,5 | 576 | 122.37 % | 122.04 % | 127.07 % | 127.03 % |
| alle | 1,0 | 7392 | 103.88 % | 103.26 % | 104.65 % | 104.08 % |
| alle | 0,5 | 1872 | 113.30 % | 111.88 % | 115.31 % | 114.04 % |

Zeile 2, Operatoren wie sie sind (V0):

| Koerper | Skalierung | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | 1,0 | 5568 | 90.31 % | 114.52 % | 91.11 % | 115.06 % |
| Platte | 0,5 | 1440 | 92.56 % | 110.55 % | 93.18 % | 110.51 % |
| Innenspule | 1,0 | 4608 | 109.10 % | 120.90 % | 111.64 % | 124.97 % |
| Innenspule | 0,5 | 1152 | 109.98 % | 118.65 % | 112.17 % | 122.45 % |
| Aussenspule | 1,0 | 4608 | 72.73 % | 132.91 % | 73.05 % | 132.31 % |
| Aussenspule | 0,5 | 1152 | 72.71 % | 132.92 % | 73.05 % | 132.29 % |
| alle | 1,0 | 14784 | 74.16 % | 132.37 % | 74.57 % | 131.89 % |
| alle | 0,5 | 3744 | 74.44 % | 132.03 % | 74.86 % | 131.51 % |

Zeile 1 mit exakten Randintegralen:

| Koerper | Skalierung | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | 1,0 | 2784 | 1.58 % | 200.31 % | 0.60 % | 200.47 % |
| Platte | 0,5 | 720 | 2.63 % | 201.74 % | 2.31 % | 201.85 % |
| Innenspule | 1,0 | 2304 | 31.53 % | 186.00 % | 7.77 % | 201.24 % |
| Innenspule | 0,5 | 576 | 32.54 % | 186.26 % | 16.89 % | 201.76 % |
| Aussenspule | 1,0 | 2304 | 25.28 % | 190.97 % | 6.94 % | 200.31 % |
| Aussenspule | 0,5 | 576 | 30.51 % | 189.94 % | 15.15 % | 199.45 % |
| alle | 1,0 | 7392 | 29.40 % | 187.85 % | 7.44 % | 200.94 % |
| alle | 0,5 | 1872 | 31.58 % | 187.74 % | 16.19 % | 201.04 % |

Zeile 2, konsistente Darstellung, D-Vorzeichen des Codes:

| Koerper | Skalierung | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | 1,0 | 5568 | 394.14 % | 265.77 % | 430.88 % | 289.82 % |
| Platte | 0,5 | 1440 | 405.68 % | 341.64 % | 437.05 % | 354.29 % |
| Innenspule | 1,0 | 4608 | 459.50 % | 342.62 % | 514.85 % | 370.67 % |
| Innenspule | 0,5 | 1152 | 453.90 % | 325.33 % | 505.03 % | 350.94 % |
| Aussenspule | 1,0 | 4608 | 1336.24 % | 1221.75 % | 1308.41 % | 1200.76 % |
| Aussenspule | 0,5 | 1152 | 1239.41 % | 1138.31 % | 1211.77 % | 1118.53 % |
| alle | 1,0 | 14784 | 1059.58 % | 955.92 % | 1049.33 % | 944.57 % |
| alle | 0,5 | 3744 | 1017.45 % | 922.00 % | 1005.04 % | 910.07 % |

Zeile 2, konsistente Darstellung, D-Vorzeichen umgekehrt:

| Koerper | Skalierung | n | +K1 | -K1 | +K2 | -K2 |
|---|---|---:|---:|---:|---:|---:|
| Platte | 1,0 | 5568 | 196.95 % | 15.29 % | 206.16 % | 23.52 % |
| Platte | 0,5 | 1440 | 190.81 % | 24.06 % | 196.74 % | 23.90 % |
| Innenspule | 1,0 | 4608 | 201.23 % | 7.74 % | 211.46 % | 27.66 % |
| Innenspule | 0,5 | 1152 | 195.56 % | 10.28 % | 204.81 % | 23.95 % |
| Aussenspule | 1,0 | 4608 | 199.55 % | 1.57 % | 197.54 % | 6.51 % |
| Aussenspule | 0,5 | 1152 | 198.04 % | 2.61 % | 195.96 % | 7.43 % |
| alle | 1,0 | 14784 | 199.62 % | 2.65 % | 198.36 % | 9.16 % |
| alle | 0,5 | 3744 | 197.85 % | 4.08 % | 196.45 % | 9.31 % |

**Messung:** Die Residuen der unveränderten Operatoren (V0) ändern sich von 0,5 auf 1,0 nur wenig
und bleiben bei 72 % und mehr. Zeile 1 mit exakten Randintegralen und +K2 sinkt von 16,2 % auf
7,4 %, Zeile 2 mit konsistenter Darstellung, umgekehrtem D und −K1 von 4,1 % auf 2,7 %. Die
inneren Zeilen sinken von 69 % auf 24 %. Der Trend ist mit zwei Netzen nicht zu einer Ordnung
auswertbar.

## Auswertung

**Welches K erfüllt Zeile 1, welches Zeile 2, mit welchem Residuum je Klasse?**
*Messung:* Operatoren unverändert: keines, auf keiner Klasse (≥ 72 %). Mit den exakten
Randintegralen: Zeile 1 mit +K2 (ebene Quads 0,6 % bis 11,8 %, Seiten 5,5 % bis 6,5 %). Zeile 2
mit konsistenter Darstellung und umgekehrtem D-Vorzeichen: −K1 (ebene Quads 1,9 % bis 16,5 %,
Seiten der Spulen 1,3 % bis 6,9 %, Plattenwand 16,4 %). Siehe die Tabellen.
*Interpretation:* Die FEM-Zeile will n × H (K2), die BEM-Zeile ∂A/∂n (K1). Das passt zur Vermutung
des Berichts; sie ist auf Seitenquads der Spulen messbar (||k1 − k2||/||k1|| 0,27 bis 0,42).

**Was ist auf ebenen Quads los?**
*Messung:* K1 = K2 (0,0000). Die Residuen der Operatoren sind dort trotzdem groß (Zeile 1: Platte 120 %,
Innenspule 106 %, Außenspule 126 %; Zeile 2: 72 % bis 135 %). Mit den Korrekturen aus den Messungen
(konsistente Darstellung, umgekehrtes D) fallen sie in Zeile 2 auf 1,9 % bis 16,5 %. Es ist ein
Umsetzungsbefund, keine Formulierungsfrage.

**Wie ändert sich alles von 0,5 auf 1,0?** Siehe oben: die Operatorwerte praktisch gleich, die
korrigierten Größen sinken (16,2 → 7,4 %, 4,1 → 2,7 %).

**Vorzeichen:** In den Tabellen stehen beide Vorzeichen von K1 und K2; a_ex wird in Zeile 2 mit
−k abgedeckt, in Zeile 1 ist sein Vorzeichen durch die rechte Seite festgelegt. Gewählt wurde keines.

## Tests

| Test | Ergebnis |
|---|---|
| `test_bem_row_check` (Host, in `ctest -L host`) | Feldauswertung (Tabelle oben), Projektion (Rundlauf 4e-16, Gegenprobe 0,39), Rooftop-Skalierung 2a/b (1e-10), Residuenbuchhaltung auf synthetischem Fall (exaktes Paar: 6e-16; Gegenproben: falsches Vorzeichen 2,0, vertauschte Zeilen 1,27) |
| `ctest -L host` | siehe Ende dieses Abschnitts |

`ctest -L host`: 37 von 37 grün (Lauf vom 04.11.2026, inklusive `test_bem_row_check`); `ctest -L tools`: 11 von 11 grün.

**Gegenprobe, Verhalten ohne die Option.** `fembem --harmonic=50 --harmonic-inner-direct
--fixed-plate --plate-z=11.3 --mesh=hex_s0p5.vtk` mit dem Programm von `main` (7f17630, eigener
Build) und mit diesem Zweig: 22 äußere Iterationen, rel. Residuum 8,35e-07, Phasenkräfte
(−0,273 / 0,164 / 0,589 / 0,151 N) und alle übrigen Zeilen des Logs sind **gleich** bis auf eine
Diagnosezeile im Rundungsbereich (`||G^T b||/||b||` 6,302e-16 gegen 6,289e-16, atomare Summation).
Logs: `build/rowcheck/gegenprobe_neu.log`, `gegenprobe_main.log`.

## Nicht erreicht, offene Punkte

- **Spulen allein** (Platte entfernt oder κ = 0) nicht gerechnet. Für den Realteil bei 2 Hz gilt die
  Verunreinigung unter 2 % aus `bemfem-field-check.md` (dort aus dem Ringmodell, hier nicht
  neu gemessen). Gemessen ist nur, dass ωC·a_ex die Zeile-1-Residuen der Platte um 0,02 % (1,0) bzw. 0,1 % (0,5) verschiebt
  (K-Form gegen W-Form).
- Die Quellfelder auf den Spulenflächen (K1, K2) sind nur auf etwa ±3 Prozentpunkte genau (Fadenmodell,
  siehe oben). Ein Volumenintegral statt Mittelpunktsfäden würde das verbessern.
- Der Wert aus `bemfem-field-check.md` für die inneren Zeilen (2,8 % / 1,6 %) ist nicht
  reproduziert (hier 24 %); anderes Netz und andere Zeilenauswahl, nicht untersucht.
- Ob die vier Befunde 3a bis 3d zusammen das falsche Feld erklären, ist **nicht geprüft** (dazu müsste
  der Operatorcode geändert werden: Randspur-Darstellung, T-Geometrie und Vorzeichen, D-Vorzeichen).
  Das war nach Auftrag nicht Teil dieser Arbeit; ich habe nicht repariert. Eine Entscheidung dazu
  liegt beim Auftraggeber.
- Die Aussage 3d (Innenraum-Identität) stützt sich auf zwei Dinge: die Herleitung
  (Handrechnung an der Einheitskugel mit u = 1/r außen: D·1 = −½, G ∂u/∂n = −1, daraus
  (½ − D)u = −G ∂u/∂n mit Normale nach außen; kein Test im Repo) und die Messung (Zeile 2
  verschwindet nur mit umgekehrtem Vorzeichen). Eine Gegenprobe der D-Quadratur an der Kugel
  gibt es nicht.
- Platte „Seite“ in Zeile 1 (48 Kanten): 100 % in allen Spalten, nicht untersucht.
- Das Seitenverhältnis (3a) wirkt auf die Platte besonders stark; ob eine höhere Auflösung das
  mildert, ist hier nicht gemessen.
- Keine Aussage über 3,8 mm oder 50 Hz; beide wurden nicht gerechnet.

## Befehle

Aus `build/` (Netze nach `rowcheck/`):

```
python3 ../tools/gmsh/team28_hex.py --scale 1.0 -o rowcheck/hex_s1.vtk
python3 ../tools/gmsh/team28_hex.py --scale 0.5 -o rowcheck/hex_s0p5.vtk
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=rowcheck/hex_s1.vtk --bem-row-check \
    > rowcheck/s1.log
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=rowcheck/hex_s0p5.vtk --bem-row-check \
    > rowcheck/s0p5.log
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=rowcheck/hex_s0p5.vtk \
    --bem-row-check=0.25 > rowcheck/s0p5_f0.25.log          # Fadenabstand
./test_bem_row_check
```

Zeit: die Diagnose selbst 6,5 s (Skalierung 0,5) bzw. 41,8 s (1,0); der Aufbau der BEM-Blöcke vor ihr
etwa 8 min bei Skalierung 1,0.
