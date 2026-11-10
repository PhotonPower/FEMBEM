# Phase 1b: Operatorbibliothek der Maxwell-Spur-Kopplung (dicht, Host), Tests T1 bis T9

Stand: 10.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), 32 CPU-Kerne, Linux; die neue Bibliothek rechnet nur auf dem Host
(Double, dicht, `std::thread`). Basis: `main` acd295e (Auftrag), Zweig `maxwell-trace-operators`. Logs der Messläufe in `build/trace/`
(nicht im Repo). Der Lösungspfad, die alten Operatoren, die Eichung und der Referenzpfad sind **unverändert** (nur neue Dateien; `CMakeLists.txt`
und `CHANGELOG.md` ergänzt). Eigen (nur Header, nur in `.cpp`) ist neu als optionale Abhängigkeit dieser Tests; ohne Eigen werden sie übersprungen.

## Ergebnis in Kürze

**Die Operatoren und das gekoppelte System (S) funktionieren diskret; T1 bis T9 bestehen mit Gegenproben. Dabei wurden vier Fehler gefunden
(zwei in der Theorie, einer im Bestand, einer in meiner ersten Implementierung) und drei Erwartungen der Theorie korrigiert.**

* Neu: `TraceSurface` (G, R, Zusammenhang, Geschlecht, harmonische Felder), `TraceBemQuad.cu` (Quadratur, V₀, D̃, M̃, V_rot, C_rot, M_rot, Außenfeld-Auswertung),
  `TraceFem.cu` (Hexaeder-K, Quelle, Rand, Baum-Kobaum des Bestands), `TraceSystem.cpp` (W, Z, Y, Q, gekoppeltes System (S) und (J)).
* **Fehler im Bestand (nicht angefasst, gemeldet):** `get_coincident_coords` (gleiches Quad) konvergiert nur algebraisch: ∫∫1/R über das Einheitsquadrat 2,8 % Fehler bei 4,
  0,75 % bei 8 Gauß-Punkten (in Python unabhängig bestätigt). Gemeinsame Kante und Ecke sind in Ordnung. Die neue Bibliothek benutzt für das gleiche Quad eine Duffy-Regel in der
  Differenzvariablen (4·10⁻¹² bei 8 Punkten). **Folge für den Bestand: die Diagonalblöcke der alten Rooftop-Einfachschicht (`integrate_singular_self_patch`) haben
  vermutlich denselben Fehler; nicht geprüft, in diesem Auftrag nicht Gegenstand.**
* **Fehler in der Theorie (korrigiert):** (i) Vorzeichen von Z (Z = −GᵀV_rotH); (ii) Q gehört auf das λ-Feld n×W_f, nicht auf W_f (M_rot, C_rot). Beides fällt nur bei harmonischen Spalten
  auf, an der Kugel nicht; am Torus waren die Residuen (S2)/(S3) vorher O(1), nachher Ordnung 1,9 bis 2.
* **T7:** Mit Baum-Kobaum und einer ψ-Konstante je Komponente bleibt **kein** Nullvektor (kleinster Singulärwert 2,6·10⁻⁴ σ_max); K allein mit Baum-Kobaum lässt am Ring genau einen
  harmonischen Nullvektor. Die Kohomologie-Eichung der Spulen ist auf kleinen Netzen durch (S3) ersetzt.
* **R3:** Die Quelle muss projiziert werden: ohne Projektion streut der Strom durch die Schnittschichten um 11 %, mit Projektion um 4·10⁻¹⁶.
* Schritt 5 (nur gemessen): siehe unten; kritisch ist die Konditionierung bei kleiner Frequenz und die Skalierung der Blöcke.

## Gemessen

### Quadratur (Schritt 2; `test_trace_quadrature`)

| Fall | Referenz | Fehler bei 4 / 6 / 8 Punkten |
|---|---|---|
| gleiches Quad, ∫∫1/R, Einheitsquadrat | scipy `dblquad` 2,973209598247 | 1,8·10⁻⁶ / 4,2·10⁻⁹ / 4,2·10⁻¹² (neue Regel); Bestand: 2,8·10⁻² / 1,3·10⁻² / 7,5·10⁻³ |
| gemeinsame Kante | 1,112128689849 | 1,5·10⁻⁶ / 2,9·10⁻⁸ / 4,2·10⁻¹⁰ |
| gemeinsame Ecke | 0,748952218549 | 6,3·10⁻⁷ / 1,2·10⁻¹⁰ / 5,2·10⁻¹³ |
| getrennt, Abstand 0,5 | 0,692962960817 | 2,6·10⁻⁹ / 2,6·10⁻¹³ / 8,7·10⁻¹⁵ |

Produktivwerte: `ng_sing = 8`, `ng_reg = 4`, Unterteilung m = ⌈h/d⌉ ≤ 4 für getrennte Paare. Quadraturfehler der Blöcke gegen die höhere Ordnung (12 / 6 / m ≤ 4, Faktor 2),
relative Frobenius-Norm, Würfel / Kugel 4×4×4 / Torus 12×6: V₀ 8·10⁻⁹ / 4·10⁻⁸ / 1·10⁻⁸, D̃ 6·10⁻⁷ / 2·10⁻⁶ / 3·10⁻⁷, V_rot 8·10⁻⁹ / 3·10⁻⁸ / 2·10⁻⁸,
C_rot 5·10⁻⁷ / 3·10⁻⁶ / 4·10⁻⁷, W ≤ 6·10⁻⁸, N ≤ 8·10⁻⁷, Q 2·10⁻⁷ (Torus). Die Blöcke sind damit um mindestens vier Größenordnungen genauer als die Diskretisierungsfehler.
V₀ und V_rot haben vor der Symmetrisierung eine Asymmetrie von 0,6 bis 1,8·10⁻⁸ (Maß der Quadraturkonsistenz); sie werden als (A + Aᵀ)/2 gespeichert.

### T1, T2, T3 (Schritt 2; `test_trace_ops`; Dipol im Innern, μ₀ = 1)

| Test | Größe | n = 3 / 6 / 12 | Ordnung |
|---|---|---|---|
| T1 Kugel, Außenfeld aus Spuren | max. rel. Fehler A / B | 9,8 / 2,6 / 0,68 % ; 11,3 / 3,1 / 0,79 % | 1,9 bis 2,0 |
| T2 Kugel, Zeile 1 (S1) / Zeile 2 (S2) | rel. Residuum | 11,4 / 3,9 / 1,1 % ; 11,6 / 3,5 / 0,97 % | 1,6 bis 1,8 |
| T2 Würfel, Zeile 1 / Zeile 2 | rel. Residuum | 17,3 / 7,3 / 2,1 % ; 16,1 / 2,4 / 0,63 % | 1,3 bis 2,7 |
| T3 Kugel, Rayleigh(W) gegen l(l+1)/(2l+1) | Fehler (l = 2) | 2,9 / 0,69 / 0,16 % | 2,1 |

Gegenproben (feinstes Netz): Vorzeichen eines Terms in (R-A)/(R-B): Abweichung 0,67 bis 0,69 (O(1)); Zeile 1 ohne V₀: 0,66 (Kugel), 0,58 (Würfel); Vorzeichen von ½ in N: 1,0; Vorzeichen
von D: 0,31/0,33; Zeile 2 mit Vorzeichen ½: 2,96/2,35; Faktor 2 auf W: 0,99/1,0; **W mit koplanarem Skalarprodukt statt des gedrehten: 0,53 (Kugel), 0,83 (Würfel), am Würfel
Faktor 131 schlechter als das Residuum 0,63 %** (die Kanten tragen). D und V auf Y_l: Ordnung 1,7 bis 2 (Fehler N 0,23 %, V 0,59 % bei l = 2, n = 12).
**Die Tabellenerwartung T1 (≤ 10⁻³ bei 6×6) war zu optimistisch;** gemessen 2,6·10⁻² bei n = 6.

### T5 (Schritt 3; `test_trace_torus`; Torus R = 3, r = 1, Kreisstrom I = 1, Schleifenformel gegen Biot–Savart-Summe 10⁻⁹)

| Größe | 12×6 | 24×12 | 48×24 |
|---|---|---|---|
| c_Meridian aus den exakten Spuren (Ampère) | 1 (3·10⁻¹³) | 1 | 1 |
| c_Meridian aus (S2),(S3) bei exaktem a, Fehler | 6,2·10⁻² | 1,7·10⁻² | 4,4·10⁻³ (Ordnung 1,96) |
| c_Längs (symmetrisches Netz) | 7·10⁻⁸ | 5·10⁻⁸ | 3·10⁻⁸ |
| c_Längs (Knoten um 0,15 h verschoben) | 2,5·10⁻⁴ | 3,9·10⁻⁵ | 3,2·10⁻⁵ |
| Residuum (S2) / (S3) | 13,6 / 5,3 % | 4,4 / 1,4 % | 1,2 / 0,37 % (Ordnung 1,9 / 2,0) |

Gegenproben: c_Meridian := 0: Residuum 54- bis 204-mal größer; falsches Vorzeichen von C in Q: 18- bis 250-mal; ohne harmonische Spalten ist ψ 11- bis 157-mal schlechter.
**Zwei Erwartungen der Theorie korrigiert:** c_Meridian ist für die *exakten* Spuren exakt I (Ampère, jede Meridianschleife), „Fehler ~h²“ gilt nur für den aus (S2),(S3) gelösten Wert;
c_Längs „→ 0 mit Verfeinerung“ ist am symmetrischen Netz ~10⁻⁸ durch Symmetrie, am verschobenen Netz ~3·10⁻⁵ und **nicht** fallend. Meridian-Orientierung: Zirkulation +I verlangt
den Umlauf entgegen dem Drehsinn der Poloidalkoordinate (reine Vorzeichenkonvention).

### T9 (Schritt 1; `test_trace_surface`, `test_trace_surface_team28`)

Würfel, Kugel, Quader, Torus, zwei Tori, Quader + Torus: R G = 0 exakt (max|RG| = 0), Euler V−E+F = 2K−2Σg, E = 2F, Rang R = F−K, Rang G = V−K, dim ker R − dim range G = 2Σg,
harmonische Felder ‖RH‖, ‖GᵀH‖ ≤ 10⁻⁹, orthonormal, Zirkulationsmatrix regulär (kleinster Singulärwert > 0,1). Gegenproben: umgedrehtes Quad wird erkannt; Zufallsvektor ist weder in
ker R noch ⟂ range G. **TEAM-28-Rand (`team28_hex.vtk`, 3696 Randquads): drei Komponenten mit g = 1, 1, 0; dim ℋ = 4; max|RH| = 5·10⁻¹², max|GᵀH| = 5·10⁻¹²; `quad_sign` stimmt mit
`mesh.boundary_edge_signs` überein, jede Flächenkante gehört zu genau einer FEM-Kante.**

### Gekoppeltes System (Schritt 4; `test_trace_system`, 42 s)

* **T8, Kugel mit Azimutalstrom** (n³ Hexaeder, Cube-to-Ball, Gauß 3×3×3 für die Quelle): n = 4 / 6 / 8 / 10: Fehler ψ 7,6 / 4,1 / 2,7 / 1,9 %; g 24 / 15 / 10 / 7,6 %; B innen 19 / 11 / 8,0 / 6,4 %.
  Reines FEM mit **exakten** Randdaten auf demselben Netz: B-Fehler 17 / 11 / 7,8 / 6,2 %; gekoppelt höchstens 2 % über dem reinen FEM-Fehler (n = 10), |B_gekoppelt − B_FEM|/|B| = 2,1 %.
  Der Fehler ist der FEM-Fehler (niedrigste Nédélec-Ordnung, O(h), und verzerrtes Netz); die Erwartung „~h²“ der Theorie ist für B und g nicht haltbar. (J) liefert dieselbe Genauigkeit wie (S)
  (n = 8: ψ 2,7 vs 2,7 %). Gegenprobe Vorzeichen von λ umgekehrt: ψ-Fehler 25 % statt 2,7 %, B 17 % statt 8 %. Ohne Quellprojektion praktisch kein Unterschied (Kugel: ‖Gᵀf‖/‖f‖ 3,8·10⁻³ → 10⁻¹⁶).
* **T6** (Ring 8×2×2, σ = 0, g = 1, + Platte 3×3×1, σ = 1): ‖A − Aᵀ‖/‖A‖ = 1·10⁻¹⁷ für (S), 4,8·10⁻² für (J); gestörter (3,1)-Block 1,2·10⁻³.
* **T7:** Nullraumdimension (Singulärwerte < 10⁻¹⁰ σ_max): (S) 0, (J) 0 (kleinster Singulärwert 2,6·10⁻⁴ bzw. 2,8·10⁻⁴). Gegenproben: ohne ψ-Konstanten 2 (= K; die Konstante ist nur bis auf den
  Quadraturfehler Nullvektor, Lücke 1,7·10⁻¹⁰ | 5·10⁻⁴ zur Schwelle 10⁻⁵); ohne Baum-Kobaum 71 = Eichkanten (70 bei 10⁻¹⁰, ein Gradient mit Randanteil nur bis auf Quadraturniveau);
  ohne BEM-Zeilen (nur K, Baum-Kobaum) genau 1 (harmonisch, die alte Kohomologie-Eichung).
* **T4** (Ring, ψ_h = z·exp(−r²/4)): relative Änderung der Zeilen (S1) 1·10⁻¹⁷, (S2) 9·10⁻¹⁸ (Flussblöcke, Schwelle 10⁻¹²), (S3) 7,9·10⁻⁶ (Q auf Quadraturniveau; Schwelle 10⁻⁵, abgeleitet aus
  Q7: 2·10⁻⁷ mal 50, **das ist keine strenge Schranke: gemessen liegt (S3) um das 40-fache über Q7, Grund nicht geklärt**). Gegenprobe zufälliger Nicht-Gradient: 7,5 bis 9,1·10⁻² in allen Zeilen.
* **R3** (Ring mit ungleichmäßiger Winkelteilung, azimutale Quelle): ‖Gᵀf‖/‖f‖ unprojiziert 0,29, projiziert 2·10⁻¹⁶. Strom durch die Schnittschichten: Streuung über die 8 Schichten 1,1·10⁻¹ ohne,
  3,5·10⁻¹⁶ mit Projektion; c_Meridian aus (S1) mit Schnittschicht i: Streuung 1,1·10⁻¹ ohne, 8·10⁻⁵ mit Projektion. Gelöstes c_Meridian 0,972 (ohne) / 0,945 (mit Projektion), erwartet 1; c_Längs ≤ 7·10⁻⁵.
  Bei gleichmäßiger Winkelteilung ist die Quelle schon exakt divergenzfrei (Randbeiträge heben sich knotenweise auf); der Effekt tritt erst bei ungleichmäßiger Teilung auf.
  **Beobachtung am Bestand:** `project_out_gradient` ohne Maske auf einem Netz aus zwei getrennten Körpern vergrößerte bei bereits divergenzfreiem f (‖Gᵀf‖/‖f‖ = 2·10⁻¹⁶) die Norm von 1,0 auf 4,2;
  bei ‖Gᵀf‖/‖f‖ = 0,29 nicht. Mit Maske für den unbeteiligten Körper ist das Ergebnis exakt (hier verwendet). Für Phase 2 (TEAM 28: drei getrennte Körper) beachten.

### Konditionierung (Schritt 5, nur gemessen; `trace_conditioning`)

Dicht, Double, Eigenwerte bzw. Singulärwerte exakt (SVD). K nach Baum-Kobaum; W ohne Konstanten; S = Schurkomplement auf (a_Γ, ψ, c) nach Elimination der inneren Kanten.

**Kugel 6³ Hexaeder, μ₀ = 1, σ = 0, μ_r im ganzen Körper:**

| μ_r | λmax(K) | λmin⁺(K) | Kond K | Kond W | Kond V₀ | Kond S | Kond S, diagonalskaliert |
|---|---|---|---|---|---|---|---|
| 1 | 4,1·10¹ | 3,9·10⁻² | 1,05·10³ | 4,6 | 57 | 7,6·10⁴ | 9,6·10² |
| 10 | 4,1 | 3,9·10⁻³ | 1,05·10³ | 4,6 | 57 | 1,6·10⁴ | 7,8·10² |
| 100 | 4,1·10⁻¹ | 3,9·10⁻⁴ | 1,05·10³ | 4,6 | 57 | 1,1·10⁴ | 5,2·10³ |
| 1000 | 4,1·10⁻² | 3,9·10⁻⁵ | 1,05·10³ | 4,6 | 57 | 5,0·10⁴ | 5,0·10⁴ |

(W: λmax 0,161, λmin⁺ 0,035; V₀: λmax 17,6, λmin 0,31; unabhängig von μ_r.)

**TEAM 28, gröbstes Netz (`near/hex_s0.5.vtk`: 696 Hexaeder, 3086 FEM-Kanten, 936 Randquads, 3 Komponenten, nH = 4; Millimeter → Meter, μ₀ = 4π·10⁻⁷, σ_Platte = 3,4·10⁷ S/m, Spulen σ = 0),
μ_r = 1; n_innen = 688, n_Γ = 2619:**

| f / Hz | Kond A_ii (Schätzung) | Kond S | Kond S, diagonalskaliert |
|---|---|---|---|
| 0,01 | 9,7·10⁷ | numerisch singulär (Singulärwert 0, Einträge über 15 Dekaden) | 1,5·10⁶ |
| 0,1 | 3,0·10⁷ | dito | 1,5·10⁵ |
| 1 | 9,7·10⁶ | dito | 1,5·10⁴ |
| 10 | 3,1·10⁶ | dito | 4,4·10³ |
| 50 | 1,4·10⁶ | dito | 4,2·10³ |

Blöcke (Meter-Einheiten): λmax(K) 5,6·10⁹, λmin⁺ 6,3·10⁵, Kond 9·10³ bei 508 Nullwerten (Gradienten in der Platte ohne jωσ-Term; im Block fixiert sie erst jωσ);
W: λmax 1,4·10⁻², λmin⁺ 9,3·10⁻⁵, Kond 152; V₀: λmax 1,3·10³, λmin 4,3, Kond 303.

**TEAM 28, gröbstes Netz, μ_r = 1000 in beiden Spulen (Testkörper), sonst wie oben:**

| f / Hz | Kond A_ii (Schätzung) | Kond S | Kond S, diagonalskaliert |
|---|---|---|---|
| 0,01 | 9,7·10⁷ | numerisch singulär | 1,5·10⁶ |
| 0,1 | 3,0·10⁷ | dito | 2,0·10⁵ |
| 1 | 1,4·10⁷ | dito | 2,0·10⁵ |
| 10 | 1,4·10⁷ | dito | 1,9·10⁵ |
| 50 | 1,4·10⁷ | dito | 1,8·10⁵ |

K (Baum-Kobaum) mit μ_r = 1000: λmin⁺ 2,0·10³ (statt 6,3·10⁵), Kond 2,9·10⁶ (statt 9·10³); W und V₀ unverändert. Bei 50 Hz steigt die diagonalskalierte Kond S von 4,2·10³ (μ_r = 1) auf 1,8·10⁵
(Faktor 42), bei 0,01 Hz bleibt sie 1,5·10⁶ (dort bestimmt die Leiterfrequenz).

## Interpretation (nicht geprüft)

* Die Kopplung selbst (Außenraum aus λ, g) ist nach den Tests T1 bis T5 und T8 konsistent; die Fehler in T8 sind FEM-Fehler. Ob (S) auf feinen Netzen **stabil** ist (inf-sup, Hiptmair 2002 bezieht sich auf ein E-basiertes
  Modell), ist mit kleinen Netzen nicht entschieden; T7 und die gute Lösbarkeit sprechen dafür, beweisen es nicht.
* Die Konditionierung bei kleiner Frequenz (Zeile 0,01 Hz) folgt wie erwartet 1/ω im Leiter (diagonalskaliert 1,5·10⁶ → 4,2·10³; Faktor ~3,6·10² bei einem Faktor 5·10³ in ω, gesättigt bei ≥ 10 Hz): die
  Randblöcke helfen dort nicht (R4′). Das unskalierte S ist wegen der Größenordnungen von K (10⁹) gegen die Randblöcke (10⁻²) numerisch singulär; eine Skalierung ist für jeden Löser Pflicht.
* Bei hohem μ_r wird der Randterm relativ größer (R7): in der Kugel steigt Kond S von 9,6·10² (μ_r = 1) auf 5·10⁴ (μ_r = 1000), die Diagonalskalierung hilft dort nicht mehr.
* Die Ursache des 40-fachen Abstands von (S3) zu Q7 in T4 ist nicht untersucht (vermutlich die Gradientenfelder mit Randanteil am Ringrand bei grobem Netz).

## Getroffene Wahlen und Abweichungen vom Auftrag

1. **Eigene Oberflächenstruktur statt `BemCouplingMap`:** `BemCouplingMap` ist auf den ACA-Baum zugeschnitten; `TraceSurface` bildet G, R und die Quad-Kanten direkt. Konsistenz mit `boundary_edge_signs` und
   `boundary_edges_to_fem_edges` ist am TEAM-28-Rand und im Kopplungsaufbau getestet (Wiederverwendung der Daten, nicht des Codes).
2. **Gleiches Quad: eigene Duffy-Regel** statt `get_coincident_coords` (siehe oben); gemeinsame Kante/Ecke aus dem Bestand (Abbildungen), mit frei wählbarer Gauß-Ordnung.
3. **FEM-Block K mit den Produktionsbasisfunktionen und -Piola-Helfern** (`NedelecHexahedron.cuh`, `NedelecDispatch.cuh`), eigene dichte Assemblierung (der Bestand hat keine dichte Host-Assemblierung; die Kopie in
   `test_cohomology_gauge.cu` benutzt eine andere J-Konvention). Meine erste Fassung hatte J transponiert (Fehler 0,1 bis 0,4 in T8) und wurde an `fem.cu` gespiegelt.
4. **Quelle 3×3×3 Gauß** (Produktion: elementweise); Gradientenprojektion über `project_out_gradient` des Bestands.
5. **T8 mit Hexaeder-Kugel:** ein Cube-to-Ball-Netz ist im Test erzeugt (kein Verzug von der Aufgabe: ein Hexaeder-Kugelnetz war nötig). Das verzerrte Netz begrenzt die Genauigkeit.
6. **T7 an einem Rechteckring** (σ = 0, g = 1) mit Platte (σ = 1), nicht am Torus mit rundem Querschnitt; Nullraumschwellen 10⁻¹⁰ und 10⁻⁵ mit Lückenangabe (Begründung oben).
7. **Reelle Rechnung in den Tests** (σ-Term als reeller Faktor); komplexe Arithmetik nur im Messtreiber.
8. **Theorie angepasst:** Z-Vorzeichen, Q auf λ-Feld, T1/T5/T8-Erwartungen, Schritt-0-Punkte 1 bis 4 (eigener Commit).
9. **Schritt 5 am gröbsten TEAM-28-Netz (936 Randquads) statt am Standardnetz (3696):** dicht ist das größere Netz machbar (1 TB RAM), die Laufzeit der SVD-Messung wäre aber eine Größenordnung höher; nicht gemessen.
10. **Laufzeiten:** `test_trace_ops` 29 s, `test_trace_quadrature` 24 s, `test_trace_torus` 24 s, `test_trace_system` 42 s auf 32 Kernen (28 CPU-Minuten im Spitzenlauf); auf wenigen Kernen > 120 s, deshalb `TIMEOUT 600/900`.

## Offene Punkte und Risiken für Phase 2

1. Stabilität/inf-sup von (S) auf TEAM-28-Größe und feinen Netzen; nur kleine Netze getestet.
2. Skalierung der Blöcke (Größenordnungen 10¹⁵ im unskalierten S) und Vorkonditionierer; Konditionierung bei 0,01 Hz (R4′) und hohem μ_r (R7).
3. Alte Diagonalblöcke der Rooftop-Einfachschicht mit dem fehlerhaften `get_coincident_coords` prüfen (Bestand, nicht angefasst).
4. `project_out_gradient` bei getrennten Körpern nur mit Maske verwenden.
5. Bewegte Körper, Transient, nichtlineares Eisen, ACA/GPU: nicht Teil.
6. Die T4-Schwelle für Q (10⁻⁵) ist empirisch; Ursache des Faktors 40 zu Q7 offen.

## Befehle

```
cd build
cmake .. && make host_tests trace_conditioning
./test_trace_surface ; ./test_trace_surface_team28 ../data/team28/team28_hex.vtk
./test_trace_quadrature ; ./test_trace_ops ; ./test_trace_torus ; ./test_trace_system
./trace_conditioning ball 6
./trace_conditioning team28 near/hex_s0.5.vtk [--mur-coil=1000]
ctest -L host ; ctest      # mit LD_LIBRARY_PATH auf die CUDA-Bibliotheken
```

## Tests

`ctest -L host`: vorher 43/43, nachher 49/49 (neu: `test_trace_surface`, `test_trace_surface_team28`, `test_trace_quadrature`, `test_trace_ops`, `test_trace_torus`,
`test_trace_system`). Volles `ctest`: vorher 76/76, nachher 82/82; alle bestanden, mit `LD_LIBRARY_PATH` auf die CUDA-Bibliotheken. Laufzeiten im vollen Lauf (Konkurrenz mit anderen Tests):
`test_trace_quadrature` 34 s, `test_trace_ops` 40 s, `test_trace_torus` 67 s, `test_trace_system` 53 s (je unter 120 s). `trace_conditioning` ist kein ctest-Test.
