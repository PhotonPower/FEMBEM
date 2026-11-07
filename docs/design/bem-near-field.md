# Nahfeldquadratur und Konvergenztrend der BEM-Kopplung

Stand: 04.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), Linux (nicht RTX 3090 /
Windows). Basis: `main` 0288ea6 (nach den vier Korrekturen aus `bem-coupling-fixes.md`), Zweig
`bem-near-field`. Netze aus `tools/gmsh/team28_hex.py`, Platte bei 11,3 mm (`--plate-z=11.3`), 50 Hz
(BEM-FEM mit `--harmonic-inner-direct`, Referenzpfad `--reference --harmonic=50`), Zeilenmessung
`--bem-row-check` bei 2 Hz. Logs, Netze und Dumps in `build/near/` (nicht im Repo).

## Frage und Ergebnis in Kürze

Nach den Korrekturen liegt F_z (50 Hz, 11,3 mm, Skalierung 1,0) bei 0,545 N gegen 0,9665 N im
Referenzpfad. Zeile 2 steht an der Platte bei 19,6 % (Skalierung 0,5: 45 %). Die Hypothese des Auftrags:
das Nahfeld disjunkter Quadpaare (Ober- und Unterseite der Platte, 3 mm Abstand) wird von der
2×2-Gauss-Regel `integrate_regular_quad_pair(_dlp)` nicht genau genug integriert. Gemessen wurde in
drei Teilen; die Hypothese wird **nicht bestätigt**.

**Antworten (Messung)**

1. **Ist die Nahfeldquadratur die Ursache des Plattenfehlers? Nein.** Die Quadratur ist bei kleinem
   Abstand tatsächlich ungenau (B1: bei d/s = 0,5 sind es 6,1 % für G und 35,5 % für D; unter 1 % erst ab
   d/s = 2 bzw. 4). Eine abstandsabhängige Unterteilung (Prototyp `--bem-near-subdiv`: auf Paarebene
   G unter 1e-3 ab d/s = 0,1, D unter 3e-3 ab d/s = 0,2) ändert das Plattenresiduum von Zeile 2 aber nicht: Skalierung 1,0 19,59 % →
   19,58 %, Skalierung 0,5 45,41 % → 47,97 %. Auf den ebenen Deckel- und Bodenquads der Platte, wo der
   Fehler sitzt, ändert sich nichts (19,62 % → 19,63 % bzw. 45,58 % → 48,16 %).
2. **Wie viel vom F_z-Defizit erklärt sie? Nichts, soweit gemessen.** Mit dem Prototyp (Skalierung 0,5)
   geht F_z von 0,0775 N auf 0,0498 N, also von der Referenz (0,853 N) weg; Skalierung 1,0: 0,545 N → 0.519 N (Referenz 0,9665 N). Die Unterteilung vergrößert das Defizit leicht, sie verringert es nicht.
3. **Konvergenztrend (A):** F_z / F_ref(gleiches Netz) ist über der Skalierung **nicht monoton** und
   zeigt keine Konvergenz (Tabelle A); eine Ordnung lässt sich nicht angeben. Das Plattenresiduum von
   Zeile 2 fällt von Skalierung 0,5 über 0,75 auf 1,0 (45 % → 28 % → 20 %), steigt aber bei 1,25 wieder
   (34,4 %): es ist keine Funktion von Panelkante/Plattendicke allein.
4. **Der Dünnquader-Test (B2)** zeigt zwar das vorhergesagte Wachstum des Deckel-/Bodenresiduums mit
   h/t (2,6 % → 3,9 % → 7,0 % für h/t = 0,83 / 1,67 / 3,33), aber auf kleinerem Niveau als beim
   Würfel gleicher Panelkante (9 %), und die Unterteilung ändert es kaum (7,0 % → 6,1 %, 3,9 % →
   3,8 %, 2,6 % → 2,6 %). Das Wachstum mit h/t ist also bei diesen Netzen vom Nahfeldfehler
   unabhängig; *Interpretation (nicht geprüft):* es ist der gewöhnliche Diskretisierungsfehler, der mit
   der Panelkante h wächst, nicht mit h/t.

**Nicht beantwortet:** woher der Plattenfehler in Zeile 2 und das F_z-Defizit stammen. Kandidaten
sind (Interpretation, nicht gemessen) die Darstellung nur der Tangentialspur in der D-Zeile, die
Behandlung der Kanten und Seitenquads der dünnen Platte und die K1/K2-Frage; dieser Auftrag hat sie
nicht untersucht.

## A: Konvergenztrend des BEM-FEM-Pfads

Alle Läufe: 50 Hz, 11,3 mm, `--harmonic-inner-direct`, Operator wie `main` 0288ea6. Skalierung 1,0 stammt
aus den Läufen von `bem-coupling-fixes.md` (derselbe Operatorstand; `main` unterscheidet sich seitdem
nur durch einen CI-Testfix). Skalierung 1,5 wurde **nicht gerechnet**: dichte Algebra
nd = 89 536, 59,7 GiB gegen 48 GiB der RTX 6000 Ada (Skalierung 1,25: nd = 58 144, 27,0 GiB, passt).

### F_z und Iterationen (50 Hz, 11,3 mm)

| Skalierung | F_z Phasen 0/45/90/135 [N] | F_z Mittel [N] | F_z Referenzpfad [N] | F_z / F_ref | aeussere / innere Iterationen |
|---|---|---:|---:|---:|---|
| 0.5 | 0.179 / -2.08 / -0.0242 / 2.24 | 0.0775 | 0.85333 | 0.091 | 18 (8.62e-07) / 36 |
| 0.75 | 0.159 / -2.28 / -0.0455 / 2.4 | 0.0569 | 0.93495 | 0.061 | 18 (9.4e-07) / 36 |
| 1.0 | 1.14 / -2.58 / -0.0459 / 3.67 | 0.5448 | 0.96649 | 0.564 | 19 (8.49e-07) / 38 |
| 1.25 | 0.997 / -3.33 / -0.0278 / 4.3 | 0.4850 | 0.98096 | 0.494 | 20 (7.47e-07) / 40 |

### Panelkanten aus dem Netz (Platte, Deckel/Boden)

| Skalierung | Plattendicke [mm] | Deckelkante mittel [mm] | sqrt(Flaeche) [mm] | Kante/Dicke |
|---|---:|---:|---:|---:|
| 0.5 | 3.000 | 9.170 | 9.376 | 3.057 |
| 0.75 | 3.000 | 6.005 | 6.066 | 2.002 |
| 1.0 | 3.000 | 4.590 | 4.657 | 1.530 |
| 1.25 | 3.000 | 3.717 | 3.780 | 1.239 |

### Zeilenresiduen bei 2 Hz (V0 = Operator; Zeile 1 mit +K2, Zeile 2 mit +K1)

| Skalierung | Z1 Platte | Z1 Innenspule | Z1 Aussenspule | Z1 gesamt | Z2 Platte (alle) | Z2 Platte (eben) | Z2 Innenspule | Z2 Aussenspule | Z2 gesamt | Kante/Dicke |
|---|---|---|---|---|---|---|---|---|---|---:|
| 0.5 | 2.31 % | 16.89 % | 15.15 % | 16.19 % | 45.41 % | 45.58 % | 9.32 % | 4.42 % | 7.66 % | 3.06 |
| 0.75 | 0.93 % | 10.95 % | 9.40 % | 10.38 % | 27.98 % | 28.01 % | 10.02 % | 3.94 % | 6.70 % | 2.00 |
| 1.0 | 0.60 % | 7.77 % | 6.94 % | 7.44 % | 19.59 % | 19.62 % | 7.92 % | 2.26 % | 4.89 % | 1.53 |
| 1.25 | 0.42 % | 5.88 % | 5.32 % | 5.65 % | 34.37 % | 34.49 % | 15.59 % | 4.96 % | 9.67 % | 1.24 |

### Felder: Projektion auf Biot-Savart (+Ringmodell) und Vergleich gegen den Referenzpfad (Zeile "alle")

Skalierung 0.5:

| Koerper | Phase | proj Biot-Savart | L2 Biot-Savart | B/Bref Biot-Savart | proj Referenzpfad | L2 Referenzpfad | B/Bref Referenzpfad | B_phi/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Platte | Re | 1.0029 | 0.3085 | 1.0492 | 1.0175 | 0.3341 | 1.0708 | 0.1274 |
| Platte | Im | 0.9304 | 0.4335 | 1.0241 | 1.1584 | 0.4631 | 1.2374 | 0.2654 |
| Innenspule | Re | 0.5401 | 0.5999 | 0.6634 | 0.6067 | 0.5460 | 0.7153 | 0.0914 |
| Innenspule | Im | 1.2983 | 0.5767 | 1.3889 | 1.5355 | 0.7895 | 1.6414 | 0.1498 |
| Aussenspule | Re | 0.4720 | 0.7134 | 0.6730 | 0.5794 | 0.6530 | 0.7650 | 0.1340 |
| Aussenspule | Im | 3.1959 | 4.1475 | 4.7533 | 3.9499 | 5.7504 | 6.3219 | 0.5100 |

Skalierung 0.75:

| Koerper | Phase | proj Biot-Savart | L2 Biot-Savart | B/Bref Biot-Savart | proj Referenzpfad | L2 Referenzpfad | B/Bref Referenzpfad | B_phi/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Platte | Re | 1.0338 | 0.3103 | 1.0788 | 1.0568 | 0.3184 | 1.1023 | 0.1130 |
| Platte | Im | 0.4924 | 0.8921 | 0.8835 | 1.2664 | 0.4577 | 1.3200 | 0.1267 |
| Innenspule | Re | 0.6544 | 0.4855 | 0.7379 | 0.7705 | 0.4239 | 0.8489 | 0.0513 |
| Innenspule | Im | 1.4496 | 0.7374 | 1.5630 | 1.5516 | 0.8338 | 1.6728 | 0.2242 |
| Aussenspule | Re | 0.3447 | 0.8064 | 0.5827 | 0.7188 | 0.5275 | 0.8461 | 0.0848 |
| Aussenspule | Im | 3.6557 | 4.6601 | 5.2941 | 3.9911 | 5.2975 | 5.9199 | 0.5759 |

Skalierung 1.0:

| Koerper | Phase | proj Biot-Savart | L2 Biot-Savart | B/Bref Biot-Savart | proj Referenzpfad | L2 Referenzpfad | B/Bref Referenzpfad | B_phi/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Platte | Re | 1.0833 | 0.3363 | 1.1313 | 1.0891 | 0.3410 | 1.1377 | 0.0960 |
| Platte | Im | 1.2892 | 0.4930 | 1.3496 | 1.3072 | 0.5161 | 1.3714 | 0.1683 |
| Innenspule | Re | 0.8706 | 0.4072 | 0.9524 | 0.8482 | 0.3958 | 0.9236 | 0.0563 |
| Innenspule | Im | 1.4810 | 0.6446 | 1.5420 | 1.5397 | 0.6995 | 1.6027 | 0.1128 |
| Aussenspule | Re | 0.7879 | 0.4862 | 0.9013 | 0.7876 | 0.4870 | 0.9013 | 0.0990 |
| Aussenspule | Im | 3.7302 | 3.9176 | 4.6699 | 3.9280 | 4.2233 | 4.9691 | 0.3986 |

Skalierung 1.25:

| Koerper | Phase | proj Biot-Savart | L2 Biot-Savart | B/Bref Biot-Savart | proj Referenzpfad | L2 Referenzpfad | B/Bref Referenzpfad | B_phi/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Platte | Re | 1.1262 | 0.7287 | 1.3354 | 1.1301 | 0.7321 | 1.3402 | 0.2535 |
| Platte | Im | 1.3434 | 1.0352 | 1.6608 | 1.3479 | 1.0373 | 1.6649 | 0.3215 |
| Innenspule | Re | 0.8068 | 0.4903 | 0.9242 | 0.9097 | 0.4861 | 1.0275 | 0.1544 |
| Innenspule | Im | 1.6260 | 3.0448 | 3.3945 | 1.6647 | 3.1271 | 3.4797 | 0.4152 |
| Aussenspule | Re | 0.7046 | 0.5905 | 0.8706 | 0.8247 | 0.4958 | 0.9462 | 0.1684 |
| Aussenspule | Im | 5.1522 | 19.2560 | 19.4961 | 5.3269 | 20.0328 | 20.2724 | 0.8192 |



**Messung:** F_z / F_ref liegt bei 0,091 (0,5), 0,061 (0,75), 0,564 (1,0) und 0,494 (1,25). Es gibt keinen
monotonen Trend, deshalb keine Ordnung (das Auftragsmuster verlangt mindestens drei Punkte; die
vorhandenen vier laufen nicht gegen einen Grenzwert). Die Iterationszahlen sind mit 18 bis 19
äußeren Iterationen praktisch konstant. Die Projektionen der Plattenfelder (Realteil) liegen bei allen
Netzen nahe 1 (1,00 bis 1,08), die Imaginärteile der Spulen bleiben bei allen Netzen zu groß.

## B1: Paartest ohne FEM

`tests/test_bem_near_field.cu --table`. Zwei gegenüberliegende parallele Quads (Kantenlänge s, Abstand
d, p unten mit Normale −z, q oben mit +z; beide Rollen gerechnet, Maximum), quadratisch, Rechteck 1:2,
je ausgerichtet und seitlich um s/2 versetzt. Referenz: gleichmäßige Unterteilung beider
Parametergebiete (`integrate_adjacent_quad_pair(_dlp)`, 24 × 24 gegen 48 × 48 je Quad;
Konvergenz der Referenz in der Tabelle; höchstens 4,1e-3 bei d/s = 0,05 (D, Rechteckfall), für d/s ≥ 0,1
höchstens 6,9e-5).

| d/s | Fehler G | Fehler D | Fehler G (Unterteilung) | Fehler D (Unterteilung) |
|---:|---:|---:|---:|---:|
| 0.05 | 2.18e+00 | 3.77e+01 | 7.92e-03 | 2.80e-01 |
| 0.10 | 9.28e-01 | 1.02e+01 | 9.73e-04 | 2.86e-02 |
| 0.20 | 3.40e-01 | 2.66e+00 | 1.12e-04 | 2.19e-03 |
| 0.50 | 6.08e-02 | 3.55e-01 | 5.17e-05 | 5.59e-04 |
| 1.00 | 1.48e-02 | 6.17e-02 | 1.54e-04 | 5.72e-04 |
| 2.00 | 3.14e-03 | 1.32e-02 | 4.51e-04 | 2.11e-03 |
| 4.00 | 3.57e-04 | 1.63e-03 | 3.57e-04 | 1.63e-03 |

Einzelwerte je Fall (Fehler der Produktionsregel; Konvergenz der Referenz; Level und Fehler der
Unterteilung, siehe C):

| Fall | d/s | Fehler G | Fehler D | Konvergenz Ref. G | Konvergenz Ref. D | Level | Fehler G mit Unterteilung | Fehler D mit Unterteilung |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quadratisch, ausgerichtet | 0.05 | 1.36e+00 | 1.94e+01 | 5.0e-07 | 1.0e-04 | 8 | 1.38e-03 | 5.68e-02 |
| quadratisch, ausgerichtet | 0.10 | 5.18e-01 | 5.13e+00 | 4.5e-07 | 5.7e-06 | 8 | 3.17e-05 | 1.83e-03 |
| quadratisch, ausgerichtet | 0.20 | 1.45e-01 | 1.19e+00 | 1.1e-07 | 8.4e-07 | 8 | 9.98e-06 | 6.82e-05 |
| quadratisch, ausgerichtet | 0.50 | 1.33e-02 | 7.85e-02 | 2.1e-08 | 8.4e-08 | 4 | 3.11e-05 | 1.27e-04 |
| quadratisch, ausgerichtet | 1.00 | 3.45e-03 | 1.39e-02 | 6.4e-09 | 2.3e-08 | 2 | 1.54e-04 | 5.72e-04 |
| quadratisch, ausgerichtet | 2.00 | 4.51e-04 | 2.11e-03 | 1.1e-09 | 4.6e-09 | 0 | 4.51e-04 | 2.11e-03 |
| quadratisch, ausgerichtet | 4.00 | 3.58e-05 | 1.76e-04 | 9.6e-11 | 4.6e-10 | 0 | 3.58e-05 | 1.76e-04 |
| quadratisch, um s/2 versetzt | 0.05 | 3.50e-01 | 2.21e+00 | 7.1e-07 | 1.2e-04 | 8 | 1.01e-03 | 5.24e-02 |
| quadratisch, um s/2 versetzt | 0.10 | 2.16e-01 | 1.90e+00 | 1.7e-07 | 2.7e-06 | 8 | 3.88e-05 | 1.90e-03 |
| quadratisch, um s/2 versetzt | 0.20 | 7.61e-02 | 6.87e-01 | 4.0e-08 | 3.8e-07 | 8 | 3.38e-06 | 2.41e-05 |
| quadratisch, um s/2 versetzt | 0.50 | 9.99e-03 | 5.76e-02 | 1.0e-08 | 3.4e-08 | 4 | 1.44e-05 | 5.02e-05 |
| quadratisch, um s/2 versetzt | 1.00 | 2.39e-03 | 9.90e-03 | 4.6e-09 | 1.5e-08 | 2 | 1.07e-04 | 3.49e-04 |
| quadratisch, um s/2 versetzt | 2.00 | 4.23e-04 | 1.82e-03 | 1.0e-09 | 4.2e-09 | 0 | 4.23e-04 | 1.82e-03 |
| quadratisch, um s/2 versetzt | 4.00 | 4.20e-05 | 2.00e-04 | 1.1e-10 | 5.3e-10 | 0 | 4.20e-05 | 2.00e-04 |
| Rechteck 1:2, ausgerichtet | 0.05 | 2.18e+00 | 3.77e+01 | 6.2e-05 | 4.1e-03 | 8 | 7.92e-03 | 2.80e-01 |
| Rechteck 1:2, ausgerichtet | 0.10 | 9.28e-01 | 1.02e+01 | 7.7e-07 | 5.3e-05 | 8 | 9.73e-04 | 2.86e-02 |
| Rechteck 1:2, ausgerichtet | 0.20 | 3.40e-01 | 2.66e+00 | 4.0e-07 | 4.0e-06 | 8 | 2.25e-05 | 9.74e-04 |
| Rechteck 1:2, ausgerichtet | 0.50 | 6.08e-02 | 3.55e-01 | 5.7e-08 | 3.3e-07 | 6 | 1.63e-05 | 8.60e-05 |
| Rechteck 1:2, ausgerichtet | 1.00 | 1.48e-02 | 6.17e-02 | 1.6e-08 | 5.7e-08 | 3 | 7.39e-05 | 2.67e-04 |
| Rechteck 1:2, ausgerichtet | 2.00 | 3.14e-03 | 1.32e-02 | 4.9e-09 | 1.7e-08 | 2 | 1.17e-04 | 4.13e-04 |
| Rechteck 1:2, ausgerichtet | 4.00 | 3.44e-04 | 1.63e-03 | 7.7e-10 | 3.4e-09 | 0 | 3.44e-04 | 1.63e-03 |
| Rechteck 1:2, um s/2 in x und y versetzt | 0.05 | 1.88e-01 | 9.06e-01 | 3.7e-05 | 3.7e-03 | 6 | 3.96e-03 | 2.31e-01 |
| Rechteck 1:2, um s/2 in x und y versetzt | 0.10 | 1.59e-01 | 8.07e-01 | 8.1e-07 | 6.9e-05 | 6 | 1.11e-04 | 2.25e-02 |
| Rechteck 1:2, um s/2 in x und y versetzt | 0.20 | 1.12e-01 | 6.12e-01 | 5.1e-08 | 2.2e-07 | 6 | 1.12e-04 | 2.19e-03 |
| Rechteck 1:2, um s/2 in x und y versetzt | 0.50 | 3.68e-02 | 2.22e-01 | 2.3e-08 | 1.1e-07 | 4 | 5.17e-05 | 5.59e-04 |
| Rechteck 1:2, um s/2 in x und y versetzt | 1.00 | 8.34e-03 | 4.27e-02 | 1.1e-08 | 3.2e-08 | 3 | 5.23e-05 | 1.66e-04 |
| Rechteck 1:2, um s/2 in x und y versetzt | 2.00 | 2.23e-03 | 8.11e-03 | 4.3e-09 | 1.4e-08 | 2 | 1.03e-04 | 3.46e-04 |
| Rechteck 1:2, um s/2 in x und y versetzt | 4.00 | 3.57e-04 | 1.58e-03 | 8.3e-10 | 3.4e-09 | 0 | 3.57e-04 | 1.58e-03 |

**Ergebnis:** Die 2×2-Gauss-Regel hat bei d/s < 1 Fehler über 5 % (d/s = 0,5: G 6,1 %, D 35,5 %). Der
Fehler fällt unter 1 % ab **d/s = 2 (G)** bzw. **d/s = 4 (D)**. D ist durchweg etwa eine Größenordnung
schlechter als G. In den Netzen dieses Berichts ist die Plattendicke 3 mm, die Deckelkante 9,2 / 6,0 /
4,6 / 3,7 mm, also d/s = 0,33 / 0,50 / 0,65 / 0,81 für die Paare Oberseite–Unterseite genau
gegenüber (Tabelle der Panelkanten oben); das liegt im Bereich der großen Fehler.

## B2: Dünner Quader als Calderón-Test

Wie `test_bem_calderon.cu` (Punktquelle innen, Außenform, k = +∂u/∂n), aber ein Quader 60 × 60 × 3 mm
(eine Schicht in z) und zum Vergleich ein Würfel 60 mm gleicher Panelkante. Residuum
||(+½T − D)a + G k|| / ||(+½T − D)a||, gesamt und getrennt nach Deckel/Boden (|n_z| > 0,99) und Seiten.

| Koerper | Panelkante h [mm] | Deckel h/t | Quads | Residuum gesamt | Deckel/Boden | Seiten |
|---|---:|---:|---:|---:|---:|---:|
| Quader 60x60x3 | 10.0 | 3.33 | 96 | 9.96 % | 7.02 % | 71.06 % |
| Wuerfel 60 | 10.0 | - | 216 | 9.27 % | 9.27 % | 9.26 % |
| Quader 60x60x3, --bem-near-subdiv=8 | 10.0 | 3.33 | 96 | 9.76 % | 6.11 % | 71.06 % |
| Wuerfel 60, --bem-near-subdiv=8 | 10.0 | - | 216 | 9.27 % | 9.27 % | 9.27 % |
| Quader 60x60x3 | 5.0 | 1.67 | 336 | 6.47 % | 3.89 % | 51.35 % |
| Wuerfel 60 | 5.0 | - | 864 | 9.08 % | 9.09 % | 9.08 % |
| Quader 60x60x3, --bem-near-subdiv=8 | 5.0 | 1.67 | 336 | 6.44 % | 3.80 % | 51.34 % |
| Wuerfel 60, --bem-near-subdiv=8 | 5.0 | - | 864 | 9.09 % | 9.09 % | 9.09 % |
| Quader 60x60x3 | 2.5 | 0.83 | 1248 | 4.48 % | 2.63 % | 30.22 % |
| Wuerfel 60 | 2.5 | - | 3456 | 9.02 % | 9.02 % | 9.02 % |
| Quader 60x60x3, --bem-near-subdiv=8 | 2.5 | 0.83 | 1248 | 4.48 % | 2.63 % | 30.20 % |
| Wuerfel 60, --bem-near-subdiv=8 | 2.5 | - | 3456 | 9.02 % | 9.02 % | 9.02 % |

**Messung:** Das Residuum der Deckel und Böden wächst mit h/t (2,63 % bei 0,83, 3,89 % bei 1,67, 7,02 % bei
3,33), die Hypothese sagt das voraus. Es liegt aber unter dem des Würfels gleicher Panelkante (9,0 bis
9,3 %, praktisch unabhängig von h), und die abstandsabhängige Unterteilung ändert es nur wenig
(Deckel 7,02 → 6,11 %, 3,89 → 3,80 %, 2,63 → 2,63 %). Die Seitenquads des Quaders (3 mm hoch) sind mit
30 bis 71 % schlecht und werden von der Unterteilung nicht verbessert; ihr Fehler fällt mit der
Panelkante.

## C: Prototyp `--bem-near-subdiv[=N]`

Die Bedingungen des Auftrags waren erfüllt (B1 zeigt bei d/s < 1 Fehler über 5 %, B2 zeigt das Wachstum
des Deckelresiduums mit h/t), also wurde der Prototyp gebaut, **Default aus**.

- `near_subdivision_level` (`BemIntegration.cuh`): kleinster Abstand d_min über je 9 Stützpunkte (Ecken,
  Kantenmitten, Mitte) beider Quads, s = größere Wurzel der Fläche. d_min/s ≥ 2: Produktionsregel; sonst
  n = ceil(2 s / d_min), mindestens 2, höchstens N (Standard 8): gleichmäßige Unterteilung beider
  Parametergebiete in n × n Zellen mit 2×2 Gauss (wie `adjacent_subdivision_level`, nur nach Abstand).
- Nur in den **dichten** Blöcken (`compute_dense_block_kernel(_H)_body`, Zweig `shared == 0`); ACA und
  Fernfeld unverändert. Option `--bem-near-subdiv[=N]` (2 bis 64, ohne Wert 8), Kernelparameter
  `near_nsub`.
- **Ohne die Option bitgleich:** Skalierung 0,5, 50 Hz: `main` und Prototyp ohne Option liefern gleiche
  Logs (18 äußere Iterationen, 8,62e-07; Phasenkräfte 0,179 / −2,08 / −0,0242 / 2,24 N) bis auf zwei
  Zeilen im Rundungsbereich (`||G^T b||/||b||` 6,301e-16 gegen 6,290e-16, atomare Summation) und die
  nicht angeforderten Dump-Zeilen. `CUDA_VISIBLE_DEVICES="" ctest -L host`: 40 von 40; vollständiges
  `ctest`: siehe Abschnitt Tests.

Genauigkeit auf Paarebene (B1, Spalten „Unterteilung“) und Wirkung im Produktionspfad:

Zeilenresiduen bei 2 Hz (V0, Operator; Zeile 1 mit +K2, Zeile 2 mit +K1):

| Skalierung | Nahfeld | Z1 gesamt (+K2) | Z2 Platte (alle) | Z2 Platte (eben) | Z2 Platte (Seite) | Z2 Platte (Kante) | Z2 Innenspule | Z2 Aussenspule | Z2 gesamt |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | Produktionsregel | 16.19 % | 45.41 % | 45.58 % | - | 10.60 % | 9.32 % | 4.42 % | 7.66 % |
| 0.5 | --bem-near-subdiv=8 | 16.19 % | 47.97 % | 48.16 % | - | 3.95 % | 9.31 % | 4.41 % | 7.63 % |
| 1.0 | Produktionsregel | 7.44 % | 19.59 % | 19.62 % | 18.49 % | 14.43 % | 7.92 % | 2.26 % | 4.89 % |
| 1.0 | --bem-near-subdiv=8 | 7.44 % | 19.58 % | 19.63 % | 12.80 % | 13.95 % | 7.92 % | 2.26 % | 4.89 % |

Kräfte und Felder bei 50 Hz, 11,3 mm (BEM-FEM mit/ohne Prototyp):

| Skalierung | Nahfeld | F_z Phasen 0/45/90/135 [N] | F_z Mittel [N] | F_ref [N] | aeussere Iterationen |
|---|---|---|---:|---:|---|
| 0.5 | Produktionsregel | 0.179 / -2.08 / -0.0242 / 2.24 | 0.0775 | 0.85333 | 18 (8.62e-07) |
| 0.5 | --bem-near-subdiv=8 | 0.12 / -1.97 / -0.0209 / 2.07 | 0.0498 | 0.85333 | 18 (7.26e-07) |
| 1.0 | Produktionsregel | 1.14 / -2.58 / -0.0459 / 3.67 | 0.5448 | 0.96649 | 19 (8.49e-07) |
| 1.0 | --bem-near-subdiv=8 | 1.08 / -2.59 / -0.0472 / 3.62 | 0.5186 | 0.96649 | 19 (8.69e-07) |

Felder (Zeile „alle“; Projektion auf Biot-Savart, L2-Fehler gegen den Referenzpfad):

| Skalierung | Koerper | Phase | proj Biot-Savart (Produktion) | proj Biot-Savart (Unterteilung) | L2 Referenzpfad (Produktion) | L2 Referenzpfad (Unterteilung) |
|---|---|---|---:|---:|---:|---:|
| 0.5 | Platte | Re | 1.0029 | 0.9710 | 0.3341 | 0.2915 |
| 0.5 | Platte | Im | 0.9304 | 0.9820 | 0.4631 | 0.4709 |
| 0.5 | Innenspule | Re | 0.5401 | 0.5403 | 0.5460 | 0.5458 |
| 0.5 | Innenspule | Im | 1.2983 | 1.3457 | 0.7895 | 0.8390 |
| 0.5 | Aussenspule | Re | 0.4720 | 0.4718 | 0.6530 | 0.6539 |
| 0.5 | Aussenspule | Im | 3.1959 | 3.2900 | 5.7504 | 5.8813 |
| 1.0 | Platte | Re | 1.0833 | 1.0876 | 0.3410 | 0.3345 |
| 1.0 | Platte | Im | 1.2892 | 1.2890 | 0.5161 | 0.5171 |
| 1.0 | Innenspule | Re | 0.8706 | 0.8706 | 0.3958 | 0.3958 |
| 1.0 | Innenspule | Im | 1.4810 | 1.4820 | 0.6995 | 0.6998 |
| 1.0 | Aussenspule | Re | 0.7879 | 0.7879 | 0.4870 | 0.4872 |
| 1.0 | Aussenspule | Im | 3.7302 | 3.7181 | 4.2233 | 4.2044 |

**Kosten:** `--bem-row-check` bei Skalierung 1,0 (2 Hz; Aufbau der BEM-Blöcke dominiert), ruhige GPU, Wanduhrzeit ohne / mit `--bem-near-subdiv`: 477 s / 824 s (Faktor 1.73); GPU-Speicherspitze (nvidia-smi, 1-s-Abtastung, einschließlich Grundbelegung): 2613 MiB / 2671 MiB. Die Blöcke haben dieselbe Größe; der Unterschied der Speicherspitze (58 MiB) liegt im Rauschen der Abtastung.

**Messung:** Der Prototyp verbessert die Paargenauigkeit stark (G unter 1e-3 ab d/s = 0,1, D unter 3e-3 ab d/s = 0,2), ändert
aber das Plattenresiduum von Zeile 2 nicht (Deckel/Boden 19,62 % → 19,63 % bei 1,0), und F_z bewegt sich
(Skalierung 0,5) von 0,0775 N auf 0,0498 N weg von der Referenz. Das Nahfeld disjunkter Paare ist damit
nicht die Ursache des Plattenfehlers.

## Tests

- `test_bem_near_field` (neu, Marke host, CPU, 7 s): B1 mit n = 8 gegen 16 (Konvergenz der Referenz,
  Fehler G > 5 % bei d/s = 0,05 als Gegenprobe, G < 1 % bei d/s = 4, Unterteilung: G < 1 % und D < 2 %
  bei d/s = 0,5), B2 mit h = 10 und 5 mm. `--table` rechnet die Tabellen dieses Berichts.
- `CUDA_VISIBLE_DEVICES="" ctest -L host`: **40 von 40** (vorher 39). Vollständiges `ctest` (mit GPU): **73 von 73** grün (vorher 72); die Fingerprint-Tests sind unverändert.

## Offene Punkte

- **Die Ursache des Plattenfehlers in Zeile 2 und des F_z-Defizits ist offen.** Gemessen ist nur, was es
  nicht ist (Nahfeldquadratur disjunkter Paare).
- Skalierung 1,5 nicht gerechnet (Speicher, siehe A); das Muster A hat damit vier Punkte, ohne
  Konvergenz.
- Der Plattenfehler fällt bei 0,5 → 0,75 → 1,0 und steigt bei 1,25 (siehe A); warum, ist nicht untersucht.
- Die Kosten des Prototyps sind nur an einem Netz (1,0) gemessen. Der Prototyp ist ohne belegten
  Nutzen für den Plattenfehler; ob man ihn für die Genauigkeit der Blöcke an sich behält, ist eine
  Entscheidung für den Auftraggeber (Default aus, kein Einfluss ohne Option).
- B1 betrachtet gleichmäßig unterteilte Referenzen für parallele Quads; gekrümmte und stark verzerrte
  disjunkte Paare sind nicht geprüft.
- Die Interpretationen oben (gewöhnlicher Diskretisierungsfehler, Kandidaten für den Plattenfehler)
  sind nicht gemessen.

## Befehle

Aus `build/` (Netze `near/hex_sS.vtk` mit `tools/gmsh/team28_hex.py --scale S`):

```
./fembem --reference --harmonic=50 --fixed-plate --plate-z=11.3 --mesh=near/hex_sS.vtk \
    --harmonic-field-dump=near/ref_sS.csv
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=near/hex_sS.vtk --bem-row-check
./fembem --harmonic=50 --harmonic-inner-direct --fixed-plate --plate-z=11.3 \
    --mesh=near/hex_sS.vtk --harmonic-field-dump=near/bf_sS.csv
# Prototyp: zusaetzlich --bem-near-subdiv (oder =N)
python3 ../tools/bem_trend_tables.py near 0.5 0.75 1.0 1.25
python3 ../tools/bem_panel_sizes.py near/hex_s1.0.vtk
./test_bem_near_field --table
CUDA_VISIBLE_DEVICES="" ctest -L host
```

## Nachtrag (force-estimator-check, 06.11.2026)

Drei Aussagen dieses Berichts sind nach `force-estimator-check.md` zu relativieren; die Zahlen oben
bleiben unverändert.

1. **Tabelle A vergleicht zwei Schätzer** (Maxwell BEM-FEM gegen Lorentz Referenz). Mit Lorentz auf beiden
   Seiten ist Lorentz BEM-FEM / Referenz 1,75 / 1,78 / 1,78 / 1,98 (0,5 / 0,75 / 1,0 / 1,25): ein
   Überschuss, kein Defizit.
2. **Der Einbruch bei 1,25 ist ACA-Kompression.** Zeile 2 gesamt 9,67 % → 0,33 % mit dichtem Aufbau
   (`--bem-eta=1e-9`), Platte 34,4 % → 1,4 %; das Netz ist unauffällig.
3. **Die Plattenresiduen von Zeile 2 (45 / 28 / 20 / 34 %) waren überwiegend Kompressionsfehler**
   (dicht bei 1,0: Platte 2,2 %, gesamt 0,77 %). Die Aussage „Nahfeldquadratur nicht die Ursache“ wurde
   mit dem komprimierten Operator gemessen; dicht wirkt `--bem-near-subdiv` auf die Platte
   (2,21 % → 1,19 %, Skalierung 1,0).
