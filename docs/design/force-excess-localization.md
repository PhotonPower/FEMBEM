# Eingrenzung des Kraftüberschusses im BEM-FEM-Pfad

Stand: 07.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), Linux. Basis: Zweig
`force-estimator-check` 3b7c906 (PR #25 war beim Start nicht gemergt), Zweig `force-excess-localization`.
Netze `near/hex_sS.vtk` aus `tools/gmsh/team28_hex.py`, Platte bei 11,3 mm. Alle BEM-FEM-Läufe **dicht**
(`--bem-eta=1e-9`), Kraft als Lorentz-Volumenkraft. Logs, Dumps und Tabellen in `build/excess/` (nicht im
Repo). Es wurde nichts repariert; am Produktivcode ist nur eine reine Diagnoseausgabe dazugekommen
(Wirbelstromgrößen für Kantenelemente, ohne Einfluss auf Lösung und Kraft).

## Ergebnis in Kürze

**In welchem Teil des BEM-FEM-Pfads entsteht der Überschuss?** Im **Quellfeld**, also in der Erregung
bzw. der Kopplung an die Spulen (magnetostatischer Anteil), **nicht** in der Reaktion der Platte und nicht
in D und G.

**Gemessen**

1. **Schon ohne Plattenreaktion ist das Feld falsch** (Schritt 1, 0,01 Hz). Die Platte ist dort
   magnetisch transparent (Referenzpfad |B_i|/|B_r| der Platte 4e-5, Q_eff 1e-4), Biot-Savart ist dort die
   exakte Lösung. Der Referenzpfad trifft es (B_z der Platte gegen Biot-Savart: 0,97 bis 1,01 je Radiusbin,
   Skalierung 1,0). Der BEM-FEM-Pfad nicht: **1,28 / 1,29 / 0,86 / 0,79** (Radiusbins von innen nach
   außen; Skalierung 1,0), also zu viel Fluss im Zentrum und 14 bis 21 % zu wenig im Rückfluss (Skalierung 0,5: 27 bis 30 %). Projektion auf
   Biot-Savart (Skalierung 1,0): Platte 1,09 / Innenspule 0,88 / Außenspule 0,79 gegen 0,99 / 1,03 / 1,00 im
   Referenzpfad.
2. **Die Wirbelströme folgen diesem Feld.** Das Moment im Nulldurchgang, |m_z|(Null), liegt bei 50 Hz um das
   2,5- bis 2,6-fache über dem Referenzpfad; mit der Stokes-Rechnung aus dem B_z-Dump des jeweiligen Pfads
   allein wird es auf 1 % bis 2 % reproduziert (beide Pfade). Der Kernel ist also nicht die Ursache, das
   Plattenfeld ist es (Schritt 2).
3. **Der K-Widerspruch besteht dicht fort und sitzt an den Spulen** (Schritt 3). Zeile 1 verlangt K2, Zeile
   2 verlangt K1, mit Zeile 2 dicht unter 1 % für K1 und rund 17 % für K2 (gesamt, Skalierung 1,0). An der
   Platte gibt es keinen Widerspruch (K1 passt in beiden Zeilen).
4. **Das Feld hängt von der Eichung ab** (Zusatzmessung, 0,5): ohne Baum-Kobaum-Eichung (`--nogauge`)
   ändert sich B_z der Platte gegen Biot-Savart von 1,13 / 1,06 / 0,73 / 0,70 auf 1,28 / 1,05 / 1,03 / 0,90.
   Das Feld einer konsistenten Formulierung dürfte davon nicht abhängen.
5. **F_z-Verhältnis über der Frequenz** (Schritt 4, 0,5): 7,2 / 3,3 / 1,75 / 1,68 / 1,57 bei 0,1 / 1 / 10 / 50 /
   200 Hz. Oberhalb von 10 Hz fast konstant (Amplitudenfehler), unterhalb steigt es stark an, weil der
   BEM-FEM-Pfad dort einen Imaginärteil hat, der um Größenordnungen über dem physikalischen liegt.

**Nicht geklärt:** welcher Teil der Erregung bzw. Kopplung den Fehler erzeugt (Eichung und Kohomologie,
K-Wahl an den Spulen, Quellterm). Das ist Interpretation, siehe unten.

## Schritt 1: Quellfeld ohne Plattenreaktion

### Ist die Plattenreaktion bei 0,01 Hz klein?

Gemessen, Referenzpfad (`ref_lf_s*.log`, Felddump `ref_lf_s*.csv`):

| Skalierung | Q_eff | |m_z| Max | |m_z| Null | \|B_i\|/\|B_r\| Platte |
|---|---:|---:|---:|---:|
| 0,5 | 1,4e-4 | 3,5e-8 A m² | 2,5e-4 A m² | 3,8e-5 |
| 1,0 | 1,0e-4 | 3,9e-8 A m² | 2,9e-4 A m² | 4,0e-5 |

Die Plattenreaktion (Imaginärteil, Wirbelstromfeld) ist rund vier Größenordnungen unter dem Quellfeld.
0,01 Hz genügt, die Frequenz musste nicht weiter gesenkt werden.

### BEM-FEM gegen Referenz und Biot-Savart

`tools/bem_lowfreq_profile.py`: Mittel von B_z (Realteil) in den Plattenelementen je Radiusbin. Biot-Savart
ist hier die exakte Lösung (Spulenfäden, `compare_fields.py`).

Skalierung 1,0 (1872 Plattenelemente):

| Radius [m] | n | B_z Biot-Savart | B_z Referenz | B_z BEM-FEM | Ref / BS | BEM-FEM / BS |
|---|---:|---:|---:|---:|---:|---:|
| 0,00 bis 0,02 | 396 | 0,05613 | 0,05599 | 0,07208 | 0,998 | 1,284 |
| 0,02 bis 0,04 | 660 | 0,0246 | 0,02397 | 0,03166 | 0,974 | 1,287 |
| 0,04 bis 0,06 | 672 | −0,03806 | −0,03846 | −0,0328 | 1,010 | 0,862 |
| 0,06 bis 0,10 | 144 | −0,06118 | −0,06097 | −0,04846 | 0,997 | 0,792 |

Skalierung 0,5 (312 Plattenelemente): Referenz 0,990 / 0,901 / 1,046 / 1,004, BEM-FEM 1,126 / 1,061 / 0,727 /
0,704 (gleiche Bins).

Projektion der Felder auf Biot-Savart (`compare_fields.py --biot-savart` **ohne** `--freq`, Zeile „alle“,
Realteil) und L2-Abweichung:

| Skalierung | Körper | proj Referenz | L2 Referenz | proj BEM-FEM | L2 BEM-FEM |
|---|---|---:|---:|---:|---:|
| 1,0 | Platte | 0,994 | 0,014 | 1,094 | 0,205 |
| 1,0 | Innenspule | 1,026 | 0,103 | 0,877 | 0,378 |
| 1,0 | Außenspule | 0,999 | 0,040 | 0,789 | 0,478 |
| 0,5 | Platte | 0,972 | 0,056 | 1,004 | 0,220 |
| 0,5 | Innenspule | 0,906 | 0,219 | 0,540 | 0,596 |
| 0,5 | Außenspule | 0,869 | 0,190 | 0,478 | 0,704 |

Der Imaginärteil, der bei 0,01 Hz verschwinden müsste, ist im BEM-FEM-Pfad deutlich größer als im
Referenzpfad (\|B_i\|/\|B_r\|, Skalierung 1,0): Platte 1,9e-4 gegen 4,0e-5, Innenspule 2,0e-5 gegen 4,0e-6,
Außenspule 3,2e-6 gegen 4,3e-7 (Skalierung 0,5: 2,9e-3 gegen 3,8e-5 usw.). Er fällt mit der Verfeinerung
(0,5 → 1,0 etwa um den Faktor 15 in der Platte), ist aber nicht ∝ ω (siehe Schritt 4).

**Entscheidung nach dem Auftrag:** BEM-FEM weicht schon hier ab. Der Fehler sitzt in der Erregung oder der
Kopplung an den Spulen. Schritt 2 wurde kurz gehalten, Schritt 3 vollständig.

### Zusatzmessung: Abhängigkeit von der Eichung (Skalierung 0,5, 0,01 Hz)

`--nogauge` (Baum-Kobaum-Eichung weggelassen; dabei werden auch keine Kohomologie-Generatoren gefunden,
„0 Kohomologie-Generatoren“), 4 äußere Iterationen, rel. Residuum 4,2e-7:

| Radius [m] | Ref / BS | BEM-FEM / BS (geeicht) | BEM-FEM / BS (`--nogauge`) |
|---|---:|---:|---:|
| 0,00 bis 0,02 | 0,990 | 1,126 | 1,278 |
| 0,02 bis 0,04 | 0,901 | 1,061 | 1,046 |
| 0,04 bis 0,06 | 1,046 | 0,727 | 1,033 |
| 0,06 bis 0,10 | 1,004 | 0,704 | 0,896 |

Der Rückfluss (äußere Bins) kommt ohne Eichung deutlich näher an Biot-Savart (0,70 → 0,90, 0,73 → 1,03),
das Zentrum nicht (1,13 → 1,28). Ein Lauf bei 50 Hz ohne Eichung brach nach 900 s mit Zeitlimit ab (kein
Ergebnis); die Kraftfolge der Eichung ist deshalb nicht gemessen.

## Schritt 2: Wirbelstromgrößen nebeneinander

**Wahl:** Diagnoseausgabe im BEM-FEM-Pfad, `harmonic_eddy_diag_body` (`HarmonicLorentz.cuh`), Gegenstück zu
`nodal_eddy_diag` für Kantenelemente. Begründung: dieselben Größen mit denselben Formeln und Bezeichnungen
(J = −σωA, m_z = ½∫(xJ_y − yJ_x)dV, P = (P_r + P_i)/2, Q_eff = |m_z(A_i)|/|m_z(A_r)|), ohne Umweg über einen
Skript-Nachbau der Kantenbasis. Der Kernel stimmt auf einem affinen Feld mit dem Knotenkernel auf
Rundung überein (`test_lorentz_kernel_agreement`, Fall 1b: m_z und P je 6e-16 bis 7e-16). Quadratur wie
`harmonic_lorentz_body` (2×2×2). Die Ausgabe ändert Lösung und Kraft nicht.

Tabelle bei 50 Hz, dicht (`tools/bem_eddy_table.py`):

| Skalierung | Größe | BEM-FEM | Referenz | BEM-FEM / Ref |
|---|---|---:|---:|---:|
| 0,5 | \|m_z\| Strommaximum [A m²] | 1,139 | 0,7014 | 1,623 |
| 0,5 | \|m_z\| Nulldurchgang [A m²] | 2,284 | 0,8693 | 2,628 |
| 0,5 | P [W] | 14,98 | 12,14 | 1,234 |
| 0,5 | Q_eff | 0,4984 | 0,8069 | 0,618 |
| 0,5 | F_z Lorentz [N] | 1,431 | 0,8533 | 1,677 |
| 1,0 | \|m_z\| Strommaximum [A m²] | 1,429 | 0,7843 | 1,822 |
| 1,0 | \|m_z\| Nulldurchgang [A m²] | 2,603 | 1,033 | 2,521 |
| 1,0 | P [W] | 19,70 | 12,94 | 1,523 |
| 1,0 | Q_eff | 0,5489 | 0,7594 | 0,723 |
| 1,0 | F_z Lorentz [N] | 1,685 | 0,9665 | 1,743 |

(Die BEM-FEM-Werte stammen aus `bf_f50_sS.log`; Referenzlog bei 1,0 ist `near/ref_s1.0.log`.)

**Gemessen: Ist der Plattenstrom zu groß oder das Feld am Ort des Stroms?** Beides hängt zusammen. Der Strom
im Nulldurchgang (aus A_r, d. h. vom Quellfeld getrieben) ist 2,5-mal zu groß, der Strom im Maximum (aus A_i,
vom Wirbelstromfeld getrieben) 1,6- bis 1,8-mal. Das Verhältnis des Stroms im Nulldurchgang ist von 0,1 bis
10 Hz (Skalierung 0,5) konstant 2,27, also frequenzunabhängig und damit eine Eigenschaft des Quellfelds
(siehe Schritt 4).

**Stokes-Prüfung** (`tools/bem_stokes_check.py`): aus dem B_z-Dump des jeweiligen Pfads allein folgt
|m_z(Null)| = (σω/4π)∫Φ(r)dV, unabhängig von beiden Kernen:

| Fall | Stokes aus B_z | berichtet |
|---|---:|---:|
| BEM-FEM, 1 Hz, 0,5 | 0,05711 | 0,05569 |
| Referenz, 1 Hz, 0,5 | 0,02425 | 0,02457 |
| BEM-FEM, 50 Hz, 1,0 | 2,625 | 2,603 |
| Referenz, 50 Hz, 1,0 | 1,032 | 1,033 |

Die berichteten Momente sind mit dem jeweils eigenen B_z konsistent (≤ 2,5 %); der Unterschied zwischen den
Pfaden steckt im B_z der Platte. Dieses B_z unterscheidet sich im Betrag nur wenig (|B_r| der Platte 1,03 gegen
0,98 bei 50 Hz, Skalierung 0,5), aber im radialen Profil (Schritt 1: Rückfluss 20 bis 30 % zu schwach). Das
Moment ist ein über den Radius aufintegrierter Fluss und reagiert darauf stark: bei 0,5 liegt der mittlere
B_z-Rückfluss der Platte (r > 4 cm) bei −0,025 und −0,042 T im BEM-FEM-Pfad gegen −0,036 und −0,059 T im
Referenzpfad.

**Gemessen: Passen die Verhältnisse multiplikativ zum Kraftverhältnis 1,74?** Teilweise. Der Term A_i × B_r
der Kraftformel gibt grob |m_z(Max)|-Verhältnis × |B_r|-Verhältnis: 0,5: 1,62 × 1,05 = 1,7 (Kraft 1,68); 1,0:
1,82 × 1,10 ≈ 2,0 (Kraft 1,74). Die Größenordnung stimmt, eine genaue Zerlegung wurde nicht gemacht (A ist
im Dump nicht enthalten). Q_eff liegt im BEM-FEM-Pfad niedriger (0,62 und 0,72 des Referenzwerts): die
Phasenverschiebung zwischen Strom und Feld ist also **nicht** zu groß, sondern zu klein; der Überschuss ist
kein Phasenfehler in dem Sinn, dass die Reaktion zu stark wäre.

## Schritt 3: K1/K2 dicht

`--bem-row-check --bem-eta=1e-9`, 2 Hz (`excess/row_dense_s0.5.log`, `force/row_dense_s1.0.log`). Die
vollständigen Tabellen (alle Klassen, ±K1/±K2, Zeile 1 und 2, V0/V1/V2, T-Test) stehen in den Logs. Auszug V0,
Zeile „alle“ je Körper (Kantenklassen siehe Log):

Zeile 1 (FEM auf Randkanten), `||K a − T k − b|| / ||K a − b||`:

| Skalierung | Körper | +K1 | −K1 | +K2 | −K2 |
|---|---|---:|---:|---:|---:|
| 1,0 | Platte | 1,58 % | 200,31 % | 0,60 % | 200,47 % |
| 1,0 | Innenspule | 31,53 % | 186,00 % | 7,77 % | 201,24 % |
| 1,0 | Außenspule | 25,28 % | 190,97 % | 6,94 % | 200,31 % |
| 1,0 | alle | 29,40 % | 187,85 % | **7,44 %** | 200,94 % |
| 0,5 | Platte | 2,63 % | 201,74 % | 2,31 % | 201,85 % |
| 0,5 | Innenspule | 32,54 % | 186,26 % | 16,89 % | 201,76 % |
| 0,5 | Außenspule | 30,51 % | 189,94 % | 15,15 % | 199,45 % |
| 0,5 | alle | 31,58 % | 187,74 % | **16,19 %** | 201,04 % |

Zeile 2 (BEM), `||(+0,5 T − D) a + µ0 G k|| / ||(+0,5 T − D) a||`:

| Skalierung | Körper | +K1 | −K1 | +K2 | −K2 |
|---|---|---:|---:|---:|---:|
| 1,0 | Platte | 2,21 % | 199,58 % | 22,42 % | 219,44 % |
| 1,0 | Innenspule | 1,11 % | 200,96 % | 30,62 % | 222,98 % |
| 1,0 | Außenspule | 0,54 % | 199,48 % | 6,68 % | 198,26 % |
| 1,0 | alle | **0,77 %** | 199,88 % | 16,89 % | 205,30 % |
| 0,5 | Platte | 9,73 % | 191,39 % | 16,41 % | 209,99 % |
| 0,5 | Innenspule | 2,85 % | 201,75 % | 32,44 % | 224,57 % |
| 0,5 | Außenspule | 0,95 % | 199,30 % | 7,02 % | 198,08 % |
| 0,5 | alle | **1,94 %** | 199,85 % | 17,52 % | 205,28 % |

T-Test (`||T k − g|| / ||g||`, V0): **0,00 %** in allen Körpern und Klassen für K1 und K2 (V1: 100 % und mehr,
wie in `bem-coupling-fixes.md`): T bildet k exakt auf das Randintegral ab, die Operatoranwendung gibt die
Nachrechnung am Host auf Rundungsniveau wieder (`T-Kontrolle` 1e-16).

**Gemessen:** Der Widerspruch bleibt **dicht bestehen**. Zeile 2 verlangt eindeutig K1 (gesamt 0,77 % /
1,94 % für K1 gegen 16,9 % / 17,5 % für K2); Zeile 1 verlangt K2 (7,4 % / 16,2 % gegen 29,4 % / 31,6 %). Er
sitzt an den **Spulen** (Zeile 1: Innenspule 31,5 % für K1 gegen 7,8 % für K2; Zeile 2: 1,1 % gegen 30,6 %). Die
Platte ist in Zeile 1 unentschieden (K1 1,6 %, K2 0,6 %, beides klein) und verlangt in Zeile 2 K1 (2,2 % gegen
22,4 %), zeigt also keinen Widerspruch. Die Zeile-2-Werte in `bem-coupling-fixes.md` waren komprimiert
gemessen; dicht ist der Abstand in Zeile 2 mit 0,8 % für K1 gegen 17 % für K2 deutlich.

**Hinweis zur Einordnung (Messung):** Auch mit exakten Feldern ist Zeile 1 nicht null: die inneren Zeilen
(`K a_ex − b`, ohne BEM) liegen bei 24,4 % (Skalierung 1,0) und 68,9 % (0,5). Der Wert 7,4 % für K2 ist also
nicht eindeutig als Fehler der Kopplung zu lesen; ein Teil kommt aus dem Vergleich Fadenmodell gegen Volumenstrom.

## Schritt 4: Frequenzreihe (Skalierung 0,5, dicht)

Schritt 4 war optional und ist als Nebenprodukt der Frequenzreihe von Schritt 1 angefallen (je Frequenz ein
BEM-FEM- und ein Referenzlauf). Kosten, gemessen: ein dichter BEM-FEM-Lauf bei 0,5 rund 2 Minuten, bei 1,0
rund 21 Minuten (`force/dense_bf_s1.0.log`: 1265 s). Eine Abschätzung vorab gab es nicht, der Aufwand war
klein genug, die Reihe gleich zu rechnen. F_z Lorentz (`tools/bem_eddy_table.py`):

| f [Hz] | F_z BEM-FEM [N] | F_z Referenz [N] | Verhältnis | \|m_z\|(Null) Verh. | Q_eff Verh. | \|B_i\|/\|B_r\| Platte BEM-FEM / Ref |
|---:|---:|---:|---:|---:|---:|---|
| 0,1 | 2,857e-5 | 3,989e-6 | 7,16 | 2,27 | 1,14 | 1,4e-2 / 3,8e-4 |
| 1 | 1,328e-3 | 3,988e-4 | 3,33 | 2,27 | 0,82 | 3,9e-2 / 3,8e-3 |
| 10 | 0,06917 | 0,03961 | 1,75 | 2,28 | 0,69 | 4,9e-2 / 3,8e-2 |
| 50 | 1,431 | 0,8533 | 1,68 | 2,63 | 0,62 | 0,197 / 0,176 |
| 200 | 7,722 | 4,915 | 1,57 | 1,63 | 1,47 | 0,385 / 0,409 |

**Gemessen:** Ab 10 Hz ist das Kraftverhältnis fast konstant (1,75 / 1,68 / 1,57): das spricht für einen
Amplitudenfehler. Unterhalb von 10 Hz steigt es stark an (3,3 bei 1 Hz, 7,2 bei 0,1 Hz). Der Grund ist der
Imaginärteil im BEM-FEM-Pfad, der bei 0,1 Hz das Vierzigfache (1,4e-2 gegen 3,8e-4) und bei 1 Hz das Zehnfache
des physikalischen Werts hat; er wächst von 0,01 bis 10 Hz nur wie etwa f^0,4 statt wie f (0,5: 2,9e-3 bei 0,01 Hz, 4,9e-2 bei 10 Hz). Das gibt der Kraft
bei kleinen Frequenzen einen Anteil, den es im Referenzpfad nicht gibt. Die Kraft selbst ist dort so klein
(1e-3 N), dass sie für die Aufgabe keine Rolle spielt.

## Interpretation (nicht geprüft)

- **Wahrscheinlichste Ursache nach den Messungen:** das statische Feld der Spulen im BEM-FEM-Pfad ist falsch
  (zu viel Fluss im Zentrum, zu wenig Rückfluss), und die Wirbelströme der Platte und damit die Kraft folgen
  daraus. Das passt zu allen vier Hinweisen: Fehler schon bei 0,01 Hz gegen die exakte Lösung, frequenz-
  unabhängiges Momentverhältnis (2,27) bei kleiner Frequenz, Konsistenz der Wirbelströme mit dem eigenen
  B_z (Stokes), Verhältnis F_z ab 10 Hz konstant.
- **Kandidaten innerhalb von Erregung/Kopplung**, nicht getrennt gemessen: (a) Eichung und Kohomologie der
  Ringspulen (der Fluss durch die Spulenringe ist genau die Größe, die falsch aussieht; die gemessene
  Abhängigkeit von `--nogauge` stützt das, erklärt das Zentrum aber nicht); (b) die K-Wahl an den Spulen
  (Zeile 1 gegen Zeile 2, Schritt 3); (c) der Quellterm (Spulenstrom auf Kantenelementen).
  `--nogauge` schaltet Baum-Kobaum-Eichung und Kohomologie gemeinsam ab; welcher der beiden Teile wirkt,
  ist nicht getrennt.
- **Der Imaginärteil bei kleiner Frequenz** (bis zu 10³-mal physikalisch) könnte dieselbe Quelle haben (eine
  nicht verschwindende Phasenverschiebung im Operator); nicht geprüft.
- Die Zeilenresiduen bei 2 Hz (Zeile 2 dicht unter 1 %) schließen D und G als Ursache aus, auch die Platte
  (Zeile 2, K1: 2,2 %); sie sagen nichts über die Spulenzeilen für Zeile 1.

## Abweichungen vom Auftrag

- **Zusatzläufe:** Frequenzreihe 0,1 bis 200 Hz bei 0,5 (Schritt 4 optional, aber billig und liefert die
  Aussage über den Imaginärteil) und `--nogauge` bei 0,5 (Zusatzmessung zur Eichung). Der `--nogauge`-Lauf
  bei 50 Hz kam nicht ins Ziel (Zeitlimit 900 s).
- **Schritt 1 bei 0,5 und 1,0;** bei 0,01 Hz ist die Plattenreaktion mit \|B_i\|/\|B_r\| 4e-5 (Referenz)
  vernachlässigbar, die Frequenz musste nicht gesenkt werden.
- **Schritt 2:** als Diagnoseausgabe im BEM-FEM-Pfad umgesetzt (erlaubt, wenn das Verhalten ohne sie
  bitgleich bleibt: die Ausgabe liest nur A und schreibt nur auf die Konsole; die Lösung wird nicht verändert).
  Bei Skalierung 1,25 wurde nichts gerechnet.
- **Werkzeug-Befund:** in den Felddumps bedeutet `body` 1 = Platte, 2 = Innenspule, 3 = Außenspule (nicht die
  Komponenten-IDs des Netzes); `tools/bem_eddy_table.py` und `tools/bem_lowfreq_profile.py` benutzen das.

## Tests

- `CUDA_VISIBLE_DEVICES="" ctest -L host`: **41 von 41** (vorher 41, auf `force-estimator-check`; die Zahl bleibt, der bestehende Test wurde erweitert)
- Vollständiges `ctest` (mit GPU): **74 von 74** grün (vorher 74). Die Fingerprint-Tests sind unverändert.

Der Test `test_lorentz_kernel_agreement` wurde um Fall 1b erweitert (Wirbelstromdiagnose Kante gegen Knoten).

## Offene Punkte

- **Welcher Teil von Erregung/Kopplung den Fehler im Quellfeld erzeugt** (Eichung, Kohomologie, K an den
  Spulen, Quellterm) ist nicht getrennt. Ein naheliegender nächster Schritt (nicht Teil dieses Auftrags): den
  Spulenfluss durch die Ringe und die Kohomologie-Generatoren einzeln messen; und den Magnetostatikfall ohne
  Platte (nur Spulen) im BEM-FEM-Pfad gegen Biot-Savart.
- Die Beobachtung, dass Zeile 1 für die Spulen K2 und Zeile 2 K1 verlangt, bleibt offen (Auftrag: nur
  messen).
- Der Imaginärteil bei kleinen Frequenzen und seine Frequenzabhängigkeit sind nur gemessen, nicht erklärt.
- Skalierung 1,25 wurde nicht untersucht.
- Die Aussage „Referenz stimmt“ stützt sich hier auf die Biot-Savart-Übereinstimmung bei 0,01 Hz (0,97 bis
  1,01 bei Skalierung 1,0), nicht auf eine Messung der Kraft.

## Befehle

Aus `build/` (Netze und Referenzläufe wie in `bem-near-field.md`; `BF` = BEM-FEM-Binary, dicht):

```
# Schritt 1: 0,01 Hz
./fembem --reference --harmonic=0.01 --fixed-plate --plate-z=11.3 --mesh=near/hex_sS.vtk \
    --harmonic-field-dump=excess/ref_lf_sS.csv
./fembem --harmonic=0.01 --harmonic-inner-direct --fixed-plate --plate-z=11.3 --mesh=near/hex_sS.vtk \
    --bem-eta=1e-9 --harmonic-field-dump=excess/bf_lf_sS.csv
python3 ../tools/bem_lowfreq_profile.py excess/bf_lf_sS.csv excess/ref_lf_sS.csv
python3 ../tools/compare_fields.py excess/bf_lf_sS.csv excess/ref_lf_sS.csv --mesh near/hex_sS.vtk \
    --biot-savart --markdown
./fembem ... --nogauge ...        # Zusatzmessung Eichung (0,5)
# Schritt 2 und 4: Frequenzreihe, Skalierung 0,5
./fembem --reference --harmonic=F ...  ;  ./fembem --harmonic=F --harmonic-inner-direct ... --bem-eta=1e-9
python3 ../tools/bem_eddy_table.py excess 0.5 0.1 1 10 50 200 --fields
python3 ../tools/bem_stokes_check.py near/hex_s1.0.vtk 50 excess/bf_f50_s1.0.csv excess/ref_f50_s1.0.csv
# Schritt 3
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=near/hex_sS.vtk --bem-row-check --bem-eta=1e-9
CUDA_VISIBLE_DEVICES="" ctest -L host
ctest
```
