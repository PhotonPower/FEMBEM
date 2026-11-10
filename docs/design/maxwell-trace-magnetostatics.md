# Phase 2: Magnetostatik der TEAM-28-Spulen mit der Maxwell-Spur-Kopplung (dicht)

Stand: 10.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), 32 CPU-Kerne, 1 TB RAM, Linux; die neue Rechnung läuft nur auf dem Host
(Double, dichtes Schurkomplement, Eigen mit OpenMP). Basis: `main` a2b0268 (Auftrag), Zweig `maxwell-trace-magnetostatics`. Logs und Felddumps in `build/trace/`
(nicht im Repo). Der Lösungspfad, `fem.cu`, die alten Operatoren und der Referenzpfad sind **unverändert**; alles Neue steht in `Trace*`-Dateien, in `tests/` und in `CMakeLists.txt`.
Netze: `near/hex_s0.5.vtk` bis `hex_s1.25.vtk` (nicht im Repo, wie in `force-excess-localization.md`), Platte bei 11,3 mm (Datei: 3,8 mm, Knoten verschoben).

## Ergebnis in Kürze

**Abnahmekriterium 2 des Fahrplans ist erreicht: Auf TEAM 28 liegt das magnetostatische Feld mit der Maxwell-Spur-Kopplung nach 0,9 % (Platte), 1,7 % (Innenspule) und
3,0 % (Außenspule) L2-Fehler gegen den Volumenstrom-Biot-Savart bei Skalierung 1,0, konvergiert mit Ordnung ≈ 2 über vier Netze, und ist besser als der Referenzpfad
(1,4 / 10,3 / 4,0 %) und um eine Größenordnung besser als der alte BEM-FEM-Pfad (20,5 / 37,8 / 47,8 %).** Das +28 % im Plattenzentrum des alten Pfads (B_z-Verhältnis 1,284)
ist verschwunden (0,997). Die Größen sind direkt vergleichbar (gleiche Netze, Orte, Maße), mit einer Einschränkung: die alten Tabellen messen gegen das Fadenmodell, hier gegen den Volumenstrom (Unterschied 0,1 bis 3 % am Spulenrand, siehe Phase 1a).

* **0a:** Die Skalierung ψ̃ = μ₀ψ, c̃ = μ₀c ist fest eingebaut; die Lösung ändert sich um 3·10⁻¹⁵, S ist nicht mehr „numerisch singulär“ (Kond 1,9·10⁴ an einem kleinen Netz gegen ∞).
* **0b:** Die Abweichung c_Meridian = 0,945 war **kein Löserfehler**: c_Meridian stimmt mit dem Nettostrom der rechten Seite überein (0,94508 gegen 0,94508, qM = 1 ± 10⁻⁴). Die Projektion ändert den
  Nettostrom, und der projizierte Nettostrom liegt um einen O(h²)-Betrag unter N·I (5,5 % / 1,4 % / 0,35 % bei 8 / 16 / 32 Winkelschichten). Gewählt: **Renormierung nach der Projektion**; danach
  c_Meridian = N·I auf 5·10⁻⁷.
* **TEAM-28-Quelle:** Die Quelle der gefitteten Spulennetze ist schon divergenzfrei (‖Gᵀf‖/‖f‖ = 10⁻¹⁵); der Nettostrom ist wie im Ring zu klein (0,57 / 0,25 / 0,14 / 0,09 % bei Skalierung 0,5 / 0,75 / 1,0 / 1,25) und
  wird auf N·I renormiert (c_Meridian dann 2·10⁻⁶ bzw. 7·10⁻⁹ neben N·I).
* **Eichunabhängigkeit:** Zwei andere Bäume ändern B um 2·10⁻⁸ bis 5·10⁻⁸ (Lösertoleranz), a um O(1). Gegenproben werden um den Faktor 27 bis 530 schlechter (Vorzeichen von λ), 25 bis 120 (ohne harmonische Spalten), 51 bis 87 (c_Meridian := 0).
* **CI:** Auf `main` ist die CI rot; die Ursache sind apt-Zeitüberschreitungen im Job „Host-Tests“ (siehe unten), kein Codefehler. Die Trace-Tests sind lokal geprüft.
* **Nicht gemacht:** Schritt 4 (harmonische Vorschau mit F_z), weil optional.

## CI-Status (Voraussetzung des Auftrags)

Die letzten vier CI-Läufe auf `main` (ab 09:39 UTC, auch nach dem Eigen-Commit dcfa36a) sind rot. Der Job „Host-Tests (ohne GPU)“ bricht in „Toolchain installieren“ ab:
„apt nach drei Versuchen nicht erfolgreich“ (3 × 300 s Zeitgrenze, `apt-Diagnose`: 24 bis 481 MB heruntergeladen, dpkg „startup packages configure“). Die anderen Jobs sind grün, auch
„Hauptprogramm bauen“ mit derselben Paketliste (also einschließlich `libeigen3-dev`) und die LaTeX-, Gmsh- und Python-Jobs; zwei früheren Läufen fiel stattdessen der Bau- bzw. der LaTeX-Job aus
demselben Grund aus. Ich habe den Lauf einmal neu gestartet (gleiches Ergebnis im Host-Test-Job). Das ist ein Infrastrukturproblem der Runner (die Dokumentation in `.github/scripts/apt-install.sh` beschreibt es), kein
Problem dieses Codes; ich habe die CI-Konfiguration nicht verändert. **Folge: Die Trace-Tests (ab `test_trace_surface`) sind in der CI noch nie gelaufen, nur lokal.**

## Gemessen

### 0a: Skalierung (`test_trace_system`, `trace_conditioning --scaling`)

* Kugel n = 4, μ₀ = 4π·10⁻⁷: |a_skal − a_unskal|/|a| = 2,9·10⁻¹⁵, |ψ_skal − ψ_unskal|/|ψ| = 1,7·10⁻¹⁵; Kond S 1,9·10⁴ (skaliert) gegen ∞ (unskaliert, kleinster Singulärwert 0); Asymmetrie 1,2·10⁻¹⁷ (Grenze 10⁻¹⁵). Bei μ₀ = 1 sind beide Fassungen identisch (0).
* T8 (μ₀ = 1) ändert sich nicht: dieselben Fehler wie in Phase 1b (n = 10: ψ 1,9 %, g 7,6 %, B 6,4 %).
* **Konditionszahl von S am gröbsten TEAM-28-Netz** (`hex_s0.5`, n_innen 688, n_Γ 2619, dicht, exakte SVD, komplex wegen jωσ):

| f / Hz | unskaliert | skaliert | diagonalskaliert (unskaliert = skaliert) |
|---|---|---|---|
| 0,01 | ∞ (numerisch singulär) | 2,5·10⁸ | 1,5·10⁶ |
| 50 | ∞ | 3,6·10⁸ | 4,2·10³ |

  Die Schätzung der Kondition von A_ii sinkt durch die Skalierung von 9,7·10⁷ auf 1,1·10⁵ (0,01 Hz) und von 1,4·10⁶ auf 1,5·10³ (50 Hz). Die Diagonalskalierung gibt dasselbe Ergebnis für beide Fassungen
  (erwartet, sie hebt jede Zeilen-/Spaltenskalierung auf). Die restliche Kondition 10⁸ der skalierten Matrix ist geometrisch (Kantenlängen gegen Randterme), nicht mehr μ₀ oder Einheiten.

### 0b: Nettostrom und c_Meridian (`test_trace_source`, Ring mit ungleichmäßiger Winkelteilung, N·I = 1)

| Winkelschichten | Nettostrom vor Projektion | nach Projektion (Defizit) | c_Meridian: ohne / nur Projektion / mit Renormierung |
|---|---|---|---|
| 8 | 0,8720 … 0,9717 (Streuung 10,8 %) | 0,94508 (5,49·10⁻²) | 0,97173 / 0,94508 / 1,0000005 |
| 16 | 0,9666 … 0,9924 | 0,98607 (1,39·10⁻²) | 0,99337 / 0,98607 / 1,00000003 |
| 32 | 0,9916 … 0,9981 | 0,99652 (3,48·10⁻³) | 0,99857 / 0,99652 / 0,9999998 |

* **Beantwortung der Fragen:** (1) Nettostrom ⟨f, v_cut⟩ vor der Projektion: schichtabhängig (0,872 … 0,972), nach der Projektion in allen Schichten gleich (Streuung 4·10⁻¹⁶), aber 5,5 % unter N·I. (2) Das gelöste c_Meridian
  stimmt mit dem Nettostrom der **verwendeten** rechten Seite überein: 0,94508 gegen 0,94508 (projiziert), 0,97173 gegen 0,97173 (unprojiziert, dort schichtweise 0,872 … 0,972, der Löser mittelt); die Spalte q_M = ⟨v_cut, (½ + n×𝒞)h_Meridian⟩
  ist 1,00000 ± 8·10⁻⁵. (3) Die Projektion ändert den Nettostrom (von der Schicht abhängig zu einem festen Wert); der feste Wert ist um O(h²) zu klein (Defizit 5,49·10⁻² → 1,39·10⁻² → 3,48·10⁻³, Faktor 3,94 bzw. 4,00 je Halbierung).
  Das ist ein Amplitudenfehler der diskreten Quelle (Sehnengeometrie der Winkelschichten), nicht des Lösers; er ginge quadratisch in die Kraft ein.
* **Gewählt: Renormierung nach der Projektion** (Skalar auf f je Spule). Begründung: (i) ein Skalar erhält Gᵀf = 0 (linear), (ii) das Strommuster ist durch die Geometrie festgelegt, der einzige freie Parameter ist die Amplitude, und
  die physikalisch vorgegebene Größe ist N·I; (iii) der Faktor ist 1 + O(h²), verändert die Konvergenzordnung also nicht, nur die Konstante; (iv) die Alternative (diskretes Quellfeld t_s mit curl t_s = J_s) braucht einen mehrwertigen
  Quellpotentialanteil für den Nettostrom und ist ein größerer Eingriff ohne Gewinn. Nicht gezeigt: dass die Renormierung die Konvergenzordnung des Feldes verbessert (am Feld in Schritt 3 gemessen: ohne Renormierung c_Meridian um 2,5·10⁻³ zu klein, das Feld um denselben Betrag).
* Tests (`test_trace_source`, host, 19 s): Gᵀf ≤ 1·10⁻¹², Nettostrom = N·I je Schicht ≤ 1·10⁻¹², gelöstes c_Meridian = N·I auf 5·10⁻⁷ (Schwelle 3·10⁻⁴); **Gegenprobe ohne Renormierung: Abweichung um den Faktor 2·10⁴ bis 5·10⁵ größer;
  ohne Projektion streut der Nettostrom (nu = 8: 10,8 %)**. Das gelöste c_Meridian kann nicht auf Lösertoleranz (10⁻¹²) gleich N·I sein: q_M ≠ 1 um 10⁻⁴ (BEM-Diskretisierung); die gemessene Abweichung 5·10⁻⁷ liegt darunter, weil sich die Schichtgleichungen mitteln.

### Schritt 1: TEAM-28-Aufbau (`trace_magnetostatics`)

σ = 0, μ = μ₀, Spulen: Innen (Komponente 2) N = 960, +, Außen (Komponente 1) N = 576, −, I = 20 A (Gleichstrom mit dem Betrag der Produktionsamplitude); Platte Komponente 3 als FEM-Körper ohne Wirkung. Rand: drei Komponenten, g = 1, 1, 0, nH = 4;
Innen: K mit Baum-Kobaum im ganzen Gebiet (eigener Kruskal-Spannwald, Seed 0 = aufsteigende Kantenordnung), keine Kohomologie-Eichung. Lösung: innere Kanten dünn eliminiert (SimplicialLDLT, in allen Läufen erfolgreich), Schurkomplement dicht direkt gelöst.

| Skalierung | Hex | FEM-Kanten | Randquads | n_innen | n_Γ (a + ψ + c) | S dicht | BEM-Operatoren | Elimination | Schur | Lösen | gesamt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0,5 | 696 | 3 086 | 936 | 806 | 1056 + 935 + 4 | 0,03 GB | 15,8 s | 0,1 s | 0,0 s | 1,0 s | 35 s |
| 0,75 | 2 034 | 8 300 | 2 106 | 2 765 | 2358 + 2105 + 4 | 0,16 GB | 45,4 s | 1,1 s | 0,5 s | 3,8 s | 104 s |
| 1,0 | 4 944 | 18 651 | 3 696 | 7 562 | 4176 + 3695 + 4 | 0,50 GB | 110,5 s | 8,2 s | 7,1 s | 17,0 s | 265 s |
| 1,25 | 9 780 | 35 224 | 5 730 | 15 917 | 6510 + 5729 + 4 | 1,20 GB | 238,4 s | 44,6 s | 22,9 s | 31,3 s | 585 s |

(Das Gesamtbild enthält die Referenz an den Elementmitten, 14 bis 197 s, und die Luftpunkte, 4 bis 17 s.) Plausibilität je Lauf (Skalierung 0,5 / 0,75 / 1,0 / 1,25):

| Größe | 0,5 | 0,75 | 1,0 | 1,25 |
|---|---|---|---|---|
| Residuum des skalierten Gesamtsystems | 7,7·10⁻¹⁵ | 3,4·10⁻¹⁴ | 6,2·10⁻¹⁴ | 1,15·10⁻¹³ |
| c_Meridian Außenspule, rel. Abweichung von N·I | 4,1·10⁻⁶ | 2,7·10⁻⁶ | 2,1·10⁻⁶ | 1,6·10⁻⁶ |
| c_Meridian Innenspule | 9,9·10⁻⁹ | 7,6·10⁻⁹ | 6,9·10⁻⁹ | 6,4·10⁻⁹ |
| c_Längs Außenspule (von N·I) | 6,9·10⁻⁵ | 4,8·10⁻⁵ | 3,7·10⁻⁵ | 3,0·10⁻⁵ |
| c_Längs Innenspule (von N·I) | 2,1·10⁻⁷ | 1,5·10⁻⁷ | 1,2·10⁻⁷ | 9,6·10⁻⁸ |
| Quelle: ‖Gᵀf‖/‖f‖ vor Projektion | 1,1·10⁻¹⁵ | 1,0·10⁻¹⁵ | 1,4·10⁻¹⁵ | 1,5·10⁻¹⁵ |
| Quelle: Nettostrom-Defizit vor Renormierung | 0,573 % | 0,254 % | 0,143 % | 0,091 % |

### Schritt 2: Feld gegen Volumenstrom-Biot-Savart

Referenz: C++-Funktion `include/TraceReference.h` (Port des Verfahrens aus `tools/theory/check_volume_biot_savart.py`); **gewählt gegen die Python-Nachauswertung**, weil Elementmitten und Luftpunkte im selben Treiber ohne Dateiumweg bedient werden
und die Referenz mit der Rechnung gemeinsam getestet wird. `test_trace_reference` (host): C++ gegen die Python-Werte an 16 Punkten (innen, Querschnitt, Rand, Luft, weit) 3,4·10⁻¹³; gegen eine unabhängige 120×120-Fadensumme 1,3·10⁻⁵; Gegenprobe Vorzeichen von B_ρ fällt durch.

**Feld in den FEM-Körpern** (curl a an den Elementmitten; proj = ⟨B,B_ref⟩/⟨B_ref,B_ref⟩, L2 = ‖B − B_ref‖/‖B_ref‖):

| Körper | Größe | 0,5 | 0,75 | 1,0 | 1,25 | Ordnung (0,5→1,25) | alt: Referenzpfad (0,5 / 1,0) | alt: BEM-FEM (0,5 / 1,0) |
|---|---|---|---|---|---|---|---|---|
| Platte | proj | 0,9747 | 0,9898 | 0,9945 | 0,9966 | | 0,972 / 0,994 | 1,004 / 1,094 |
| Platte | L2 | 0,0400 | 0,0159 | 0,0092 | 0,0063 | 2,02 | 0,056 / 0,014 | 0,220 / 0,205 |
| Innenspule | proj | 0,9861 | 0,9942 | 0,9969 | 0,9980 | | 0,906 / 1,026 | 0,540 / 0,877 |
| Innenspule | L2 | 0,0719 | 0,0308 | 0,0173 | 0,0110 | 2,05 | 0,219 / 0,103 | 0,596 / 0,378 |
| Außenspule | proj | 0,9239 | 0,9689 | 0,9831 | 0,9893 | | 0,869 / 0,999 | 0,478 / 0,789 |
| Außenspule | L2 | 0,1158 | 0,0521 | 0,0295 | 0,0190 | 1,97 | 0,190 / 0,040 | 0,704 / 0,478 |

**B_z in der Platte je Radiusbin** (Verhältnis gerechnet / Biot-Savart, Realteil; Spalten „alt“ aus `force-excess-localization.md`, dort gegen das Fadenmodell):

| Radius [m] | neu 0,5 | alt Ref 0,5 | alt BEM-FEM 0,5 | neu 0,75 | neu 1,0 | alt Ref 1,0 | alt BEM-FEM 1,0 | neu 1,25 |
|---|---|---|---|---|---|---|---|---|
| 0,00 bis 0,02 (Zentrum) | 0,9918 | 0,990 | 1,126 | 0,9953 | 0,9974 | 0,998 | **1,284** | 0,9984 |
| 0,02 bis 0,04 | 0,9360 | 0,901 | 1,061 | 0,9752 | 0,9864 | 0,974 | 1,287 | 0,9923 |
| 0,04 bis 0,06 (Rückfluss) | 1,0071 | 1,046 | 0,727 | 1,0045 | 1,0028 | 1,010 | **0,862** | 1,0020 |
| 0,06 bis 0,10 (Rand) | 0,9951 | 1,004 | 0,704 | 0,9978 | 0,9999 | 0,997 | **0,792** | 1,0011 |

**Luftraum über die Darstellungsformel (R-B)** aus λ = −Gψ + Hc und g = R a/|q| (relativer Fehler |B − B_ref|/|B_ref|; Auswertepunkte mit Gauß 6×6 je Teilquad, 6×6 Teilquads; Kontrolle mit 8×10 am nächsten Punkt: 9·10⁻⁹ bzw. 6·10⁻¹²):

| Punkt (Abstand zum Körper) | 0,5 | 0,75 | 1,0 | 1,25 |
|---|---|---|---|---|
| Platte oben, Achse, 1 / 5 / 20 mm | 1,1·10⁻² / 1,9·10⁻² / 5,0·10⁻² | 1,4·10⁻² / 9,1·10⁻³ / 2,2·10⁻² | 8·10⁻⁵ / 5,1·10⁻³ / 1,2·10⁻² | 4,2·10⁻³ / 3,3·10⁻³ / 7,7·10⁻³ |
| Platte Seite, 1 / 5 / 20 mm | 3,3·10⁻³ / 1,1·10⁻² / 2,0·10⁻² | 3,0·10⁻³ / 5,0·10⁻³ / 9,0·10⁻³ | 3,1·10⁻³ / 3,0·10⁻³ / 5,0·10⁻³ | 3,0·10⁻³ / 2,0·10⁻³ / 3,2·10⁻³ |
| Platte unten (ρ = 70 mm), 1 / 5 / 20 mm | 1,0·10⁻² / 5,9·10⁻³ / 3,4·10⁻³ | 4,7·10⁻³ / 3,1·10⁻³ / 3,3·10⁻³ | 2,8·10⁻³ / 1,9·10⁻³ / 2,1·10⁻³ | 1,9·10⁻³ / 1,3·10⁻³ / 1,4·10⁻³ |
| Innenspule außen, 1 / 5 / 20 mm | 3,6·10⁻² / 2,1·10⁻² / 2,0·10⁻² | 1,5·10⁻² / 7,6·10⁻³ / 7,3·10⁻³ | 8,6·10⁻³ / 4,4·10⁻³ / 4,2·10⁻³ | 5,3·10⁻³ / 2,8·10⁻³ / 2,6·10⁻³ |
| Innenspule innen (Loch), 1 / 5 / 20 mm | 3,9·10⁻² / 1,8·10⁻² / 1,0·10⁻² | 1,7·10⁻² / 5,8·10⁻³ / 4,2·10⁻³ | 9,6·10⁻³ / 3,3·10⁻³ / 2,4·10⁻³ | 5,8·10⁻³ / 2,0·10⁻³ / 1,5·10⁻³ |
| Außenspule außen, 1 / 5 / 20 mm | 6,9·10⁻² / 2,1·10⁻³ / 2,0·10⁻² | 3,1·10⁻² / 4,5·10⁻³ / 9,4·10⁻³ | 1,6·10⁻² / 2,8·10⁻³ / 5,3·10⁻³ | 9,6·10⁻³ / 2,1·10⁻³ / 3,4·10⁻³ |
| Unter den Spulen (ρ = 41 mm), 1 / 5 / 20 mm | **1,2·10⁻¹** / 3,9·10⁻² / 3,8·10⁻² | 3,0·10⁻² / 9,5·10⁻³ / 1,5·10⁻² | **3,4·10⁻²** / 4,5·10⁻³ / 8,0·10⁻³ | **2,2·10⁻²** / 4,4·10⁻³ / 5,0·10⁻³ |
| Achse z = −26 / +30 / +60 mm | 9,4·10⁻³ / 3,8·10⁻² / 3,0·10⁻² | 4,1·10⁻³ / 1,7·10⁻² / 1,3·10⁻² | 2,3·10⁻³ / 9,3·10⁻³ / 7,1·10⁻³ | 1,5·10⁻³ / 6,0·10⁻³ / 4,5·10⁻³ |
| größter Fehler | 1,15·10⁻¹ | 3,1·10⁻² | 3,4·10⁻² | 2,2·10⁻² |

Die Fehler fallen im Mittel mit Ordnung ≈ 2; Ausnahme ist der Punkt 1 mm unter der Spulenunterseite (nicht monoton: 1,2·10⁻¹ / 3,0·10⁻² / 3,4·10⁻² / 2,2·10⁻²). Bei Abständen unter einer Elementgröße (1 mm gegen etwa 3 bis 6 mm Randquads) ist
die Darstellungsformel mit stückweise konstantem g und Rooftop-λ im Nahfeld weniger genau.

### Schritt 3: Eichunabhängigkeit und Gegenproben

**Zwei andere Bäume** (Kruskal mit zufälliger Kantenordnung, Seeds 1 und 2, gegen Seed 0 = aufsteigend), `tools/trace_dump_compare.py`:

| Netz | Baum | |B₁−B₂|/|B| Platte / Innen / Außen | |A₁−A₂|/|A| Platte / Innen / Außen |
|---|---|---|---|
| 0,5 | Seed 1 | 2,7·10⁻⁸ / 2,6·10⁻⁸ / 5,3·10⁻⁸ | 7,8 / 1,2 / 1,7 |
| 0,5 | Seed 2 | 2,7·10⁻⁸ / 2,4·10⁻⁸ / 5,4·10⁻⁸ | 4,6 / 1,3 / 2,3 |
| 1,0 | Seed 1 | 3,5·10⁻⁹ / 1,1·10⁻⁸ / 2,5·10⁻⁸ | 5,4 / 1,4 / 2,5 |

B ist bis auf die Lösertoleranz gleich (Rückwärtsfehler 10⁻¹⁴, Kondition ~10⁸ gibt 10⁻⁸ in der Lösung), a ist verschieden (O(1)). Im Luftraum sind alle 24 Fehlerwerte bei Seed 1 und 2 auf drei Stellen gleich (Skalierung 0,5). `test_trace_system` zeigt dasselbe an der Kugel:
|B₁−B₂|/|B| = 2,3·10⁻¹⁴, |a₁−a₂|/|a| = 2,7. **Das ist das Kriterium 4 des Fahrplans für σ = 0: erfüllt.**

**Gegenproben** (Skalierung 0,75; Basis: L2 Platte / Innen / Außen 0,0159 / 0,0308 / 0,0521, größter Luftfehler 3,1·10⁻²):

| Gegenprobe | L2 Platte | L2 Innenspule | L2 Außenspule | Faktor schlechter | größter Luftfehler |
|---|---|---|---|---|---|
| Vorzeichen von λ in der FEM-Zeile umgekehrt | 8,44 | 0,825 | 1,39 | 530 / 27 / 27 | 28,4 |
| ohne harmonische Spalten (c weggelassen) | 0,395 | 3,69 | 3,46 | 25 / 120 / 66 | 5,7 |
| c_Meridian := 0 (beide Spulen) | 1,39 | 1,57 | 4,32 | 87 / 51 / 83 | 7·10⁴ |
| ohne Renormierung des Nettostroms | 0,0176 | 0,0313 | 0,0535 | 1,11 / 1,02 / 1,03 | 3,2·10⁻² |
| ohne Quellprojektion | 0,0159 | 0,0308 | 0,0521 | 1,00 | 3,1·10⁻² |

Ohne harmonische Spalten bzw. mit c_Meridian := 0 ist das System singulär (Residuum 1,3 bzw. 0,25; im zweiten Fall c_Längs = 10¹⁰ A), die Zahlen sind dann Werte einer gestörten, nicht eindeutig lösbaren Aufgabe; die Aussage ist, dass das gestörte System sichtbar
falsch ist, nicht der genaue Faktor. Die letzten beiden Zeilen sind keine Gegenproben im Sinn des Auftrags, sondern zeigen: (i) die Renormierung ist eine O(h²)-Korrektur der Amplitude (c_Meridian −11490,8 A statt −11520 A, Feld um 0,25 % kleiner), (ii) die Quellprojektion
ist auf den gefitteten TEAM-28-Netzen ohne Wirkung (Quelle schon divergenzfrei auf 10⁻¹⁵).

### Schritt 4: nicht bearbeitet

Der optionale harmonische Lauf (50 Hz, σ_Platte = 3,4·10⁷ S/m, F_z als Lorentz-Volumenkraft) ist nicht gemacht. Er verlangt komplexe Arithmetik im Löser (`SimplicialLDLT` ist für komplex-symmetrische Matrizen nicht geeignet, es wäre
`SparseLU`) und den Kraftkernel aus dem Bestand; beides ist nicht in diesem Auftrag enthalten.

## Interpretation (nicht geprüft)

* Der Fehler des alten Pfads (Überschuss im Zentrum, Defizit im Rückfluss) sitzt nach diesen Messungen in der Kopplung bzw. im Quellfeld, nicht im FEM-Teil oder in den Spulen: mit gleichem FEM-Aufbau (K, Spulenstrom, Netz) und der neuen Kopplung verschwindet er. Das stützt die Hypothese
  aus `bem-trace-consistency.md` (K-Widerspruch an den Spulen), beweist aber nicht, welcher der drei Unterschiede (Spurmodell, harmonische Spalten, Quelle) ihn im alten Pfad ausgelöst hat.
* Dass die L2-Fehler mit Ordnung ≈ 2 fallen, obwohl B = curl a bei niedrigster Nédélec-Ordnung nur O(h) im Volumen konvergieren sollte, hängt vermutlich an der Auswertung an den Elementmitten (Superkonvergenz auf strukturierten Netzen); nicht untersucht. Die Kugel in Phase 1b
  (verzerrtes Cube-to-Ball-Netz) zeigte nur O(h).
* Die Nichtmonotonie des Luftfehlers 1 mm unter der Spulenunterseite ist nicht untersucht (Nahfeld der Darstellungsformel, Quadraturordnung ist nach der Kontrolle nicht die Ursache: 6·10⁻¹² beim Übergang zu höherer Ordnung an einem anderen Punkt).
* Dass auch ohne Renormierung das Feld fast gleich bleibt (0,2 bis 0,5 %), heißt nicht, dass die Renormierung bedeutungslos ist: die Kraft geht quadratisch ein (0,5 % bei Skalierung 0,75, 1,1 % bei 0,5), was bei Phase 3 relevant sein kann.
* Aus den Läufen folgt nichts über den transienten Fall oder über σ > 0; das ist Phase 3.

## Abweichungen vom Auftrag

1. **Schritt 4 (optional) nicht bearbeitet** (siehe oben).
2. **Zwei Commits für 0a und 0b, ein gemeinsamer Commit für Schritt 1 und 2** (derselbe Treiber), Schritt 3 ohne eigenen Code-Commit (Läufe und Bericht).
3. **Schritt 0a:** Die Konditionsmessung am gröbsten TEAM-28-Netz ist an einem Netz gemessen (`hex_s0.5`), nicht am Standardnetz; die Phase-1b-Zahlen wurden wiederholt (unskaliert ∞ statt der damals nur qualitativ genannten „numerisch singulär“).
4. **Quellkonstruktion nach 0b:** Renormierung, nicht das diskrete Quellfeld t_s; Begründung oben.
5. **Eigene Baumeichung (Kruskal) statt `build_tree_cotree_gauge`** im TEAM-28-Treiber, damit der Seed die Kantenreihenfolge ändern kann; `build_tree_cotree_gauge` bleibt für die Phase-1b-Tests und wurde im Treiber nicht benutzt. Gleichwertig geprüft ist nur durch die Eichinvarianz selbst (zwei Seeds).
6. **Strom 20 A als Gleichstrom** mit dem Betrag der Produktionsamplitude (I_max aus `fem.cu`); der Vergleich mit den alten Zahlen (bei 0,01 Hz, relative Größen) ist davon unabhängig.
7. **Referenzvergleich gegen den Volumenstrom, die alten Tabellen gegen das Fadenmodell** (siehe Kopf).
8. **`compare_fields.py` nicht benutzt:** die Metriken (proj, L2 über alle Komponenten und Elemente je Körper, B_z-Radiusbins) sind im Treiber nachgebildet; der Treiber schreibt aber einen kompatiblen Dump (`--dump`), den `compare_fields.py` lesen kann. Nicht gelaufen.
9. **Platte bei 11,3 mm durch Verschieben der Plattenknoten** (Datei: 3,8 mm); in der Magnetostatik hat die Platte keine Wirkung, die Auswerteorte entsprechen den alten Tabellen.

## Offene Punkte

1. CI auf `main` rot durch apt-Zeitüberschreitung im Job „Host-Tests“; die Trace-Tests sind in der CI noch nie gelaufen. Behebung im Workflow (z. B. längere Zeitgrenze oder Pakete aus einem Cache) ist nicht Teil dieses Auftrags.
2. Schritt 4 und Phase 3 (σ > 0, transient, Kraft).
3. Der Luftfehler im Nahfeld (< 1 Elementgröße) hängt von der Auswertung der Darstellungsformel ab; für die Maxwell-Spannung auf einer Fläche nahe der Platte (Phase 3) beachten.
4. `project_out_gradient` bei getrennten Körpern nur mit Maske (Phase 1b); auf den TEAM-28-Netzen ohne Wirkung, weil die Quelle schon divergenzfrei ist.
5. Rechenzeit: Skalierung 1,25 dicht 585 s und 1,2 GB für S; Skalierung 1,5 (15 984 Hexaeder) wäre dicht noch machbar, ist hier nicht gerechnet. ACA, GPU: nicht Teil.

## Befehle

```
cd build
cmake .. && make trace_magnetostatics test_trace_system test_trace_source test_trace_reference trace_conditioning
./trace_magnetostatics near/hex_s{0.5,0.75,1.0,1.25}.vtk --dump=trace/ms_s<S>.csv       # Schritt 1 und 2
./trace_magnetostatics near/hex_s0.5.vtk --seed=1 --dump=trace/ms_s0.5_seed1.csv         # Schritt 3, Baum
python3 ../tools/trace_dump_compare.py trace/ms_s0.5.csv trace/ms_s0.5_seed1.csv
./trace_magnetostatics near/hex_s0.75.vtk --flip-lambda | --no-harmonic | --cm0 | --no-renorm | --no-project
./trace_conditioning team28 near/hex_s0.5.vtk --scaling                                  # 0a
ctest -L host ; ctest      # mit LD_LIBRARY_PATH auf die CUDA-Bibliotheken
```

## Tests

`ctest -L host`: vorher 49/49, nachher 51/51 (neu: `test_trace_reference`, `test_trace_source`; `test_trace_system` um 0a und den Schur-Test erweitert). Volles `ctest`: vorher 82/82, nachher 84/84; alle bestanden, mit `LD_LIBRARY_PATH` auf die CUDA-Bibliotheken.
Laufzeiten im vollen Lauf: `test_trace_system` 63 s, `test_trace_torus` 60 s, `test_trace_ops` 36 s, `test_trace_quadrature` 32 s, `test_trace_source` 22 s, `test_trace_reference` 1,2 s (alle unter 120 s). Die großen TEAM-28-Läufe sind Messtreiber (`trace_magnetostatics`, `trace_conditioning`), keine Tests.
