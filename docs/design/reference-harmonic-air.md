# BEM-Kopplung am harmonischen Knoten-Referenzlöser mit Luftgebiet

Stand: 04.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB),
Linux (nicht RTX 3090 / Windows wie im Auftrag). Programm aus Zweig
`reference-harmonic-air`, Stand der Läufe `b0f9eb7` (die danach folgenden
Änderungen betreffen nur Bericht, CHANGELOG und Hilfetext). Netze aus
`tools/gmsh/team28_hex_air.py`, Läufe und Tabellen aus
`tools/reference_series.py --groups F`. Logs, Netze und Dumps in
`build/refair/` (nicht im Repo; `summary.md` und `results.json` dort sind
die Quelle aller Zahlen unten, der erste vollständige Lauf liegt in
`build/refair_erstlauf/`).

## Frage

Der harmonische Referenzlöser (`--reference --harmonic --fixed-plate`)
koppelt die Knoten-FEM über eine dichte BEM an den Außenraum. Auf den
Netzen ohne Luft liegt die Kopplungsfläche auf den Leiteroberflächen, mit
Spalten von nur 0,6 bis 3 Panelbreiten zwischen Spule und Platte. Wo die
Kopplung versagt, ist offen. Der Auftrag: die Kopplungsfläche schrittweise
von einem fernen Rand eines mitvernetzten Luftgebiets auf die
Leiteroberflächen verschieben und dabei **dasselbe Netz** verwenden (Luft
nur löschen, nie neu vernetzen). Misst man eine Änderung, wird sie der
Annäherung der Fläche zugeschrieben, nicht der Diskretisierung.

## Ergebnis in Kürze

**Gemessen** (Zahlen und Läufe in den Abschnitten darunter)

1. Mit Dirichlet-Rand (Stufe D) hängt F_z stark von der Kastengröße ab und
   ist nicht monoton (+28 % gegen B0 bei Faktor 3, +47 % bei Faktor 2,
   +2,1 % bei 5×5×10). Die BEM am Außenrand (B0) ist dagegen fast
   unabhängig von der Kastengröße: 0,98301 bis 0,98623 N, **Streuung
   0,326 %**. Das liegt über dem Vorschlag 0,1 % und unter der
   Abbruchgrenze 0,5 %.
2. Die Feldprüfung auf der Achse (`--verify-field`, σ = 0) gibt für B0 eine
   größte Abweichung von 0,15 % gegen Biot-Savart, für D 9,3 %.
3. Verschiebt man die Kopplungsfläche nach innen, driftet F_z gleichmäßig
   weg, **ohne einzelnen Sprung**: B1 −0,007 %, B2 −0,134 %, B3 −0,451 %,
   B_S −0,759 % gegen B0. Das 0,1-%-Kriterium des Auftrags ist ab B2
   überschritten. B_rille, B_kern, B_platte liegen bei −0,73 bis −0,78 %,
   B_end bei −0,604 %.
4. Der Drift wird mit feinerem Netz kleiner: B_S gegen B0 beträgt −2,557 %
   (Skalierung 0,5), −0,759 % (1,0), −0,393 % (1,5).
5. Bei 3,8 mm: B_S −0,431 %, B_platte −0,393 %, B_end −0,966 %.
6. Alle Läufe der Gruppe F konvergieren in
   höchstens 11 äußeren FGMRES-Iterationen, innere Iterationen höchstens
   920 (Grenze 600 je Aufruf, keine Überschreitung).
7. Gegenprobe des unveränderten Pfads: `--reference --harmonic --fixed-plate
   --plate-z=11.3` auf `gmsh_team28_hex_coarse.vtk`: F_z = 0,853334 N,
   9 äußere Iterationen (vorher 0,85333 N, 9/288). Gleich auf alle ausgegebenen Stellen.

**Nicht gemessen, nur gedeutet** (so im Text gekennzeichnet)

- Der Drift kommt von der Diskretisierung der BEM nahe den Flächen (gemittelte
  Normalen an Kanten), nicht von einem Fehler der Formulierung.
- Die Abweichung zu B_end ist dann dieselbe Fehlerklasse, nur größer.

**Abbruchkriterien:** keines der sechs ausgelöst. Die B0-Streuung von
0,326 % liegt über der Vorgabe 0,1 %, daher hier gemeldet.

## Methode

**Netz (AP2).** `tools/gmsh/team28_hex_air.py`: strukturiertes
Rotationsgitter, Kernquadrat n_q × n_q (Ecken bei 0,8·27 mm), Ringe bis
r = 27 mm, Zonen bis 55 / 65 / 80 / 95 mm, danach geometrisch wachsende
Luftschichten bis zur Kastengröße (Wachstum ≤ 1,35). Komponenten 1
Außenspule, 2 Innenspule, 3 Platte, 0 Luft. Leitervolumina exakt wie das
N-Eck (N = 4·n_q), Prüfung auf 1e-15 relativ. Platte als volle Scheibe.
Jede Luftzelle trägt genau eine Zone: 1 bis S Schalen (1 innen, S außen;
Abstand max(radial, axial) auf S = 4 Gruppen verteilt), 10 Spalt,
11 Kern (r < 27 mm, z ∈ [−52, 0] mm), 12 Rille (55 bis 80 mm),
13 Plattenseite (65 bis 95 mm auf Plattenhöhe). Die Plattenhöhe setzt der
Generator (`--plate-z0`); `--plate-z` ist mit Luftnetzen abgewiesen.
Abschneidefaktoren: Luftausdehnung / Leiterausdehnung je Achse wie in
`FemDirectRun`.

Die z-Kanten werden von den Leiterrändern nach außen aufgebaut. Dadurch
sind Leiter- und Spaltknoten für alle Abschneidefaktoren bitgleich
(Prüfung `f_check_conductor_nodes`, Ergebnis: ja). Eine erste Fassung
mit kumulierten Gleitkommasummen verfehlte das; der Lauf wurde nach der
Korrektur komplett wiederholt.

**Stufen (AP3).** `tools/strip_air.py NETZ --remove-zones LISTE -o AUS`
löscht Zellen, behält die Knotenreihenfolge der übrigen und prüft die neue
Oberfläche (`tools/meshcheck.py`: Randkanten in genau zwei Flächen mit
entgegengesetzter Richtung, Eckenfächer, Flächenvektorsumme je Komponente,
Volumenvorzeichen, Euler-Charakteristik, Hohlraumprüfung gegen das
Ausgangsnetz). Alle Stufen entstehen aus dem vollen Netz, die Kette ist
kumulativ: B1 = Zone 1 entfernt, B2 = 1–2, B3 = 1–3, B_S = 1–4,
B_rille = 1–4 und 12, B_kern = 1–4, 11, 12, B_platte = 1–4, 11, 12, 13,
B_end = alle Luft (1–4, 10–13).

**Dirichlet-Stufe (AP4).** `--reference-dirichlet`: A = 0 an allen
Randknoten; Zeilen **und** Spalten werden eliminiert, das Muster bleibt,
K und W bekommen dieselbe Diagonale (Mittel der K-Diagonale über die freien
Zeilen), damit C = (W − K)/ω in den Dirichlet-Zeilen verschwindet. Die BEM
wird nicht aufgebaut (`BEM nicht aufgebaut`). Nur mit
`--reference --harmonic`, sonst Exit 1.

**A-Export (AP5).** `--reference-dump-a=DATEI` schreibt A_r, A_i an den
Leiterknoten; `tools/compare_a.py` vergleicht je Körper (ΔA relativ, Phase,
größte Knotenabweichung mit Ort). Die Läufe der Gruppe F vergleichen gegen
B0 der jeweiligen Netzfamilie.

**Unverändert:** Formulierung, `NodalBem.cuh`, Strafterm, Vorkonditionierer.
Die Gegenprobe oben stimmt auf alle ausgegebenen Stellen überein.

## Tests (AP6)

| Test | Ergebnis |
|---|---|
| `test_reference_harmonic` [6] Dirichlet-Elimination (n = 40, 7 Dirichlet-Knoten) | FGMRES gegen dichte Direktlösung des reduzierten Systems 1,5e-13, 25 Iterationen; Gegenproben: nur Zeilen eliminiert → Asymmetrie 1,0 wird gemeldet, ohne Elimination weicht die Lösung um 0,11 ab, fehlende Diagonale wirft |
| `test_reference_solver` [6] nahe Flächen (flacher Quader, d/h = 4 bis 0,5) | Residuum der Innengleichung mit u = x: 0,114 / 0,183 / 0,300 / 0,501 für d/h = 4 / 2 / 1 / 0,5; mit u = z: 0,073 / 0,061 / 0,038 / 0,026. Nahfeldquadratur automatisch gleich fein (3e-11 bzw. 2e-8); Gegenprobe mit grober Quadratur weicht bei d/h = 0,2 um 100 % ab; falsches Vorzeichen von ½I: Residuum 1,44 |
| `tools_team28_hex_air_tests` | Maße, Zonen, Jacobi-Determinante, bitgleiche Leiterknoten über Abschneidefaktoren |
| `tools_strip_air_tests` | Löschen, Oberflächenprüfungen mit Gegenproben |
| `tools_compare_a_tests` | synthetische Felder, Gegenproben |
| `reference_series_air_smoke` | Kettenrauchtest (kleinste Skalierung) |
| `ctest -L host` | 35 von 35 grün (Lauf vom 04.11.2026) |
| `ctest -L tools` | 7 von 7 grün |

(In dieser Umgebung braucht ctest `LD_LIBRARY_PATH` auf cudart 12.9 und
cublas 12.9 des HPC-SDK.) Die Schwellen des Nahflächentests sind gemessen
und nicht streng physikalisch begründet; **gemessen:** das Residuum für
u = x wächst mit abnehmendem d/h, für u = z fällt es.
*Interpretation:* Der Anstieg bei u = x kommt von den gemittelten Normalen
an den Kanten (stetige Neumann-Spur, siehe `ReferenceSolver.cuh`), nicht von
der Quadratur, denn diese ist schon konvergiert (zweite Zeile des Tests).

## Netze

Alle Netze bestehen meshcheck und check_map (0 nicht positive Ecken, kleinste
Jacobi-Determinante absolut 4,86; skaliert: Leiter 0,998 bei Skalierung 1,
0,707 für Platte und Luft). Das Minimum 0,707 liegt in den 45°-Zellen am
Kernquadrat; die Schwelle 0,5 wird eingehalten, die Leiterteilung musste
nicht angepasst werden (Abbruchkriterium 5 nicht ausgelöst).

| Netz | Hexaeder | Zellen je Komponente | Zellen je Zone | Abschneidefaktoren | Randknoten voll |
|---|---|---|---|---|---:|
| air_s1_t3 | 47 520 | Außenspule 1536, Innenspule 2688, Platte 2160, Luft 41136 | 1: 9504, 2: 9840, 3: 8208, 4: 4512, 10: 3312, 11: 1920, 12: 2688, 13: 1152 | 3 / 3 / 3 | 4610 |
| air_s1_t1p5 | 29 568 | wie oben, Luft 23184 | 1: 6144, 2: 2112, 3: 4512, 4: 1344, 10–13 wie oben | 1,5 / 1,5 / 1,5 | 3746 |
| air_s1_t2 | 36 000 | Luft 29616 | 1: 6144, 2: 5616, 3: 4800, 4: 3984 | 2 / 2 / 2 | 4082 |
| air_s1_t5 | 57 120 | Luft 50736 | 1: 13056, 2: 11808, 3: 8784, 4: 8016 | 5 / 5 / 5 | 4994 |
| air_s1_t5x10 | 65 520 | Luft 59136 | 1: 13056, 2: 16128, 3: 14160, 4: 6720 | 5 / 5 / 10 | 5234 |
| air_s1_t3_z3p8 | 44 352 | Luft 37968 | 1: 9504, 2: 5952, 3: 7776, 4: 5664, 10–13 wie oben | 3 / 3 / 3 | 4514 |
| air_s1p5_t3 | 124 488 | Außenspule 5184, Innenspule 9504, Platte 6480, Luft 103320 | 1: 20520, 2: 23112, 3: 16308, 4: 14868, 10: 9936, 11: 6480, 12: 8640, 13: 3456 | 3 / 3 / 3 | 9290 |
| air_s0p5_t3 | 9 324 | Außenspule 192, Innenspule 384, Platte 360, Luft 8388 | 1: 2736, 2: 1512, 3: 1704, 4: 888, 10: 828, 11: 240, 12: 288, 13: 192 | 3 / 3 / 3 | 1394 |

Kleinste skalierte Jacobi-Determinante (Leiter / Platte und Luft):
0,998 / 0,707 (air_s1_*), 0,999 / 0,707 (air_s1p5), 0,991 / 0,707 (air_s0p5).
Die Gesamtzahlen der Hexaeder sind Summen der Spalten Komponente. Dichte
Algebra 0,03 bis 2,06 GiB, weit unter der Grenze 48 GiB; kein Lauf wegen
Speichers abgelehnt (Abbruchkriterium 4 nicht ausgelöst).

Randknoten und Hexaeder je Stufe auf `air_s1_t3` (alle Kopplungsflächen
bestehen die Randprüfung; Euler-Charakteristik 2 = eine Kugelfläche):

| Stufe | Hexaeder | n (FEM) | Randknoten | Randkomponenten (Euler) |
|---|---:|---:|---:|---|
| B0 (und D) | 47 520 | 149 637 | 4610 | 1 (2) |
| B1 | 38 016 | 135 807 | 4418 | 1 (2) |
| B2 | 28 176 | 110 526 | 3986 | 1 (2) |
| B3 | 19 968 | 80 199 | 3458 | 1 (2) |
| B_S | 15 456 | 50 805 | 2882 | 1 (2) |
| B_rille | 12 768 | 43 893 | 3650 | 1 (2) |
| B_kern | 10 848 | 38 685 | 4034 | 1 (2) |
| B_platte | 9 696 | 35 229 | 4034 | 1 (2) |
| B_end | 6 384 | 25 788 | 4178 | 3 (0, 0, 2) |

B_end besteht aus zwei Tori (Spulen, χ = 0) und der Platte (χ = 2) und
besteht `check_separate_bodies`.

## Abschneidestudie D / B0 (AP7), 11,3 mm

F_z in N (Ordnung 2 und 3 gleich in allen Läufen auf alle Stellen).

| Netz | Faktoren | B0 | D | D gegen B0(t3) | B0 gegen B0(t3) |
|---|---|---:|---:|---:|---:|
| air_s1_t1p5 | 1,5 | 0,98301 | 1,26883 | +28,87 % | −0,158 % |
| air_s1_t2 | 2 | 0,98411 | 1,45105 | +47,38 % | −0,047 % |
| air_s1_t3 | 3 | 0,98457 | 1,26197 | +28,17 % | 0 |
| air_s1_t5 | 5 | 0,98577 | 1,06880 | +8,55 % | +0,122 % |
| air_s1_t5x10 | 5/5/10 | 0,98623 | 1,00494 | +2,07 % | +0,168 % |

**Gemessen:** B0 steigt monoton mit der Kastengröße, Spanne 0,326 %. D ist
nicht monoton. Bei D ändern sich Q_eff (0,36 bis 0,73) und die
Verlustleistung (13,1 bis 19,8 W) stark; bei B0 liegt Q_eff bei 0,746 bis
0,751 und die Leistung bei 12,98 bis 12,99 W.
*Interpretation:* Die BEM am fernen Rand ist die belastbare Referenz der
Kette; der Dirichlet-Rand verfälscht das Feld bei diesen Kastengrößen
(vermutlich über die Rückwirkung der Wirbelströme auf das Streufeld).
Nicht untersucht.

**Feldprüfung s = 0** (`--verify-field`, `air_s1_t3`, B_z auf der Achse gegen
Biot-Savart): B0 größte Abweichung 0,15 %, mittlere
+0,127 %, Abklingverhältnis oben/unten 0,8947 gegen 0,8952 analytisch
(−0,05 %). D: größte Abweichung 9,32 %, mittlere +8,83 %. Die Vorgabe
0,2 % ist für B0 erfüllt.

## Stufenkette (AP8)

### 11,3 mm, `air_s1_t3`

Spalten: F_z, Abweichung gegen B0, ΔA je Körper (relative L2-Norm gegen B0,
Real- und Imaginärteil), Phase der Platte.

| Stufe | F_z [N] | gegen B0 | Platte ΔA_r / ΔA_i | Innenspule ΔA_r / ΔA_i | Außenspule ΔA_r / ΔA_i | Phase Platte |
|---|---:|---:|---|---|---|---:|
| B0 | 0,98457 | 0 | – | – | – | – |
| B1 | 0,98450 | −0,007 % | 1,3e-4 / 9,0e-5 | 2,5e-5 / 1,1e-4 | 4,5e-5 / 4,0e-4 | −0,001° |
| B2 | 0,98325 | −0,134 % | 1,2e-3 / 1,4e-3 | 2,7e-4 / 1,8e-3 | 5,0e-4 / 5,2e-3 | −0,002° |
| B3 | 0,98013 | −0,451 % | 3,1e-3 / 4,5e-3 | 5,2e-4 / 5,9e-3 | 1,1e-3 / 1,4e-2 | +0,003° |
| B_S | 0,97710 | −0,759 % | 4,7e-3 / 7,1e-3 | 1,6e-3 / 9,1e-3 | 2,5e-3 / 2,1e-2 | +0,016° |
| B_rille | 0,97734 | −0,734 % | 4,2e-3 / 5,9e-3 | 2,4e-3 / 6,3e-3 | 3,8e-3 / 1,8e-2 | +0,014° |
| B_kern | 0,97693 | −0,777 % | 4,4e-3 / 6,2e-3 | 2,4e-3 / 6,1e-3 | 3,8e-3 / 1,9e-2 | +0,008° |
| B_platte | 0,97710 | −0,760 % | 4,7e-3 / 6,6e-3 | 2,4e-3 / 6,1e-3 | 3,8e-3 / 1,9e-2 | +0,005° |
| B_end | 0,97863 | −0,604 % | 4,9e-3 / 4,6e-3 | 2,5e-3 / 4,0e-3 | 3,7e-3 / 1,2e-2 | −0,025° |

Größte Knotenabweichung der Platte: 1,7e-4 (B1) bis 9,7e-3 (B_end), Ort
steht je Lauf in `summary.md` (meist am Plattenrand r = 65 mm, bei B_end an
der Ecke (−46, 46, 11,3) mm).

Zeit je Stufe: 28 s (B3) bis 64 s (B0); Abfall mit dem FEM-Umfang.

### 3,8 mm, `air_s1_t3_z3p8`

| Stufe | F_z [N] | gegen B0 | Platte ΔA_r / ΔA_i | Innenspule ΔA_r / ΔA_i | Außenspule ΔA_r / ΔA_i |
|---|---:|---:|---|---|---|
| B0 | 3,24353 | 0 | – | – | – |
| B_S | 3,22955 | −0,431 % | 3,7e-3 / 4,5e-3 | 1,8e-3 / 4,6e-3 | 2,6e-3 / 7,8e-3 |
| B_platte | 3,23077 | −0,393 % | 4,0e-3 / 3,9e-3 | 2,5e-3 / 2,3e-3 | 4,0e-3 / 7,3e-3 |
| B_end | 3,21221 | −0,966 % | 4,8e-3 / 4,9e-3 | 2,6e-3 / 4,3e-3 | 3,7e-3 / 9,2e-3 |

### Skalierung (Netzfeinheit der Kette)

| Skalierung | Randknoten B0 | B0 [N] | B2 | B3 | B_S |
|---|---:|---:|---:|---:|---:|
| 0,5 | 1394 | 0,94140 | −0,356 % | −1,013 % | −2,557 % |
| 1,0 | 4610 | 0,98457 | −0,134 % | −0,451 % | −0,759 % |
| 1,5 | 9290 | 0,99692 | −0,147 % | −0,291 % | −0,393 % |

**Gemessen:** B_S gegen B0 fällt von −2,56 über −0,76 auf −0,39 %. Aus den
drei Punkten folgt rechnerisch eine Ordnung von etwa 1,6 (1,0 → 1,5) bis
1,75 (0,5 → 1,0). B2 ändert sich zwischen den Skalierungen 1,0 und 1,5
nicht (−0,134 / −0,147 %), B3 und B_S schon.

### Nahflächentest und Abstand

d/h im Spalt zwischen Spule und Platte (Abstand zu Panelbreite √A der
Spulenoberseite 3,9 bis 6,8 mm): bei 3,8 mm etwa 0,6 bis 1,0, bei 11,3 mm
etwa 1,7 bis 2,9. Der Nahflächentest (oben) zeigt an einer ebenen
Doppelfläche bei d/h = 1 ein Innengleichungsresiduum von 0,30 (u = x) bzw.
0,038 (u = z), bei d/h = 2: 0,18 / 0,061.

## Einordnung

Zahlen gegen B0 und die Referenzen (Ringmodell 1,00684 N bei 11,3 mm,
3,28965 N bei 3,8 mm; A_s1 der Netzserie = 0,96649 N auf
anderer Netzfamilie):

| Größe | Wert | gegen Ringmodell | gegen B0 |
|---|---:|---:|---:|
| B0, 11,3 mm | 0,98457 N | −2,21 % | – |
| B_end, 11,3 mm | 0,97863 N | −2,80 % | −0,604 % |
| B_end gegen A_s1 (anderes Netz, keine Luft) | +1,256 % | | |
| B0, 3,8 mm | 3,24353 N | −1,40 % | – |
| B_end, 3,8 mm | 3,21221 N | −2,35 % | −0,966 % |

**Gemessen:**

- B0 und B_end unterscheiden sich um 0,604 % (11,3 mm) und 0,966 %
  (3,8 mm). Das überschreitet die 0,1 % des Auftrags.
- B_end liegt um 1,26 % über A_s1. Beide rechnen die Konfiguration
  „Leiter allein“, aber auf Netzen verschiedener Familie (A_s1: andere
  Teilungen, andere Plattenschichten; B_end: Spulenteilung wie
  `air_s1`). Der Unterschied ist damit nicht der Annäherung der Fläche
  zuzuschreiben; ein Vergleich auf dem Netz von A_s1 wurde nicht gerechnet.
- Der Verlauf B_S → B_rille → B_kern → B_platte ist flach (−0,76 ± 0,02 %);
  die Annäherung an einzelne Leiter erzeugt keinen Sprung, erst der letzte
  Schritt (Spaltluft entfernt, B_platte → B_end) bringt +0,16 % bei 11,3 mm
  und −0,57 % bei 3,8 mm, also mit wechselndem Vorzeichen.

*Interpretation (nicht bewiesen):*

- Der Drift B1 → B_S gleicht der Wirkung einer BEM mit näher rückender
  Fläche; sein Verschwinden mit feinerem Netz (Ordnung etwa 1,6 bis 1,75)
  spricht für einen Diskretisierungsfehler der Kopplung, nicht für ein
  Versagen der Formulierung.
- Das wechselnde Vorzeichen des letzten Schritts zwischen 11,3 und 3,8 mm
  deutet auf zwei konkurrierende Anteile (Spaltabstand und Kantenfehler).
  Mit zwei Höhen ist das nicht trennbar.
- Dass B0 mit der Kastengröße fast konstant ist (0,33 %) und D nicht,
  spricht dafür, dass die Streuung der Kette größtenteils von der
  Diskretisierung der Kopplung kommt und nicht vom Luftgebiet.

## Laufzeiten

Alle Läufe einzeln auf der RTX 6000 Ada, nacheinander.

| Lauf | Zeit |
|---|---:|
| B0 `air_s1_t3` (149 637 Zeilen, 4610 Randknoten) | 64 s |
| D `air_s1_t3` | 37 s |
| B_S | 17 s |
| B_end | 13 s |
| `air_s1p5_t3` B0 (387 621 Zeilen, 9290 Randknoten, 2,06 GiB) | 183 s |
| `air_s1p5_t3` B2 | 217 s |
| `air_s0p5_t3` B0 | 9 s |

## Offene Punkte

1. B0-Streuung 0,326 % statt der gewünschten 0,1 % (Abbruchgrenze 0,5 %
   nicht erreicht). Ob sie mit der Kastengröße gegen null geht, zeigt der
   Faktor 5×5×10 (+0,168 % gegen t3) nicht eindeutig; der Anstieg wird
   bei großen Faktoren flacher (0,122 → 0,168 %).
2. Der Drift der Kette überschreitet 0,1 % ab B2/B3 und sinkt mit der
   Netzfeinheit; die Ordnung folgt nur aus drei Punkten. Eine
   vierte Skalierung (2,0) würde sie absichern.
3. D ist nicht monoton über der Kastengröße; Ursache nicht untersucht.
4. Das Vorzeichen des letzten Kettenschritts wechselt zwischen 11,3 und
   3,8 mm; ungeklärt.
5. Kein Vergleich B_end gegen A_s1 auf gleichem Netz (die Netzfamilien
   unterscheiden sich).
6. Es gab keine Gegenprobe gegen `NodalBem.cuh` mit analytischer
   Referenz an den Kanten; **nicht angefasst** nach Vorgabe.

**Abweichungen vom Auftrag**

- Auftrag: „Exit-Code ... propagieren“ in `reference_series.py`; er wurde
  schon propagiert, nur ergänzt (AP1).
- Auf Luftnetzen scheiterte `find_cohomology_generators` an Komponente 0.
  Für den Referenzlöser wird die Suche übersprungen, sobald das Netz
  Komponente 0 enthält (`src/fem.cu`, Variable
  `skip_cohomology_for_reference`). Der Referenzlöser braucht sie nicht.
- Maschine RTX 6000 Ada / Linux; Programm `build/fembem` statt
  `Release/fembem`.
- Der Auftrag nennt eine Option `--h-gap`, die in der Optionsliste fehlt;
  sie wurde nicht umgesetzt (Spalt und Platte per `--gap-nz`, `--plate-nz`).
- Axialer Sprungfaktor Spalt gegen Platte 3,77 bei 11,3 mm (der Auftrag
  fordert die Grenze 2 nur radial); radial 1,48. Als Hinweis geloggt.
- Bei Skalierung 0,5 war der radiale Sprungfaktor 2,59 nicht erreichbar;
  `n_ring` wird aus dem zulässigen Bereich gewählt (mindestens 1 Ring).
- Die Gegenprobe zeigt Rauschen der Größe 1e-14 im Moment zwischen
  Läufen; F_z gleich auf alle ausgegebenen Stellen.

## Befehle

Aus `build/` (die Netze liegen in `build/refair/`):

```
python3 ../tools/gmsh/team28_hex_air.py -o refair/air_s1_t3.vtk --scale 1 \
    --plate-z0 11.3 --truncation-radial 3 --truncation-axial 3
python3 ../tools/strip_air.py refair/air_s1_t3.vtk --remove-zones 1-4 \
    -o refair/air_s1_t3_B_S.vtk
./fembem --reference --harmonic --fixed-plate --mesh=refair/air_s1_t3.vtk \
    --reference-dump-a=refair/air_s1_t3_B0.csv                      # B0
./fembem --reference --harmonic --fixed-plate --reference-dirichlet \
    --mesh=refair/air_s1_t3.vtk                                      # D
./fembem --reference --harmonic --fixed-plate --verify-field \
    --mesh=refair/air_s1_t3.vtk                                      # Feldprüfung
python3 ../tools/compare_a.py refair/air_s1_t3_B_S.csv \
    refair/air_s1_t3_B0.csv --mesh refair/air_s1_t3_B_S.vtk
python3 -u ../tools/reference_series.py --groups F \
    --f-extra 1.5:11.3:B2,B3,B_S --f-extra 0.5:11.3:B2,B3,B_S \
    > refair/groupF.out                                              # Gruppe F
./fembem --reference --harmonic --fixed-plate --plate-z=11.3 \
    --mesh=refair/gmsh_team28_hex_coarse.vtk                        # Gegenprobe
```

`--plate-z` darf mit Luftnetzen nicht verwendet werden (Fehlermeldung),
die Höhe setzt der Generator.
