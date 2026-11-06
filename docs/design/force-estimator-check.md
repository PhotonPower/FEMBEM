# Kraftvergleich mit gleichem Schätzer, Einbruch bei Skalierung 1,25

Stand: 06.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), Linux. Basis: `main`
482546a (nach #23 und #24), Zweig `force-estimator-check`. Netze `near/hex_sS.vtk` aus
`tools/gmsh/team28_hex.py`, Platte bei 11,3 mm, 50 Hz (BEM-FEM mit `--harmonic-inner-direct`,
Referenzpfad `--reference --harmonic=50`), Zeilenmessung `--bem-row-check` bei 2 Hz. Logs in
`build/near/` (aus #24, nicht neu gerechnet) und `build/force/` (neu), nicht im Repo.

## Ergebnis in Kürze

**Gemessen**

1. **Die beiden Lorentz-Kernel sind vergleichbar.** Gleiche Formel, gleiche Elementmenge (nur die
   Platte), auf demselben Phasorfeld stimmen sie auf Rundungsniveau überein (A1).
2. **Das F_z-Defizit gibt es mit dem Lorentz-Schätzer nicht; es ist ein Überschuss.** Lorentz BEM-FEM /
   Lorentz Referenz = 1,75 / 1,78 / 1,78 / 1,98 für die Skalierungen 0,5 / 0,75 / 1,0 / 1,25. Der
   Maxwell-Wert (0,09 / 0,06 / 0,56 / 0,49 des Referenzwerts) schwankt dagegen stark und ist als
   Vergleichsgröße nicht brauchbar (A2, A3).
3. **Der Einbruch bei 1,25 ist Kompression.** Zeile 2 fällt bei 1,25 mit dichtem Aufbau der
   BEM-Blöcke von 9,67 % auf 0,33 % (Platte 34,4 % → 1,4 %), mit ACA-Toleranz 1e-6 auf 3,37 %. Das ist
   keine Eigenschaft von Netz oder Platte (B).
4. **Die Kompression verfälscht auch die Skalierungen davor.** Bei 1,0 fällt Zeile 2 gesamt von 4,89 %
   auf 0,77 %, die Platte von 19,6 % auf 2,2 %. Das in `bem-near-field.md` ausgewiesene
   Plattenresiduum von 20 % (und 45 % bei 0,5) war überwiegend ACA-Fehler.
5. **Das ändert F_z kaum.** Mit dichtem Aufbau: Lorentz 1,68 N (1,0) bzw. 1,43 N (0,5) gegen 1,72 N
   bzw. 1,49 N komprimiert. Der Überschuss bleibt (A3, B).

**Nicht geklärt:** warum der Lorentz-Wert des BEM-FEM-Pfads rund das 1,7-fache des Referenzpfads
beträgt, und warum das Oberflächenintegral des BEM-FEM-Pfads (Maxwell) davon so weit abweicht.

## A1: Vergleichbarkeit der beiden Lorentz-Kernel

**Gemessen / gelesen**

| | `harmonic_lorentz` (BEM-FEM) | `nodal_lorentz_harmonic` (Referenz) |
|---|---|---|
| Datei | `HarmonicLorentz.cuh` | `NodalMaxwell.cuh` |
| Elemente | Kantenelemente (Nédélec 1. Art) | Q1-Knoten |
| Formel | `k (A_i × B_r − A_r × B_i)`, `k = σω/2` | dieselbe, `k = κω/2` |
| Quadratur | 2×2×2 Gauss (`element_num_quad_points`) | wählbar, hier 3×3×3 (2×2×2 gibt im Referenzlog denselben Wert) |
| Beitragende Elemente | alle mit `kappa > 0`, Maske `nullptr` | `kappa > 0` und Plattenmaske |
| Volumenelement | `quad_w · |detJ|` | `wg · detJ`, `detJ ≤ 0` wird übersprungen |

- **Elementmenge:** im BEM-FEM-Netz (4944 Hexaeder = 1872 Platte + 1536 + 1536 Spulen, keine Luft) hat
  nur die Platte `kappa > 0` (Log: Spulen `kappa=0`; die Luftregularisierung betrifft nur
  `--fem-direct`, dieser Pfad setzt `nullptr`, „regularisiert nicht“). Beide Kernel integrieren also
  über dieselben 1872 Plattenhexaeder.
- **Neuer Test `test_lorentz_kernel_agreement`** (host, CPU): beide Kernel auf demselben Feld.
  - Parallelepiped, affine Felder (beide Räume enthalten sie exakt): Kantenkernel gegen exakt 6,8e-16,
    Knotenkernel 1,4e-15 bis 1,7e-15, beide untereinander 1,0e-15.
  - **Gegenproben:** A_r und A_i vertauscht, und A_i mit falschem Vorzeichen: Abweichung vom Original
    jeweils 2,0 in beiden Kerneln (Kraft kehrt das Vorzeichen um); beide Kernel bleiben dabei
    untereinander gleich (1e-15). Eine Vorzeichen- oder Vertauschungsabweichung zwischen den Kerneln
    würde auffallen.
  - Verzerrtes Hexaeder, quadratisches Feld: Kanten- gegen Knotenkernel 2,2 %, gegen das Exakte
    22,6 % (Kante) bzw. 24,7 % (Knoten). Das ist der Diskretisierungsfehler eines einzelnen stark
    verzerrten Elements und kein Kernelunterschied; Knotenkernel 2×2×2 gegen 3×3×3: 1,6e-16.

**Bewertung:** die Kernel sind vergleichbar, ein Angleichen ist nicht nötig. Die Abweichung der
Gesamtkräfte (A2) kommt nicht vom Kernel.

## A2: Tabelle A mit beiden Schätzern

Aus den vorhandenen Logs in `build/near/` (nichts neu gerechnet), erzeugt mit
`python3 tools/bem_trend_tables.py build/near 0.5 0.75 1.0 1.25`. Der Referenzpfad gibt nur die
Lorentzkraft aus; eine Spalte „Maxwell Referenz“ gibt es nicht (der Spannungstensor ist dort nicht
ausgewertet; Aufwand nicht betrieben).

| Skalierung | F_z Maxwell BEM-FEM [N] | F_z Lorentz BEM-FEM [N] | F_z Lorentz Referenz [N] | Maxwell / Ref | Lorentz / Ref |
|---|---:|---:|---:|---:|---:|
| 0,5 | 0,0774 | 1,49 | 0,85333 | 0,091 | 1,746 |
| 0,75 | 0,0569 | 1,66 | 0,93495 | 0,061 | 1,776 |
| 1,0 | 0,545 | 1,72 | 0,96649 | 0,564 | 1,780 |
| 1,25 | 0,485 | 1,94 | 0,98096 | 0,494 | 1,978 |
| 0,5 (`--bem-near-subdiv`) | 0,0498 | 1,45 | 0,85333 | 0,058 | 1,699 |
| 1,0 (`--bem-near-subdiv`) | 0,518 | 1,72 | 0,96649 | 0,536 | 1,780 |

Zusätzlich neu gerechnet, dichter Aufbau der BEM-Blöcke (`--bem-eta=1e-9`, 50 Hz, siehe B):

| Skalierung | F_z Maxwell BEM-FEM [N] | F_z Lorentz BEM-FEM [N] | Lorentz / Ref (dicht) | Lorentz / Ref (komprimiert) |
|---|---:|---:|---:|---:|
| 0,5 | 0,0092 | 1,43 | 1,676 | 1,746 |
| 1,0 | 0,584 | 1,68 | 1,738 | 1,780 |
| 1,25 | @@1,25_MAXWELL@@ | @@1,25_LORENTZ@@ | @@1,25_RATIO@@ | 1,978 |

Die Maxwell-Werte sind 3-stellig aus dem Log gelesen (8-Phasen-Mittel, `ZEITGEMITTELTE KRAFT`).

## A3: Bewertung

**Gemessen**

- **Maßgeblich ist der Lorentz-Wert.** Der Spannungstensor ist im Projekt als Schätzer des
  Gleichanteils bereits als untauglich dokumentiert (Kommentar in `ReferenceSolver.cu`;
  `HarmonicLorentz.cuh`), und die Tabelle bestätigt das am BEM-FEM-Pfad: Maxwell / Ref springt
  zwischen 0,06 und 0,56, während die Platte sich nach Projektion und Biot-Savart kaum ändert. Der
  Maxwell-Wert ist eine kleine Differenz großer Terme (2ω-Anteil 6- bis 40-mal so groß wie der Mittelwert,
  Auftrag), das Volumenintegral hat diese Auslöschung nicht.
- **Defizit:** mit Lorentz auf beiden Seiten besteht keines. Der BEM-FEM-Pfad liegt 1,7- bis 2,0-mal
  über dem Referenzpfad. Der Vergleich 0,545 N gegen 0,9665 N in den älteren Berichten war ein Vergleich
  zweier verschiedener Schätzer.
- **Konvergenz von Lorentz / Ref (ohne 1,25):** nein. 1,746 → 1,776 → 1,780 für 0,5 / 0,75 / 1,0,
  also praktisch konstant (und leicht steigend), nicht auf 1 zulaufend. Mit dichtem Operator
  1,676 → 1,738 (0,5 → 1,0), ebenfalls nicht auf 1 zulaufend. Das spricht für einen systematischen
  Unterschied zwischen den Pfaden, nicht für einen Diskretisierungsfehler, der mit der Verfeinerung
  verschwindet.
- Die Felder der Platte im BEM-FEM-Pfad sind gegen den Referenzpfad größer, auch mit dichtem Operator
  (Skalierung 1,0, `compare_fields.py`, B/Bref Referenzpfad): Platte Re 1,100, Im 1,326; (komprimiert:
  1,138 / 1,371). Plattenfeld L2 gegen Referenzpfad dicht 0,21 / 0,38 (komprimiert 0,34 / 0,52).

**Interpretation (nicht geprüft)**

- Die um 10 % bis 33 % zu großen Plattenfelder passen der Größenordnung nach zum Faktor 1,7 in der
  Kraft (die Kraft ist ein Produkt aus A und B der Platte). Ob es daran liegt, wurde nicht geprüft,
  insbesondere nicht, welcher Teil des Operators die Plattenamplitude hebt (K1/K2-Frage, Darstellung
  der Tangentialspur): diese Fragen sind ausdrücklich nicht Teil des Auftrags.
- Der große Abstand zwischen Maxwell (0,545 N) und Lorentz (1,72 N) im selben Lauf deutet darauf hin,
  dass das diskrete Feld an der Plattenoberfläche nicht konsistent genug für das Oberflächenintegral
  ist. Das Oberflächenintegral wäre bei exakten Feldern gleich dem Volumenintegral.
- Ob der Referenzpfad selbst der richtige Wert ist, ist nicht gemessen. Beide Pfade sind nicht an einer
  analytischen Lösung oder einer Messung (TEAM 28) für F_z kalibriert.

Die Nachträge in `bem-coupling-fixes.md` und `bem-near-field.md` verweisen hierher; die alten Zahlen
sind nicht überschrieben.

## B: Einbruch bei Skalierung 1,25

### Netz

Gemessen (`tools/meshcheck.py near/hex_sS.vtk --require-hex --separate-bodies`,
`tools/bem_quad_quality.py`, Kopfzeilen von `--bem-row-check`):

- `meshcheck` bei 1,0 und 1,25: nur Hexaeder, keine doppelten oder verwaisten Knoten, Körper getrennt,
  konform, Jacobi-Determinante an allen Ecken positiv (kleinste 8,45 bzw. 4,22).
- Normalenorientierung: **0 von 3696 bzw. 0 von 5730** Quads mit nach innen zeigender Normale.
- Quadklassen (eben / Seite / Kante): Platte 1248 / 48 / 96 (1,0) und 1890 / 120 / 120 (1,25); Spulen
  je 384 / 576 / 192 bzw. 600 / 960 / 240.
- Quadqualität (Seitenverhältnis lang/kurz; Mittelwert, Spanne siehe `build/force/quad_quality.md`):

| Netz | Körper | Deckel/Boden AR mittel (max) | Seiten AR mittel (max) | Hexaeder AR max |
|---|---|---:|---:|---:|
| 1,0 | Platte | 1,37 (2,18) | 8,50 (8,50) | 8,50 |
| 1,25 | Platte | 1,34 (2,09) | 9,07 (9,07) | 9,07 |
| 1,0 | Innenspule | 1,50 (1,98) | 1,47 (1,84) | 1,98 |
| 1,25 | Innenspule | 1,47 (1,98) | 1,47 (1,84) | 1,98 |
| 1,0 | Aussenspule | 3,12 (3,31) | 1,76 (1,91) | 3,31 |
| 1,25 | Aussenspule | 3,11 (3,31) | 1,76 (1,91) | 3,31 |

  Maximale Winkelabweichung der Plattenquads 45° (Kernzone) bei 0,75, 1,0 und 1,25 gleich; Warp 0.
  Die Spulenquads ändern sich zwischen 1,0 und 1,25 praktisch nicht, die Platte wird mit
  der Verfeinerung etwas gleichmäßiger (Deckel) und die Randquads schlanker (8,5 → 9,1).

**Ergebnis:** am Netz ist bei 1,25 nichts Auffälliges gemessen. Das Netz ist als Ursache **ausgeschlossen**,
soweit diese Größen es erfassen (Normalen, Konformität, Jacobi, Quadform).

### Wo kippt Zeile 2?

`tools/bem_row_compare.py build/near/row_s1.0.log build/near/row_s1.25.log`, V0, 2 Hz. Zeile 2 (+K1),
Auszug:

| Körper | Klasse | 1,0 | 1,25 |
|---|---|---:|---:|
| Platte | eben | 19,62 % | 34,49 % |
| Platte | Seite | 18,49 % | 20,16 % |
| Platte | Kante | 14,43 % | 19,45 % |
| Innenspule | alle | 7,92 % | 15,59 % |
| Aussenspule | Seite | 1,10 % | 4,05 % |
| Aussenspule | Kante | 1,01 % | 4,46 % |
| alle | alle | 4,89 % | 9,67 % |

Zeile 1 (+K2) fällt weiter (gesamt 7,44 % → 5,65 %). Die Zunahme in Zeile 2 sitzt nicht an der Platte
allein: sie betrifft alle Körper und Klassen, bei den Spulen sogar relativ stärker (Aussenspule Seite
um das 3,7-fache). Das passt zur Vermutung des Auftrags (D/G und deren Aufbau, nicht T/FEM). Der
Clusterbaum wechselt von 255 auf 511 Knoten, der Anteil der ACA-Blöcke von 2602 von 4516 auf 6694 von
10498.

### Kompression

Die BEM-Blöcke sind komprimiert (ACA+ mit Toleranz 1e-4, Zulässigkeit η = 1,0, Blattgröße 30, alles
fest im Code). Für die Messung gibt es zwei reine Diagnose-Optionen `--bem-aca-tol=X` und
`--bem-eta=X`; die Standardwerte (1e-4, 1,0) sind **nicht** geändert, ohne die Optionen ist das
Verhalten bitgleich (die Standardläufe `row_default_s*` reproduzieren die alten Logs auf alle
angezeigten Stellen). `--bem-eta=1e-9` macht alle Blöcke dicht (65536 dichte Blockpaare; Speicher
reichte; Wanduhrzeit 407 s (1,0) und 848 s (1,25), teils parallel zu anderen Läufen, also nur grob).

Zeile 2 (+K1), V0, 2 Hz, gesamt und Platte (alle Klassen):

| Skalierung | Aufbau | Z2 gesamt | Z2 Platte | Z2 Innenspule | Z2 Aussenspule |
|---|---|---:|---:|---:|---:|
| 1,0 | Standard (ACA 1e-4) | 4,89 % | 19,59 % | 7,92 % | 2,26 % |
| 1,0 | ACA 1e-6 | 2,72 % | 6,77 % | 4,15 % | 1,82 % |
| 1,0 | dicht | 0,77 % | 2,21 % | 1,11 % | 0,54 % |
| 1,25 | Standard (ACA 1e-4) | 9,67 % | 34,37 % | 15,59 % | 4,96 % |
| 1,25 | ACA 1e-6 | 3,37 % | 11,98 % | 5,02 % | 2,12 % |
| 1,25 | dicht | 0,33 % | 1,43 % | 0,52 % | 0,15 % |

Zeile 1 ist von der Kompression unberührt (gesamt 7,44 % bei 1,0 und 5,65 % bei 1,25, in allen drei Aufbauten
identisch), wie zu erwarten: T und K sind nicht komprimiert.

**Ergebnis (gemessen):** Zeile 2 hängt stark an der ACA-Toleranz und fällt mit dichtem Aufbau bei
beiden Skalierungen auf etwa 1 % (gesamt unter 1 %). Die Nicht-Monotonie bei 1,25 ist damit **auf die
Kompression zurückgeführt, nicht auf das Netz**; sie verschwindet im dichten Aufbau (0,77 % → 0,33 %).
Die Kompression ist außerdem schon bei 1,0 und 0,5 der Hauptanteil des gemessenen Plattenfehlers.

**Interpretation (nicht geprüft):** warum der Fehler bei 1,25 gegenüber 1,0 so viel größer ist, könnte an
dem anderen Clusterbaum (kleinere Blätter, mehr ACA-Blöcke, 2,6-mal mehr) liegen; welche Blöcke den
Fehler tragen, wurde nicht aufgeschlüsselt. Eine Toleranz 1e-4 ist ein Fehler im Operator, der in D und G
nicht klein gegen die gesuchte Differenz ist.

### Folgen für #24

- Der Befund von `bem-near-field.md`, die Nahfeldquadratur disjunkter Paare sei **nicht** die Ursache
  des Plattenresiduums, wurde mit dem komprimierten Operator gemessen und gilt nur für diesen. Mit
  dichtem Aufbau gemessen (Skalierung 1,0, 2 Hz, `--bem-near-subdiv`): Zeile 2 Platte 2,21 % → 1,19 %,
  gesamt 0,77 % → 0,72 % (`row_dense_subdiv_s1.0.log`). Die Unterteilung wirkt dann also auf die
  Platte, in einer Größenordnung von einem Prozentpunkt; der ACA-Fehler von 17 Prozentpunkten hatte
  sie zuvor überdeckt.
- F_z ändert sich mit dichtem Aufbau kaum (A2); der Überschuss gegenüber dem Referenzpfad bleibt.

## Optional: Test, der eine Änderung wie in #23 bemerkt (nur Vorschlag, nicht umgesetzt)

Die Fingerprint-Tests auf `team28.vtk` blieben über #23 grün, obwohl sich F_z von −25,2 N auf +0,5 N
(Maxwell) änderte. Vorschlag:

- **Netz:** `near/hex_s0.5` (696 Hexaeder; der 50-Hz-Lauf mit dichtem Aufbau dauerte 105 s) oder ein noch gröberes Plattennetz mit mindestens einer Zelle über der Plattendicke.
- **Größe:** Lorentz-F_z, nicht Maxwell, und zusätzlich die Zeilenresiduen aus `--bem-row-check`
  (Zeile 2 gesamt, Platte); beides ist von der Kompression abhängig, also mit `--bem-eta=1e-9`
  (dicht) messen, damit die Größe eine Eigenschaft des Operators und nicht der Kompression ist.
- **Toleranz:** Zeile 2 gesamt (dicht) unter 2 %, Platte unter 5 % (Messwerte 0,33 % bis 0,77 % bzw.
  1,4 % bis 2,2 %); F_z Lorentz im Band ±15 % um den heutigen Wert (1,43 N bei 0,5), mit dem Hinweis,
  dass der heutige Wert selbst um das 1,7-fache über dem Referenzpfad liegt, der Test also eine
  Regression sichert und keine Richtigkeit. Ein Test gegen den Referenzpfad (gleiches Netz) wäre
  strenger; seine Grenze müsste auf den bekannten Faktor 1,7 gelegt werden und wäre erst nach Klärung
  des Überschusses sinnvoll.

## Tests

- `CUDA_VISIBLE_DEVICES="" ctest -L host`: **41 von 41** (vorher 40; neu `test_lorentz_kernel_agreement`).
- Vollständiges `ctest` (mit GPU): @@CTEST_FULL@@

## Offene Punkte

- **Ursache des Lorentz-Überschusses (Faktor ~1,7) im BEM-FEM-Pfad gegenüber dem Referenzpfad.** Nicht
  untersucht (K1/K2-Formulierung, T, D, G sind ausdrücklich nicht Teil dieses Auftrags). Es ist auch
  nicht gemessen, welcher der beiden Pfade näher an der Physik liegt.
- **ACA-Toleranz und Zulässigkeit als Standardwerte** sind nicht geändert (Auftrag). Gemessen ist, dass
  1e-4 für Zeile 2 nicht genügt; ob 1e-6 oder ein anderes η als Standard taugt, ist eine Entscheidung
  des Auftraggebers (Kosten: ACA 1e-6 +25 % Wanduhrzeit bei 1,25, grob, Läufe nicht auf ruhiger GPU; dicht: nur bis 1,25 gerechnet, Speicher
  wächst quadratisch).
- Zeile 2 bei ACA 1e-6 liegt noch bei 2,7 % bis 3,4 % gesamt; unter 1 % wurde nur dicht erreicht. Eine
  Toleranz dazwischen wurde nicht vermessen.
- Die Zeilenmessung ist bei 2 Hz, die Kraftmessung bei 50 Hz; dicht bei 50 Hz nur für 0,5, 1,0 und
  1,25 (siehe A2), nicht für 0,75.
- Die Aussage, welche Blöcke den ACA-Fehler tragen (Nah- oder Fernfeld, Platte-Platte oder
  Platte-Spule), ist nicht aufgeschlüsselt.
- Für den Referenzpfad gibt es keine Maxwell-Spalte.

## Befehle

Aus `build/` (Netze und Referenzläufe wie in `bem-near-field.md`):

```
python3 ../tools/bem_trend_tables.py near 0.5 0.75 1.0 1.25
python3 ../tools/bem_quad_quality.py near/hex_s0.75.vtk near/hex_s1.0.vtk near/hex_s1.25.vtk
python3 ../tools/meshcheck.py near/hex_s1.25.vtk --require-hex --separate-bodies
python3 ../tools/bem_row_compare.py near/row_s1.0.log near/row_s1.25.log
./test_lorentz_kernel_agreement
# Kompression, Zeilenmessung (2 Hz); gleiche Optionen fuer Skalierung 1.0
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=near/hex_s1.25.vtk --bem-row-check
./fembem ... --bem-row-check --bem-aca-tol=1e-6
./fembem ... --bem-row-check --bem-eta=1e-9
./fembem ... --bem-row-check --bem-eta=1e-9 --bem-near-subdiv
# Kraft (50 Hz) mit dichtem Aufbau
./fembem --harmonic=50 --harmonic-inner-direct --fixed-plate --plate-z=11.3 \
    --mesh=near/hex_s1.0.vtk --bem-eta=1e-9 --harmonic-field-dump=force/dense_bf_s1.0.csv
CUDA_VISIBLE_DEVICES="" ctest -L host
ctest
```
