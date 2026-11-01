# BEM-FEM-Hauptpfad: das Feld stimmt nicht

Stand 2026-11-01. Anlass: Der harmonische BEM-FEM-Pfad konvergiert seit
`--harmonic-inner-direct` und `apply_many` auf hex_s1 in 24 Iterationen, gibt
aber F_z = 0,014 N statt 1,007 N (Ringmodell, 11,3 mm). Das Gleichungssystem
ist gelöst; gesucht war, wo der Fehler sitzt.

## Werkzeuge

- Der harmonische Pfad druckt B_z auf der Achse in der Platte gegen
  Biot-Savart (`report_field_verification`, dieselbe Tabelle wie
  `--verify-field`).
- `--harmonic-field-dump=DATEI` schreibt A und B je Element (CSV) und dazu
  das FEM-System: Kanten mit Endpunkten, Rand- und Eichkennung, b_r und
  W = K + ωC als Tripel.
- `tools/team28_field_check.py` vergleicht das Feld je Körper mit
  Biot-Savart abseits der Achse (Kreisstromfäden, elliptische Integrale;
  die Formeln sind gegen die exakte Gegeninduktivität und gegen numerische
  Ableitungen auf 1e-9 geprüft), auf Wunsch mit den Wirbelströmen aus dem
  Ringmodell.

Bei `--harmonic=2` ist die Rückwirkung der Platte unter 2 % (Ringmodell:
Plattenstrom 28 A gegen 648 A bei 50 Hz); der Realteil muss dann das reine
Spulenfeld sein.

## Befunde

**1. Das Feld ist auf beiden Netzen falsch.** B_z auf der Achse in der
Platte, 11,3 mm, Soll ≈ 0,06 T:

| Netz | 50 Hz | 2 Hz |
|---|---|---|
| hex_s1 (Gmsh) | −0,12 … +0,04 T, Vorzeichen wechseln auf gleicher Höhe | — |
| team28.vtk | 0,1 … 1 mT, glatt | 0,2 … 1,7 mT |

**2. Schon in den Spulen.** team28.vtk, 2 Hz, Projektion des Lösungsfelds
auf Biot-Savart (Soll 1):

| Körper | Projektion B | B_φ/B (Soll 0) |
|---|---|---|
| Platte | +0,002 | 0,13 |
| Innenspule | +0,28 | 0,24 |
| Außenspule | +0,04 | 0,17 |

**3. Nicht die Kohomologie-Projektion.** Mit `--nocohomology`: +0,33 /
+0,05, praktisch unverändert.

**4. Quellterm und FEM-Matrix stimmen.** Die exakte Biot-Savart-Lösung als
Kantenintegrale (4-Punkt-Gauß) in die Zeilen der Spuleninnenkanten
eingesetzt — dort wirkt die BEM nicht:

| Spule | Innenkanten | ‖W A_ex − b‖/‖b‖ | bester Faktor |
|---|---|---|---|
| innen | 1471 | 2,8 % | 0,997 |
| außen | 481 | 1,6 % | 1,012 |

Damit bleibt die Kopplung an den Außenraum (T, D_BEM, G_BEM).

**Nebenbefund, behoben:** Bei `--harmonic=f` wurde der Quellterm mit
1/sin(2πf·dt) auf die Amplitude zurückgerechnet, die Spulen schwingen in
`update_state()` aber mit 50 Hz. Bei 2 Hz war der Quellterm 25-fach zu
groß. Jetzt wird mit der Spulenfrequenz skaliert.

## Vermutung: die Formulierung, nicht ein Vorzeichen

Die BEM stellt A komponentenweise dar,
(½ + D) A = G ∂A/∂n, und identifiziert μ₀K mit ∂A/∂n. Der FEM-Randterm aus
der partiellen Integration von rot ν rot A ist dagegen n × H. Auf einer
ebenen Fläche gilt

    n × rot A = ∇(A·n) − ∂A/∂n,

beide stimmen also nur überein, wenn A·n auf dem Rand keinen Gradienten hat.
Das ist hier zweimal verletzt:

- Die Kanten-Elemente mit Baum-Kobaum-Eichung liefern A nur bis auf einen
  Gradienten; die komponentenweise Darstellung verlangt das Coulomb-geeichte,
  harmonische A im Außenraum.
- Auf den facettierten Zylinderflächen der Spulen ist A_φ e_φ · n über jede
  Facette nicht konstant; ∇(A·n) ist dort von der Größe A_φ/R, und
  ∂A/∂n ~ A/h hat bei R = 27…95 mm und h ≈ 5 mm dieselbe Größenordnung.

Gegenprobe ohne Baum-Eichung (`--nogauge --nocohomology`, 2 Hz, exakter
innerer Löser): der äußere Löser stagniert bei 0,017 (900 Iterationen in
einer Stunde). (W+T) ist ohne Eichung also fast singulär -- die BEM-Zeilen
legen die Gradientenrichtungen nicht kräftig fest. Das widerlegt die
Vermutung nicht (die Eichabhängigkeit kann klein und trotzdem
feldverfälschend sein), stützt sie aber auch nicht; ein Feld ergab der Lauf
nicht.

Der Referenzpfad (`--reference`, Knotenelemente, Strafterm-Eichung, volle
BEM) ist für genau diese Darstellung gebaut. Läuft seine Feldprüfung gegen
Biot-Savart sauber, ist die Kopplung der Kanten-Elemente an die
komponentenweise BEM die Ursache.

## Was daraus folgt

Ein Vorzeichen- oder Faktorfehler in T oder D_BEM ist nicht ausgeschlossen,
aber unwahrscheinlich: das Feld ist nicht skaliert falsch, sondern in der
Form (Projektionen 0,04 bis 0,3, große B_φ-Anteile). Gegenproben in dieser
Reihenfolge:

1. Feldprüfung des Referenzpfads gegen Biot-Savart.
2. Das exakte Paar (A, K) in die BEM-Zeilen einsetzen, einmal mit
   μ₀K = ∂A/∂n und einmal mit K = −n × H: welches erfüllt sie?
3. Ist die Formulierung die Ursache: Kopplung über n × A und n × H mit einer
   Maxwell-BEM (Johnson-Nédélec bzw. symmetrisch nach Costabel), oder
   reduziertes Skalarpotential im Außenraum (A-ψ).
