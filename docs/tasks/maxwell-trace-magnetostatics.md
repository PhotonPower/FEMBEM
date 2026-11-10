# Auftrag Phase 2: Magnetostatik der TEAM-28-Spulen mit der Maxwell-Spur-Kopplung (dicht)

Stand: 10.11.2026. Basis: `main` nach diesem Auftrag. Zweig: `maxwell-trace-magnetostatics`, ein Commit je
Schritt, nach jedem Commit pushen. Bericht: `docs/design/maxwell-trace-magnetostatics.md`.

## Anlass und Grundlage

Phase 1b ist abgeschlossen (`docs/design/maxwell-trace-operators.md`, #30). Die Operatoren und das
gekoppelte System (S) sind auf kleinen Netzen konsistent: T1 bis T9 bestehen, und T8 (Kugel, Ende zu Ende)
liegt höchstens 2 % neben einer reinen FEM-Rechnung mit exakten Randdaten.

Phase 2 setzt das System erstmals auf TEAM 28 ein, und zwar **magnetostatisch**: σ = 0 überall,
μ = μ₀ überall, Quelle sind die beiden Spulen. Die Platte ist dabei ein FEM-Körper ohne Wirkung. Die exakte
Lösung ist der Volumenstrom-Biot-Savart der beiden Rechteckringe. Das ist das Abnahmekriterium 2 des
Fahrplans (`docs/design/maxwell-trace-coupling-plan.md`): Feldfehler auf dem Niveau des Referenzpfads
(1 bis 3 %), konvergent über die Netze.

**Vergleich mit dem alten Pfad:** `force-excess-localization.md` hat bei 0,01 Hz für den alten
BEM-FEM-Pfad +28 % im Zentrum der Platte und −14 bis −21 % im Rückfluss gemessen, für den Referenzpfad
1 bis 3 %. Die Auswertung in diesem Auftrag soll direkt mit diesen Zahlen vergleichbar sein (gleiche Orte,
gleiche Größen).

**Der bestehende Lösungspfad bleibt unverändert.** Die neue Rechnung läuft als eigener Treiber bzw. als
eigene Option im Trace-Code, nicht über `fem.cu`.

**Hinweis CI:** Eigen3 ist seit Commit dcfa36a Pflicht (`FEMBEM_REQUIRE_EIGEN`, Standard ON), und die CI
installiert `libeigen3-dev`. Die Trace-Tests laufen dort also tatsächlich. Prüfen, dass die CI auf `main`
grün ist, bevor du anfängst; sonst zuerst das beheben.

## Schritt 0: Klärungen aus Phase 1b (zuerst, je eigener Commit)

### 0a. Skalierung in die Formulierung

Im unskalierten System skaliert der ψψ-Block (−μ₀W) mit μ₀, K und der aψ-Block mit 1/μ₀; dazu kommen
Millimeter gegen Meter. Phase 1b hat S deshalb als „numerisch singulär“ gemessen.

Symmetrisch umskalieren: ψ = ψ̃/μ₀ und c = c̃/μ₀. Damit skalieren alle Blöcke mit 1/μ₀, und die Symmetrie
bleibt erhalten. In `TraceSystem` fest einbauen, die Rückskalierung bei der Ausgabe von ψ, c und λ
vornehmen. Geometrie konsequent in Metern.

Prüfen:
- T6 (Symmetrie) bleibt auf 1e-15.
- Die Lösungen von T8 sind nach Rückskalierung bis auf etwa 1e-12 identisch mit vorher.
- Die Konditionszahl von S am gröbsten TEAM-28-Netz (`near/hex_s0.5.vtk`, 50 Hz und 0,01 Hz), unskaliert
  gegen skaliert gegen diagonalskaliert. Die Messung aus Phase 1b wiederholen.

### 0b. c_Meridian = 0,945 statt 1 (R3-Test)

Im R3-Test (Ring mit ungleichmäßiger Winkelteilung) ergibt das gelöste System c_Meridian = 0,945 mit
Quellprojektion und 0,972 ohne, erwartet ist 1. Nach der Theorie (Abschnitt 4.2, Testfunktion v_cut) muss
c_Meridian exakt der Nettostrom durch die Schnittfläche der **verwendeten** rechten Seite sein.

Zu klären und im Bericht festzuhalten:
- Wie groß ist der Nettostrom ⟨f, v_cut⟩ vor und nach der Projektion?
- Stimmt das gelöste c_Meridian mit diesem Nettostrom überein (bis auf Lösertoleranz)?
- Wenn die Projektion den Nettostrom ändert: Das ist ein Amplitudenfehler der Quelle, und er ginge
  quadratisch in die Kraft ein. Die Quelle dann so aufbauen, dass sie diskret divergenzfrei ist **und**
  je Spule exakt N·I trägt. Zwei Möglichkeiten: Renormierung nach der Projektion (begründen, warum das
  zulässig ist), oder die Quelle über ein diskretes Quellfeld t_s mit curl t_s = J_s, sodass Divergenzfreiheit
  und Nettostrom per Konstruktion gelten. Die Wahl begründen.
- Wenn die Projektion den Nettostrom nicht ändert: woher die Abweichung dann kommt.

Test (host): Ring mit ungleichmäßiger Teilung. Gᵀf = 0 und Nettostrom = N·I je Schnittschicht auf 1e-12,
gelöstes c_Meridian = N·I auf Lösertoleranz. Gegenprobe: ohne die Korrektur muss der Test durchfallen.

## Schritt 1: TEAM-28-Aufbau mit der neuen Kopplung

Das System (S) auf `team28_hex.vtk` und der Netzreihe `near/hex_s0.5.vtk`, `hex_s1.0.vtk` (und `hex_s1.25.vtk`,
falls dicht machbar) aufbauen:

- **Rand:** drei Komponenten, g = 1, 1, 0, also 4 harmonische Felder. Das ist schon in T9 geprüft.
- **Innen:** K mit σ = 0 überall, Baum-Kobaum im ganzen FEM-Gebiet. Keine Kohomologie-Eichung, das
  übernimmt (S3).
- **Quelle:** Spulenstrom wie im Produktionsfall (Windungszahl, Strom, Richtung aus der Konfiguration), über
  die Quellkonstruktion aus 0b.
- **Lösung:** innere Kanten dünn eliminieren (Eigen, LDLᵀ oder LU), dann das dichte Schurkomplement auf
  (a_Γ, ψ̃, c̃) direkt lösen. Größe und Speicher je Netz angeben.

Plausibilitätsausgaben je Lauf:
- c_Meridian je Spule gegen N·I,
- c_Längs je Spule (erwartet ≈ 0, weil die Ströme azimutal sind),
- Residuum des Gesamtsystems,
- Laufzeit je Teilschritt.

## Schritt 2: Feld gegen Volumenstrom-Biot-Savart

Referenz: der Volumenstrom-Biot-Savart aus Phase 1a (`tools/theory/check_volume_biot_savart.py`,
Genauigkeit etwa 1e-7). Als C++-Funktion oder als Nachauswertung in Python; die Wahl begründen.

Auszuwerten:
- **in der Platte** (FEM, curl a), am besten mit `tools/compare_fields.py` oder gleichwertig, damit die
  Zahlen direkt neben denen aus `force-excess-localization.md` stehen: Projektion Re auf Biot-Savart,
  relativer L2-Fehler, B_z im Zentrum, im Rückfluss, am Rand;
- **in den Spulen** (FEM): dieselben Maße;
- **im Luftraum** über die Darstellungsformel (R-B) aus λ und g, an Punkten in 1, 5 und 20 mm Abstand zu
  den Körpern und auf der Achse.

Tabelle je Netz und Konvergenzordnung. Daneben in einer Spalte die alten Werte (alter BEM-FEM-Pfad bei
0,01 Hz, Referenzpfad) aus `force-excess-localization.md`.

## Schritt 3: Eichunabhängigkeit und Gegenproben

- **Zwei verschiedene Bäume** (z. B. anderer Startknoten oder andere Kantenreihenfolge): B in allen
  Auswertungen bis auf Lösertoleranz gleich, a dagegen verschieden. Das ist das Kriterium 4 des
  Fahrplans für den Fall σ = 0.
- **Gegenproben:** Vorzeichen von λ in der FEM-Zeile; ohne harmonische Spalten; mit c_Meridian := 0. Jede
  muss sichtbar schlechter werden (Faktor angeben).

## Schritt 4 (optional, nur als Vorschau)

Wenn Schritt 1 bis 3 bestehen und die Rechnung dicht machbar ist: ein harmonischer Lauf bei 50 Hz,
Platte mit σ = 3,4·10⁷ S/m bei 11,3 mm, Skalierung 0,5 und 1,0. F_z als Lorentz-Volumenkraft (Kernel aus
`force-estimator-check`) gegen den Referenzpfad (0,9665 N) und m g (1,05 N).

Nur berichten, nicht bewerten und nichts daraufhin ändern. Die eigentliche Abnahme ist Phase 3.

## Nicht Teil dieses Auftrags

- ACA, GPU, Vorkonditionierer, iterative Löser.
- Einbau in `fem.cu` oder in den Hauptlauf.
- Transienter Pfad.
- Entfernen des alten Pfads.
- Änderungen am Referenzpfad.

## Form

- Bericht wie bisher: Hardware und Basis-Commit im Kopf, „Ergebnis in Kürze“, „Gemessen“ klar getrennt
  von „Interpretation (nicht geprüft)“, Befehle am Ende.
- Neue Tests als ctest mit Marke host, wo die Laufzeit es zulässt (unter 120 s, sonst begründen). Die
  großen TEAM-28-Läufe sind Messtreiber, keine Tests.
- Testzählung vorher und nachher (`ctest -L host`, vollständiges `ctest`). Abweichungen vom Auftrag
  begründen.
