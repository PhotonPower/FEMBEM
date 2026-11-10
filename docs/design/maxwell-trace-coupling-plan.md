# Umstellung der FEM-BEM-Kopplung auf Maxwell-Spuren (Option c): Fahrplan

Stand: 08.11.2026. Entscheidung und Phasenplan. Grundlage: `exterior-formulation-options.md`,
`bem-trace-consistency.md` und `bem-normal-components.md`.

## Entscheidung

- **Option (c):** Die Kopplung läuft über die eichinvarianten Spuren γ_t A (wirksam über curl_Γ) und
  λ = n × H, mit Laplace-Kernen (quasistatisch).
- **Ziel ist allgemeine Wirbelstrom- und Magnetostatik, nicht nur TEAM 28.** Die TEAM-Serie soll
  durchgerechnet werden können: Spulen als FEM-Körper, Eisen (auch nichtlinear), Leiter beliebiger
  Topologie.
- **Ersetzt werden** der harmonische und der transiente BEM-FEM-Pfad, also der jetzige Operator mit
  D, G, k und K1/K2.
- **Erhalten bleibt** der knotenbasierte Referenzpfad (`--reference`, `NodalBem.cuh`, `NodalMaxwell.cuh`)
  als unabhängige Vergleichsrechnung.

### Einschränkung des Referenzpfads bei Eisen

Knotenbasiertes A mit div-div-Strafterm konvergiert in Gebieten mit einspringenden Ecken oder Kanten,
auch an Materialgrenzen mit Sprung in μ, im Allgemeinen gegen eine **falsche** Lösung. Das ist ein
bekanntes Ergebnis von Costabel und Dauge; Literaturangabe vor Verwendung prüfen. Für Probleme ohne
Eisen (TEAM 28, TEAM 7) bleibt er eine gute Referenz. Bei Eisen mit Ecken (z. B. TEAM 13) sind die
Messdaten und publizierten Ergebnisse der TEAM-Probleme die Referenz, nicht der Knotenpfad.

## Formulierungsfragen, die vor dem Code geklärt sein müssen (Phase 1a)

1. **Raum für λ.** Auf der Luftseite gilt div_Γ λ = −n·curl H = 0. In der quasistatischen Kopplung lebt
   λ daher im Raum der oberflächen-divergenzfreien Felder. Diskret bietet sich λ = curl_Γ ψ mit
   knotenbasiertem ψ auf dem Rand an (Q1, wie in `NodalBem.cuh`), plus die harmonischen Felder der
   Oberfläche. Mit dieser Darstellung wird der Vektor-Einfachschichtblock ⟨λ', V λ⟩ zum skalaren
   hypersingulären Operator in ψ (Nédélec-Identität).
   Die Alternative, die volle RT-Basis mit dem Maxwell-Einfachschichtoperator, braucht einen Term
   ∝ 1/κ² und bricht bei κ = 0 zusammen. Sie ist hier ungeeignet.
2. **Topologie.** Eine Oberfläche vom Geschlecht g hat 2g harmonische Tangentialfelder; die Spulen
   sind Tori mit g = 1. Diese Felder müssen im λ-Raum enthalten sein, sonst fehlt genau die
   Zirkulation von H um den Spulenquerschnitt. Ihre Berechnung ist ein Oberflächengegenstück zu
   `CohomologyGauge.cpp`.
   Meine frühere Aussage, (c) komme ohne Topologiebehandlung aus, war zu optimistisch. Schnittflächen
   im Volumen braucht es nicht, harmonische Oberflächenfelder schon.
3. **Eindeutigkeit und Eichung.** curl_Γ tötet ∇_Γ-Anteile von γ_t A. Wie das gekoppelte System
   eindeutig wird, muss geklärt sein: Innen-Eichung (Baum-Kobaum mit Kohomologie, wie bisher), λ bis
   auf Konstanten von ψ je Zusammenhangskomponente, und das Zusammenspiel mit den nichtleitenden
   Gebieten (Spulen, Eisen ohne σ).
4. **Symmetrische (Hiptmair) oder unsymmetrische (Johnson-Nédélec) Kopplung.** Die Wahl ist zu
   begründen. Symmetrie ist für den Vorkonditionierer und den transienten Pfad wertvoll, die
   unsymmetrische Variante braucht weniger Operatoren.
5. **Vorzeichen, Normalenrichtung, Sprungrelationen.** Ausdrücklich festlegen und jede Identität einem
   numerischen Test zuordnen. In #23 lagen drei von vier Fehlern genau hier.

Literatur, aus dem Gedächtnis zitiert und vor Verwendung zu prüfen: R. Hiptmair, „Symmetric coupling
for eddy current problems“, SIAM J. Numer. Anal. 40 (2002); M. Kuhn und O. Steinbach (FEM-BEM-Kopplung
für 3D-Magnetfeld- und Wirbelstromprobleme, um 2002); A. Bossavit, *Computational Electromagnetism*,
1998 (Topologie, Kohomologie).

## Phasen

| Phase | Inhalt | Abnahme |
|---|---|---|
| 1a | Formulierungsdokument (Fragen oben), keine Code-Änderung am Lösungspfad | Herleitung vollständig, jede Identität einem Test zugeordnet |
| 1b | Operatorbibliothek, dicht, CPU: Randoperatoren, singuläre Quadratur, harmonische Oberflächenfelder | Calderón-Tests an Kugel, Würfel, **Torus** mit exakten Außenfeldern; Eichtest; Konvergenzraten |
| 2 | Magnetostatik: Spulen als FEM-Körper, ohne Platte bzw. σ = 0, neue Kopplung, dicht | Feld gegen Volumenstrom-Biot-Savart auf 1 bis 3 %; Feld eichunabhängig (mit/ohne Eichung) |
| 3 | Harmonisch, TEAM 28 mit Platte, dicht | F_z (Lorentz) bei 50 Hz und 11,3 mm nahe Referenzpfad (0,9665 N) und m g; Netzreihe 0,5 / 1,0 / 1,25 |
| 4 | ACA, GPU-Apply, Vorkonditionierer für die neuen Operatoren | Ergebnisse wie Phase 3 bei ACA-Toleranz mit belegtem Fehler; Laufzeit und Speicher |
| 5 | Transienter Pfad auf die neue Kopplung | TEAM-28-Schwebedynamik gegen Referenz und Messung |
| 6 | Alten BEM-FEM-Operator entfernen (D, G, k, K1/K2, Row-Check-Varianten); Tests umstellen; Fingerprint-Tests, die die Kopplung wirklich abdecken | volle Testsuite grün, Referenzpfad unverändert bitgleich |
| 7 | Erste weitere TEAM-Probleme: TEAM 7 (Leiter mit Loch, Topologie), danach Eisen (z. B. TEAM 13, nichtlinear) | gegen publizierte Mess- bzw. Referenzwerte |

Bis zum Abschluss von Phase 3 bleibt der alte Pfad als Vergleich hinter einer Option erhalten.

## Hilfsbaustein, früh einplanen

**Volumenstrom-Biot-Savart** für rechteckige Ringquerschnitte, als Referenz in Phase 2 und für den
Row-Check an den Spulen. (Korrektur 10.11.2026: Die früher hier genannten 24 bis 69 % waren das Residuum von
Zeile 1 auf den inneren Spulenzeilen, wo das Fadenfeld zwischen den Fäden singulär ist, nicht eine Abweichung im |B| am Rand;
am Rand liegt das Fadenmodell bei 0,1 bis 3 %, siehe `maxwell-trace-formulation.md`.) Er gehört zu Phase 1b oder
direkt davor.

## Was bleibt, was ersetzt wird

**Bleibt:**
- FEM-Assemblierung (Nédélec), Baum-Kobaum- und Kohomologie-Eichung,
- Cluster-Baum, Blockpaare, ACA-Infrastruktur, GPU-Apply,
- `BemCouplingMap` (Randkante → FEM-Kante),
- Sauter-Schwab-Rahmen, Lorentz- und Wirbelstromdiagnose, Referenzpfad.

**Ersetzt:**
- D, G und die Unbekannte k (kovariante Rooftops),
- T in seiner jetzigen Rolle (die Massenkopplung ⟨λ, γ_t v⟩ entsteht neu),
- Row-Check-Varianten K1/K2 und Normalanteile.

**Neu:**
- λ-Raum mit harmonischen Oberflächenfeldern,
- hypersingulärer Operator in curl-Form,
- Doppelschichtoperator mit ∇G ×,
- Volumenstrom-Biot-Savart.
