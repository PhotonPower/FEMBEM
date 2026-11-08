# Auftrag Phase 1a: Formulierung der Kopplung in Maxwell-Spuren

Stand: 08.11.2026. Basis: `main` nach Merge von `bem-normal-components`. Zweig:
`maxwell-trace-formulation`. Ergebnis: `docs/theory/maxwell-trace-coupling.md` plus ein kurzer Bericht
`docs/design/maxwell-trace-formulation.md`. Den Zweig nach jedem Commit pushen.

## Anlass

Die bestehende Kopplung (komponentenweise Vektor-Laplace-BEM mit einer Unbekannten k) ist als
Formulierungsfehler diagnostiziert (`bem-trace-consistency.md`, `bem-normal-components.md`). Sie wird
ersetzt durch eine eichinvariante quasistatische Kopplung in den Spuren γ_t A und λ = n × H, mit
Laplace-Kernen. Fahrplan und Entscheidung: `maxwell-trace-coupling-plan.md`.

Ziel des Codes ist allgemeine Wirbelstrom- und Magnetostatik (TEAM-Serie): Spulen als FEM-Körper,
Eisen (auch nichtlinear) und Leiter beliebiger Topologie.

**Dieser Auftrag ist reine Theorie.** Es entsteht ein Formulierungsdokument, das so vollständig ist,
dass Phase 1b (Operatorbibliothek mit Tests) danach ohne Formulierungsfragen umgesetzt werden kann.
**Kein Code am Lösungspfad.** Kleine Python- oder Host-Skripte zur Überprüfung einzelner Identitäten
sind erlaubt und erwünscht.

## Was das Dokument klären muss

### 1. Kontinuierliches Problem

Gebiete und Annahmen festlegen:
- Leiter Ω_c mit σ > 0,
- nichtleitende FEM-Gebiete (Spulen mit eingeprägtem J_s, Eisen mit μ(|B|)),
- Luftraum Ω_e außerhalb aller FEM-Körper, μ = μ₀.

Gleichungen innen (A-Formulierung, harmonisch und transient) und außen (curl curl A = 0, div A = 0,
Abklingverhalten). Spuren festlegen: γ_t, γ_×, γ_N, Normalenrichtung, Orientierung. Bezug auf die
Konventionen im Code (`BemIntegration.cuh`, `NodalBem.cuh`), Abweichungen benennen.

### 2. Außenraum-Darstellung und Calderón-Operatoren bei κ = 0

Die Darstellungsformel für das Außenfeld in den Spuren γ_t A und λ herleiten. Die Randoperatoren mit
Laplace-Kern angeben, mindestens:
- Einfachschicht auf λ,
- hypersingulärer Operator in curl_Γ-Form,
- Doppelschichttyp mit ∇G × und seinen Adjungierten.

Jeweils angeben:
- Sprungrelationen,
- Symmetrieeigenschaften,
- Kerne (welche Felder sie annullieren).

### 3. Raum für λ

Zeigen, dass auf der Luftseite div_Γ λ = 0 gilt. Begründen, warum die volle RT-Basis mit
Maxwell-Einfachschicht bei κ = 0 ungeeignet ist (Term ∝ 1/κ²).

Die diskrete Darstellung λ = curl_Γ ψ (ψ knotenbasiert, Q1 auf den Randquads) plus harmonische
Oberflächenfelder ausarbeiten. Prüfen und angeben, ob der Einfachschichtblock damit zum skalaren
hypersingulären Operator in ψ wird (Nédélec-Identität). Falls ja: Was davon ist im Code schon
vorhanden (`NodalBem.cuh`, Kollokation statt Galerkin)?

### 4. Topologie

- **Oberflächen vom Geschlecht g:** Dimension und Konstruktion der harmonischen Tangentialfelder
  (2g je Fläche) auf einem Quadnetz. Algorithmus in Anlehnung an `CohomologyGauge.cpp`, aber auf der
  Oberfläche.
- **Physikalische Bedeutung:** Welche Größe legen diese Freiheitsgrade fest (Zirkulation von H um den
  Spulenquerschnitt, Fluss durch das Loch)? Wie ist das mit dem eingeprägten Strom verträglich?
- **Leiter mit Loch (TEAM 7):** Was zusätzlich gilt, und ob die Innen-Kohomologie dort weiter nötig ist.
- **Mehrere Körper:** Je Zusammenhangskomponente des Rands eine Konstante in ψ. Wie wird sie fixiert?

### 5. Gekoppeltes System

Das diskrete Blocksystem in (a, ψ, harmonische Koeffizienten) explizit hinschreiben, für die
symmetrische Kopplung (Hiptmair) und für die unsymmetrische (Johnson-Nédélec). Dazu:

- **Wahl mit Begründung:** Vorkonditionierer, transienter Pfad, Anzahl der Operatoren, Nutzen der
  vorhandenen `--harmonic-inner-direct`-Struktur.
- **Eindeutigkeit:** Innen-Eichung (Baum-Kobaum mit Kohomologie), Konstanten in ψ, nichtleitende
  Gebiete. Wo genau wird geeicht, und warum hängt das Feld B dann nicht von der Eichung ab?
- **Eichinvarianz:** Ausdrücklich zeigen, dass jeder Block, der auf a wirkt, a + G ψ_h für beliebige
  diskrete ψ_h (also auch nicht harmonische) gleich behandelt. Das ist der Fall 4 aus
  `bem-normal-components.md`, an dem der alte Operator scheitert.
- **Eingeprägter Strom:** Wo steht J_s, und wie geht er in die Gleichungen ein?
- **Transient:** Zeitdiskretisierung (implizit, wie bisher). Bleiben die Randoperatoren
  zeitunabhängig? Was gilt bei bewegten Körpern (Neuaufbau, welche Blöcke)?
- **Eisen:** Nichtlinearität nur im FEM-Teil. Bestätigen, dass die Randoperatoren unberührt bleiben,
  und angeben, was am Newton- bzw. Fixpunktverfahren zu beachten ist.

### 6. Diskretisierung und Quadratur

- **Integranden:** Welche singulären Integranden sind neu (Selbst-, Kanten-, Eckpaare)? Welche lassen
  sich durch partielle Integration auf schwach singuläre zurückführen? Was davon deckt der vorhandene
  Sauter-Schwab-Rahmen ab?
- **Kollokation oder Galerkin:** für jeden Block, mit Begründung.
- **ACA:** Welche Blöcke sind komprimierbar, und mit welchen Kernen?

### 7. Testplan für Phase 1b

Jede Identität aus 2 bis 5 einem numerischen Test zuordnen: Geometrie, exaktes Feld, erwartete Größe,
erwartete Konvergenzordnung und eine Gegenprobe, die durchfallen muss.

Mindestens:
- **Kugel:** Dipolfeld, A = μ₀ m × r / (4π r³) in Coulomb-Eichung.
- **Würfel:** wie Kugel, mit Kanten und Ecken.
- **Torus:** mit Kreisstrom im Inneren, sodass H außen eine Zirkulation um den Querschnitt hat. Prüft
  die harmonischen Felder.
- **Eichtest:** a + G ψ_h mit nicht harmonischem ψ_h.

Dazu die Volumenstrom-Biot-Savart-Rechnung für rechteckige Ringquerschnitte als Referenz für Phase 2
spezifizieren: Verfahren, Genauigkeit, Test.

## Form

- `docs/theory/maxwell-trace-coupling.md`: Herleitungen mit Vorzeichen und Normalenkonvention,
  Literaturangaben mit vollständigen bibliographischen Daten (selbst geprüft).
- `docs/design/maxwell-trace-formulation.md`: kurzer Bericht mit offenen Punkten, den getroffenen
  Wahlen und einer Liste der Risiken für Phase 1b.
- Eigene Verifikationsskripte unter `tools/theory/` mit Angabe, welche Identität sie prüfen.

## Nicht Teil dieses Auftrags

- Änderungen am Lösungspfad, an Operatoren oder an der Eichung.
- Implementierung der Operatorbibliothek (Phase 1b).
- ACA und GPU.
- Der Referenzpfad wird nicht angefasst.
