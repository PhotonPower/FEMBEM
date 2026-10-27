#ifndef COHOMOLOGY_GAUGE_H
#define COHOMOLOGY_GAUGE_H

#include <vector>
#include "Mesh.cuh"
#include "MaterialTypes.cuh"
#include "TreeCotreeGauge.h"

// =============================================================================
// Kohomologie-Erweiterung der Baum-Kobaum-Eichung -- diskrete Hodge-Zerlegung
// =============================================================================
//
// PROBLEM
// -------
// build_tree_cotree_gauge() entfernt aus Ker(curl) exakt den Gradientenanteil.
// Ist das nichtleitende Gebiet nicht einfach zusammenhaengend -- bei TEAM 28
// ist jede Ringspule ein Volltorus mit erster Bettizahl b1 = 1 --, bleibt pro
// Loch zusaetzlich ein "harmonischer" Kozykel im Nullraum: eine Kantenbewertung
// a mit delta(a) = 0 (verschwindende Zirkulation um jede Flaeche), die aber
// KEIN Gradient ist.
//
// EIN FEHLGESCHLAGENER ERSTER VERSUCH (zur Warnung dokumentiert)
// ---------------------------------------------------------------
// Naheliegend, aber FALSCH: den Fundamentalkreis einer Kobaum-Kante (Baumpfad
// plus diese Kante) direkt als Kandidaten zu testen. Ein Fundamentalkreis ist
// ein KETTEN-Zyklus (geschlossener Weg, d. h. der RAND-Operator angewendet auf
// die Kette ist null: d(Kreis) = 0). Ein Kozykel ist eine andere Bedingung auf
// demselben Vektorraum: delta(a) = 0, verschwindende Zirkulation um JEDE
// Flaeche. Diese beiden Eigenschaften fallen nicht automatisch zusammen --
// ein Gradient etwa ist immer ein Kozykel (delta*d0 = 0, Kettenkomplex-
// Identitaet), aber im Allgemeinen KEIN geschlossener Kettenzyklus. Der erste
// Versuch fand deshalb bei keiner einzigen von ueber 6000 Kobaum-Kanten einen
// gueltigen Generator.
//
// KORREKTE LOESUNG: diskrete Helmholtz-Hodge-Zerlegung
// -----------------------------------------------------
// Fuer jede nichtleitende Zusammenhangskomponente (hier: jede Ringspule):
//
//   1. Zufaelligen Startvektor a0 auf den Kanten der Komponente waehlen.
//   2. Auf Ker(delta_1) projizieren: (delta_1 delta_1^T + eps*I) y = delta_1 a0
//      loesen (CG), a_proj = a0 - delta_1^T y. Jetzt gilt delta_1(a_proj) = 0.
//   3. Den Gradientenanteil herausprojizieren: das Knotenpotential phi loesen,
//      das a_proj am besten als Gradient erklaert
//      (delta_0^T delta_0 + eps*I) phi = delta_0^T a_proj (Graph-Laplace, CG),
//      a_harm = a_proj - delta_0(phi).
//
// a_harm liegt (numerisch) in Ker(delta_1) und ist orthogonal zum
// Gradientenraum -- also ein Vertreter der fehlenden Kohomologieklasse.
// delta_1 und delta_0 sind rein kombinatorisch (Vorzeichen aus den bereits an
// anderer Stelle etablierten Konventionen), unabhaengig von Materialwerten
// oder Anregungsart.
//
// ANWENDUNG ALS EICHUNG
// ----------------------
// Anders als die Baum-Kobaum-Eichung (die einzelne KOORDINATEN auf null
// setzt) ist a_harm ein DICHTER Vektor ueber viele Kanten. Die Eichung wird
// daher als zusaetzliche Projektion entlang dieses Vektors formuliert:
//
//     P_c(x) = x - c * (c^T x) / (c^T c)
//
// mit c = a_harm (nach Herausprojizieren der bereits geeichten
// Baum-Koordinaten, damit die Gesamtprojektion idempotent bleibt). Da die
// beiden Spulen keine Kante teilen (siehe Projektstand-Dokumentation,
// "es gibt kein Luftnetz"), haben ihre beiden a_harm-Vektoren disjunkten
// Traeger und sind damit automatisch orthogonal zueinander -- die beiden
// Zusatzprojektionen lassen sich unabhaengig nacheinander anwenden.
// =============================================================================

struct CohomologyGenerator {
    int component_id = -1;
    // Dichter Vektor ueber ALLE Kanten (ausserhalb der Komponente exakt null),
    // bereits orthogonal zu den Baum-Koordinaten der Basis-Eichung.
    std::vector<double> vector;
    double residual_cocycle = 0.0;   // ||delta_1 a_harm|| / ||a_harm||, zur Diagnose
    double norm = 0.0;               // ||a_harm||, zur Diagnose (darf nicht ~0 sein)
};

struct CohomologyGaugeResult {
    std::vector<CohomologyGenerator> generators; // ein Eintrag pro gefundenem Loch

    /// Komponenten mit b1 = 0, also einfach zusammenhaengende. Nur belegt,
    /// wenn allow_simply_connected gesetzt war -- sonst wirft der Fall.
    /// Getrennt gezaehlt, damit "keine Generatoren" von "nicht gesucht"
    /// unterscheidbar bleibt: beides ergibt generators.empty().
    int num_simply_connected = 0;
};

// Findet die Kohomologie-Generatoren fuer alle nichtleitenden Komponenten von
// base_gauge (siehe build_tree_cotree_gauge) und gibt sie als dichte Vektoren
// zurueck. Veraendert base_gauge NICHT -- die Generatoren werden separat als
// zusaetzliche Projektionsvektoren zurueckgegeben, weil sie (anders als
// Baumkanten) keine einzelnen Koordinaten sind.
//
// materials wie in build_tree_cotree_gauge.
//
// Wirft std::runtime_error bei numerisch auffaelligen Ergebnissen (Residuum zu
// gross, Norm zu klein) -- das waere ein Zeichen echter Inkonsistenz und soll
// nicht still uebergangen werden.
/// `allow_simply_connected` erlaubt Komponenten mit b1 = 0.
///
/// Standard ist false, und das ist fuer den BEM-Pfad richtig: dort ist das
/// nichtleitende Gebiet die Spule selbst, ein Volltorus, und jede Komponente
/// MUSS einen Generator haben. Ein verschwindender waere dort ein Fehler und
/// soll werfen.
///
/// Auf dem Pfad mit mitvernetzter Luft (--fem-direct-gauge) ist es umgekehrt:
/// das nichtleitende Gebiet ist Luft plus Spulen, also Kasten minus Platte,
/// und das ist einfach zusammenhaengend. Dort ist b1 = 0 der NORMALFALL, und
/// der Wurf waere falsch.
///
/// Die beiden Alternativen in der Fehlermeldung ("b1 = 0" oder "eps zu gross")
/// sind uebrigens unterscheidbar: eine verschwindende NORM heisst b1 = 0, ein
/// schlechter KOZYKEL-Rest heisst eps zu gross. Letzteres prueft die Zeile
/// darunter weiterhin, unabhaengig von diesem Kennzeichen.
CohomologyGaugeResult find_cohomology_generators(
    const Mesh& mesh,
    const GaugeInfo& base_gauge,
    const MaterialProperties* materials,
    double kappa_threshold = 0.0,
    bool allow_simply_connected = false);

#endif // COHOMOLOGY_GAUGE_H
