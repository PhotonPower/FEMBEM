#ifndef ALGEBRAIC_MULTIGRID_CUH
#define ALGEBRAIC_MULTIGRID_CUH

// =============================================================================
// AlgebraicMultigrid.cuh -- geglaettete Aggregation fuer die AMS-Hilfsraeume
// =============================================================================
//
// Wozu. Die Messung des AMS-Vorkonditionierers zeigte einen Gewinn von nur
// rund Faktor 2 gegenueber Jacobi, statt der netzunabhaengigen Konvergenz, die
// das Verfahren eigentlich liefert. Der Engpass sind nicht Struktur oder
// Hilfsraeume, sondern deren LOESUNG: dort stand ILU(0) mit Diagonalshift.
// Die Hilfsprobleme A_G = G^T A G und A_Pi = Pi^T A Pi sind aber
// Knoten-Laplace-Probleme -- genau der Fall, fuer den AMG gebaut ist und fuer
// den ILU(0) bekannt schlecht skaliert.
//
// -----------------------------------------------------------------------------
// Verfahren: geglaettete Aggregation (smoothed aggregation)
// -----------------------------------------------------------------------------
//   1. Staerke der Kopplung:  i und j gelten als stark gekoppelt, wenn
//        |a_ij|^2 > theta^2 |a_ii a_jj|
//   2. Aggregation: gieriges Dreipassverfahren auf dem Graphen der starken
//      Kopplungen.
//   3. Vorlaeufige Prolongation P_tent: stueckweise konstant ueber die
//      Aggregate, spaltenweise normiert.
//   4. Glaettung: P = (I - omega D^-1 A) P_tent mit omega = 4/(3 lambda_max).
//      Das ist der Schritt, der geglaettete von reiner Aggregation
//      unterscheidet und erst die netzunabhaengige Konvergenz bringt.
//   5. Grobgitteroperator nach Galerkin: A_c = P^T A P.
//   6. Rekursion bis unter die Grenze, dort direkte Loesung.
//
// V-Zyklus mit gedaempfter Jacobi-Glaettung vor und nach der Grobgitter-
// korrektur.
//
// -----------------------------------------------------------------------------
// Systeme mit mehreren Freiheitsgraden je Knoten
// -----------------------------------------------------------------------------
// A_Pi hat DREI Freiheitsgrade je Knoten (die kartesischen Komponenten). Die
// Aggregation laeuft dann auf dem KNOTENgraphen, und jedes Aggregat erhaelt
// drei Grobgitter-Freiheitsgrade, einen je Komponente. Wuerde man stattdessen
// skalar ueber alle 3N Freiheitsgrade aggregieren, koennten Komponenten
// verschiedener Richtungen in dasselbe Aggregat geraten -- der Nahnullraum der
// konstanten Vektorfelder waere dann nicht mehr exakt darstellbar, und genau
// den muss die Grobgitterkorrektur einfangen.
//
// -----------------------------------------------------------------------------
// Singulaere Hilfsmatrizen
// -----------------------------------------------------------------------------
// A_G ist bei sigma = 0 identisch null, A_Pi hat mindestens den
// dreidimensionalen Nullraum der konstanten Vektorfelder. Als
// VORKONDITIONIERER ist das zulaessig -- er muss nicht invertieren, nur
// annaehern und darf nicht entarten. Auf der groebsten Ebene wird deshalb mit
// Diagonalshift direkt geloest und bei Bedarf auf Jacobi zurueckgefallen.
// =============================================================================

#include <vector>

#include "DiscreteGradient.cuh"

class AlgebraicMultigrid {
public:
    struct Options {
        // Schwelle der starken Kopplung im Staerkegraphen.
        //
        // 0.08 ist der uebliche Wert fuer ISOTROPE Probleme. Bei gestreckten
        // Zellen ist er zu klein: dann gelten auch die schwachen Richtungen als
        // stark, die Aggregate werden isotrop gebildet, und das Verfahren
        // verliert seine Mehrgittereigenschaft. Gemessen am Poisson-Problem auf
        // einem 16x16x3-Gitter (Faktor je V-Zyklus, fein 1156):
        //
        //   Zell-Verh.  theta=0.08          theta=0.35
        //      1.0      0.237 (-> 42)       keine Vergroeberung
        //      4.5      0.774 (-> 40)       0.439 (-> 289)
        //     10.0      0.854 (-> 40)       0.478 (-> 289)
        //     30.0      0.873 (-> 41)       0.777 (-> 189)
        //
        // Bei Seitenverhaeltnis 4.5 -- dem Median der Aluminiumplatte im
        // TEAM-28-Netz -- halbiert ein groesseres theta den Faktor, weil die
        // Aggregate sich dann als Saeulen entlang der starken (duennen)
        // Richtung bilden. Das ist Semicoarsening und lehrbuchmaessig richtig.
        //
        // ACHTUNG, eine Falle: noch groesseres theta liefert scheinbar
        // traumhafte Faktoren (0.0013 bei theta = 0.7), aber dann gilt gar
        // nichts mehr als stark, das Grobgitter ist so gross wie das feine, und
        // das "AMG" ist eine teure direkte Loesung. Wer theta anfasst, MUSS die
        // Grobgittergroesse mitlesen.
        //
        // Ein einziger globaler Wert kann uebrigens nicht beiden Regimen
        // dienen: die Spulen haben Seitenverhaeltnis 1.4 bis 1.9, die Platte
        // 4.5. Richtig waere eine richtungsabhaengige Aggregation.
        /// Zeilenrelativer Staerkemaszstab statt des diagonalen. Siehe die
        /// ausfuehrliche Begruendung an strong_graph() in
        /// src/AlgebraicMultigrid.cpp. Kurz: der diagonale Maszstab kennt keine
        /// Richtungen und scheitert an gestreckten Zellen; der zeilenrelative
        /// normiert je Zeile auf die groesste Kopplung und richtet die
        /// Aggregate von selbst entlang der starken Richtung aus.
        ///
        /// Mit dieser Wahl ist theta ANDERS zu lesen: es ist ein Anteil der
        /// groessten Kopplung DER ZEILE, nicht des Diagonalelements. Gemessen
        /// am Poisson-Problem (16x16x3, Faktor je V-Zyklus, Grobgitter in
        /// Klammern, fein 1156):
        ///
        ///   Zell-Verh.  diagonal 0.08   zeilenrel. 0.60
        ///      1.0      0.237 (42)      0.273 (21)
        ///      4.5      0.774 (40)      0.439 (36)
        ///     10.0      0.854 (40)      0.478 (36)
        ///     30.0      0.873 (41)      0.490 (36)
        ///
        /// Eine einzige Einstellung fuer alle Seitenverhaeltnisse, und mit
        /// gesunder Vergroeberung. Der diagonale Maszstab braucht fuer
        /// vergleichbare Faktoren theta = 0.35 und kommt dann nur noch auf ein
        /// Grobgitter von 289 -- also fast keine Vergroeberung.
        ///
        /// Der isotrope Fall wird dabei minimal schlechter (0.273 gegen 0.237).
        /// Das ist der Preis und er ist klein gegen den Gewinn bei 4.5, dem
        /// Median der Aluminiumplatte im TEAM-28-Netz.
        bool   row_relative_strength = true;

        double theta          = 0.60;
        int    max_levels     = 20;
        int    coarsest_size  = 60;    // darunter wird direkt geloest
        int    pre_smooth     = 2;
        int    post_smooth    = 2;
        double diagonal_shift = 1e-10; // relativ, gegen singulaere Grobmatrizen
    };

    /// dofs_per_node: 1 fuer skalare Probleme (A_G), 3 fuer A_Pi.
    /// A muss symmetrische Sparsity und aufsteigende Spaltenindizes haben.
    // Options wird per Wert uebergeben; ein Default-Argument Options() waere
    // an dieser Stelle noch nicht bildbar, weil die Klasse selbst noch
    // unvollstaendig ist. Daher eine zweite Ueberladung.
    void setup(const HostCsr& A, int dofs_per_node, Options opt);
    void setup(const HostCsr& A, int dofs_per_node = 1) { setup(A, dofs_per_node, Options()); }

    /// Ein V-Zyklus, x wird mit null initialisiert. Wirkt damit als
    /// Vorkonditionierer x = B b.
    void apply(const double* b, double* x) const;

    int num_levels() const { return static_cast<int>(m_levels.size()); }

    /// Diagonalen, die als nicht vorhanden eingestuft wurden. Eine grosse Zahl
    /// ist keine Panne, sondern zu erwarten, wenn A_G = G^T A G ueber
    /// nichtleitende Gebiete laeuft -- dort ist der Eintrag rundungsklein.
    int num_tiny_diagonals() const { return m_num_tiny_diag; }
    int level_size(int l) const { return m_levels[l].A.num_rows; }
    /// Verhaeltnis der Nichtnullen aller Ebenen zu denen der feinsten --
    /// bei gesunder Vergroeberung unter etwa 2.
    double operator_complexity() const;
    bool   coarsest_direct_ok() const { return m_coarse_direct_ok; }

private:
    struct Level {
        HostCsr A;
        HostCsr P;                       // nur auf Ebenen < letzte
        std::vector<double> inv_diag;
        double omega = 0.0;              // Daempfung des Jacobi-Glaetters
        mutable std::vector<double> r, xc, rc;
    };

    void   smooth(const Level& lv, const double* b, double* x, int sweeps) const;
    void   vcycle(int level, const double* b, double* x) const;

    std::vector<Level>  m_levels;
    Options             m_opt;
    // Groebste Ebene: dichte LU mit Pivotisierung
    std::vector<double> m_coarse_lu;     // column-major
    std::vector<int>    m_coarse_piv;
    bool                m_coarse_direct_ok = false;

    /// Zahl der Diagonalen, die ueber alle Ebenen als nicht vorhanden
    /// eingestuft wurden (|d| <= 1e-10 * max|d| der jeweiligen Ebene).
    /// Siehe fill_inv_diag in src/AlgebraicMultigrid.cpp.
    int                 m_num_tiny_diag = 0;
    mutable std::vector<double> m_coarse_rhs;
};

#endif // ALGEBRAIC_MULTIGRID_CUH
