#include "AlgebraicMultigrid.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

/// Graph der starken Kopplungen auf KNOTENebene. Bei mehreren Freiheitsgraden
/// je Knoten wird blockweise zusammengefasst: als Kopplungsstaerke zwischen
/// den Knoten I und J dient der groesste Betrag im zugehoerigen Block.
// Staerkegraph der Aggregation.
//
// Zwei Maszstaebe, umschaltbar ueber Options::row_relative_strength:
//
//   DIAGONAL (alt):      |a_IJ|  >  theta * sqrt(|a_II * a_JJ|)
//   ZEILENRELATIV (neu): |a_IJ|  >  theta * max_{K != I} |a_IK|
//
// Der diagonale Maszstab ist absolut und kennt keine Richtungen. Bei
// gestreckten Zellen sind die Kopplungen aber richtungsabhaengig: quer zur
// duennen Richtung ist a_IJ um das Quadrat des Seitenverhaeltnisses kleiner.
// Mit einem festen theta gelten dann entweder alle Richtungen als stark (die
// Aggregate werden isotrop gebildet, das Verfahren verliert seine
// Mehrgittereigenschaft) oder keine (die Vergroeberung bricht ab). Gemessen am
// Poisson-Problem auf 16x16x3:
//
//   Zell-Verh.  theta=0.08          theta=0.35
//      1.0      0.237 (-> 42)       keine Vergroeberung
//      4.5      0.774 (-> 40)       0.439 (-> 289)
//
// Ein einziger Wert kann Platte (Seitenverhaeltnis 4.5) und Spulen (1.4 bis
// 1.9) also nicht zugleich bedienen.
//
// Der zeilenrelative Maszstab (klassisch nach Ruge und Stueben) normiert je
// Zeile auf die groesste Kopplung. Damit ist er skaleninvariant: im isotropen
// Fall sind alle Kopplungen aehnlich gross und bestehen die Schwelle, im
// anisotropen Fall besteht nur die starke Richtung. Die Aggregate richten sich
// von selbst aus -- Semicoarsening, ohne dass theta je Gebiet angepasst werden
// muss.
std::vector<std::vector<int>> strong_graph(const HostCsr& A, int bs, double theta,
                                           bool row_relative) {
    const int nnode = A.num_rows / bs;
    std::vector<double> blockmax(static_cast<size_t>(nnode), 0.0);   // Diagonalbloecke

    // Diagonalreferenz je Knoten: groesster Betrag im Diagonalblock
    for (int I = 0; I < nnode; ++I) {
        for (int c = 0; c < bs; ++c) {
            const int row = I * bs + c;
            for (int k = A.row_offsets[row]; k < A.row_offsets[row+1]; ++k) {
                if (A.col_indices[k] / bs == I)
                    blockmax[I] = std::fmax(blockmax[I], std::fabs(A.values[k]));
            }
        }
    }

    // Schwelle anwenden
    std::vector<std::vector<int>> S(nnode);
    for (int I = 0; I < nnode; ++I) {
        std::vector<std::pair<int,double>> acc;
        for (int c = 0; c < bs; ++c) {
            const int row = I * bs + c;
            for (int k = A.row_offsets[row]; k < A.row_offsets[row+1]; ++k) {
                const int J = A.col_indices[k] / bs;
                if (J == I) continue;
                acc.emplace_back(J, std::fabs(A.values[k]));
            }
        }
        std::sort(acc.begin(), acc.end());

        // Groesste Nichtdiagonalkopplung dieser Zeile -- der Bezug des
        // zeilenrelativen Maszstabs.
        double row_max = 0.0;
        for (const auto& kv : acc) row_max = std::fmax(row_max, kv.second);

        for (size_t q = 0; q < acc.size(); ) {
            size_t p = q; double m = 0.0;
            while (p < acc.size() && acc[p].first == acc[q].first) { m = std::fmax(m, acc[p].second); ++p; }
            const int J = acc[q].first;

            bool strong;
            if (row_relative) {
                // Skaleninvariant: nur die groessten Kopplungen der Zeile
                // gelten als stark. Bei row_max == 0 hat die Zeile ueberhaupt
                // keine Kopplung -- dann gibt es nichts zu aggregieren, und
                // "alles ist stark" waere die falsche Antwort.
                strong = (row_max > 0.0) && (m > theta * row_max);
            } else {
                const double ref = std::sqrt(blockmax[I] * blockmax[J]);
                strong = (m * m > theta * theta * ref * ref) || (ref == 0.0);
            }
            if (strong) S[I].push_back(J);
            q = p;
        }
    }
    return S;
}

/// Gieriges Dreipassverfahren. Rueckgabe: Aggregatnummer je Knoten, und die
/// Zahl der Aggregate.
int aggregate(const std::vector<std::vector<int>>& S, std::vector<int>& agg) {
    const int n = static_cast<int>(S.size());
    agg.assign(static_cast<size_t>(n), -1);
    int na = 0;

    // Pass 1: Wurzeln, deren gesamte starke Nachbarschaft noch frei ist
    for (int i = 0; i < n; ++i) {
        if (agg[i] != -1) continue;
        bool free_nb = true;
        for (int j : S[i]) if (agg[j] != -1) { free_nb = false; break; }
        if (!free_nb) continue;
        agg[i] = na;
        for (int j : S[i]) agg[j] = na;
        ++na;
    }
    // Pass 2: Reste an ein benachbartes Aggregat anhaengen
    for (int i = 0; i < n; ++i) {
        if (agg[i] != -1) continue;
        for (int j : S[i]) if (agg[j] != -1) { agg[i] = agg[j]; break; }
    }
    // Pass 3: was jetzt noch frei ist, bildet eigene Aggregate
    for (int i = 0; i < n; ++i) {
        if (agg[i] != -1) continue;
        agg[i] = na;
        for (int j : S[i]) if (agg[j] == -1) agg[j] = na;
        ++na;
    }
    return na;
}

/// C = A * B, beide CSR, Spalten je Zeile aufsteigend.
HostCsr spmm(const HostCsr& A, const HostCsr& B) {
    HostCsr C;
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.row_offsets.assign(A.num_rows + 1, 0);
    std::vector<double> scratch(static_cast<size_t>(B.num_cols), 0.0);
    std::vector<char>   hit(static_cast<size_t>(B.num_cols), 0);
    std::vector<int>    touched;

    for (int i = 0; i < A.num_rows; ++i) {
        C.row_offsets[i] = static_cast<int>(C.col_indices.size());
        for (int ka = A.row_offsets[i]; ka < A.row_offsets[i+1]; ++ka) {
            const int    k  = A.col_indices[ka];
            const double av = A.values[ka];
            if (av == 0.0) continue;
            for (int kb = B.row_offsets[k]; kb < B.row_offsets[k+1]; ++kb) {
                const int col = B.col_indices[kb];
                if (!hit[col]) { hit[col] = 1; touched.push_back(col); }
                scratch[col] += av * B.values[kb];
            }
        }
        std::sort(touched.begin(), touched.end());
        for (int col : touched) {
            if (scratch[col] != 0.0) { C.col_indices.push_back(col); C.values.push_back(scratch[col]); }
            scratch[col] = 0.0; hit[col] = 0;
        }
        touched.clear();
    }
    C.row_offsets[A.num_rows] = static_cast<int>(C.col_indices.size());
    return C;
}

HostCsr transpose(const HostCsr& A) {
    HostCsr T;
    T.num_rows = A.num_cols;
    T.num_cols = A.num_rows;
    T.row_offsets.assign(A.num_cols + 1, 0);
    for (int k = 0; k < A.nnz(); ++k) ++T.row_offsets[A.col_indices[k] + 1];
    for (int i = 0; i < A.num_cols; ++i) T.row_offsets[i+1] += T.row_offsets[i];
    T.col_indices.resize(A.nnz());
    T.values.resize(A.nnz());
    std::vector<int> pos(T.row_offsets.begin(), T.row_offsets.end() - 1);
    for (int i = 0; i < A.num_rows; ++i) {
        for (int k = A.row_offsets[i]; k < A.row_offsets[i+1]; ++k) {
            const int c = A.col_indices[k];
            T.col_indices[pos[c]] = i;
            T.values[pos[c]] = A.values[k];
            ++pos[c];
        }
    }
    return T;
}

} // namespace

double AlgebraicMultigrid::operator_complexity() const {
    if (m_levels.empty()) return 0.0;
    double tot = 0.0;
    for (const auto& lv : m_levels) tot += lv.A.nnz();
    return tot / std::fmax(1.0, static_cast<double>(m_levels[0].A.nnz()));
}

namespace {

// Relative Schranke, unterhalb derer eine Diagonale als "nicht vorhanden" gilt.
//
// Das ist der Kern eines Fehlers, der den AMS-Zweig unbrauchbar gemacht hat.
// Die Pruefung lautete ueberall "(d != 0.0) ? 1.0/d : 0.0" -- sie faengt also
// nur die EXAKTE Null. Der gefaehrliche Fall ist aber die winzige,
// rundungsbedingte Diagonale: 1e-20 ergibt inv_diag = 1e+20, und der
// Jacobi-Glaetter multipliziert das Residuum damit.
//
// Und diese Diagonalen entstehen hier systematisch, nicht zufaellig. Die
// skalare Hilfsmatrix ist A_G = G^T A G. Fuer den Curl-Curl-Anteil gilt
// A_cc * G = 0 EXAKT in der Analysis -- numerisch bleibt davon Rundung
// stehen. Nur wo sigma > 0 ist (die Aluminiumplatte), traegt der Massenterm
// eine echte Diagonale bei. In Spulen und Luft (kappa = 0) steht damit
// Rundungsrauschen der Groessenordnung eps * ||A|| auf der Diagonalen, und
// genau dort explodierte der Glaetter. Gemessen: ||z_A|| = 8.6e+14 bei
// ||r_A|| = 1.
//
// Die Zaehlung "888 mit LEERER Diagonale" konnte das nicht zeigen, weil auch
// sie auf d == 0.0 prueft. Exakt null werden nur die Knoten, bei denen sich
// alles ausloescht; die vielen rundungskleinen bleiben unsichtbar.
//
// Wird eine Diagonale als nicht vorhanden eingestuft, ist inv_diag = 0. Der
// Jacobi-Glaetter laesst diese Zeile dann unveraendert, statt sie zu
// zerstoeren. Das ist auch inhaltlich richtig: wo kein sigma ist, hat der
// skalare Hilfsraum nichts zu korrigieren.
constexpr double kTinyDiagonalRatio = 1e-10;

/// Fuellt inv_diag und liefert die Gershgorin-Schranke lam sowie die Zahl der
/// als nicht vorhanden eingestuften Diagonalen.
int fill_inv_diag(const HostCsr& A, std::vector<double>& inv_diag, double& lam) {
    const int n = A.num_rows;
    inv_diag.assign(static_cast<size_t>(n), 0.0);

    // Erst den Massstab bestimmen -- "winzig" ist nur relativ zu etwas.
    double dmax = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int k = A.row_offsets[i]; k < A.row_offsets[i+1]; ++k) {
            if (A.col_indices[k] == i) {
                dmax = std::fmax(dmax, std::fabs(A.values[k]));
                break;
            }
        }
    }
    const double tiny = kTinyDiagonalRatio * dmax;

    lam = 0.0;
    int num_tiny = 0;
    for (int i = 0; i < n; ++i) {
        double d = 0.0, rowsum = 0.0;
        for (int k = A.row_offsets[i]; k < A.row_offsets[i+1]; ++k) {
            if (A.col_indices[k] == i) d = A.values[k];
            rowsum += std::fabs(A.values[k]);
        }
        if (std::fabs(d) > tiny) {
            inv_diag[i] = 1.0 / d;
            lam = std::fmax(lam, rowsum / std::fabs(d));   // Gershgorin auf D^-1 A
        } else {
            inv_diag[i] = 0.0;
            ++num_tiny;
        }
    }
    return num_tiny;
}

} // namespace

void AlgebraicMultigrid::setup(const HostCsr& A, int dofs_per_node, Options opt) {
    m_opt = opt;
    m_levels.clear();
    m_coarse_direct_ok = false;
    m_num_tiny_diag = 0;

    m_levels.emplace_back();
    m_levels.back().A = A;

    for (int l = 0; l < opt.max_levels - 1; ++l) {
        Level& lv = m_levels[l];
        const int n = lv.A.num_rows;

        // Diagonale und Daempfung. Siehe fill_inv_diag oben -- die Schranke
        // ist RELATIV, nicht "ungleich null".
        double lam = 0.0;
        m_num_tiny_diag += fill_inv_diag(lv.A, lv.inv_diag, lam);
        if (lam <= 0.0) lam = 2.0;
        lv.omega = 1.0 / lam;             // gedaempfter Jacobi

        if (n <= opt.coarsest_size) break;

        // Aggregation auf Knotenebene (bs = dofs_per_node nur auf der feinsten
        // Ebene; darunter hat jedes Aggregat wieder bs Freiheitsgrade)
        const int bs = dofs_per_node;
        if (n % bs != 0) break;           // sollte nicht vorkommen
        auto S = strong_graph(lv.A, bs, opt.theta, opt.row_relative_strength);
        std::vector<int> agg;
        const int na = aggregate(S, agg);
        if (na <= 0 || na * bs >= n) break;   // keine Vergroeberung erreicht

        // Vorlaeufige Prolongation, spaltenweise normiert
        std::vector<int> agg_size(static_cast<size_t>(na), 0);
        for (int i : agg) ++agg_size[i];
        HostCsr Pt;
        Pt.num_rows = n;
        Pt.num_cols = na * bs;
        Pt.row_offsets.assign(n + 1, 0);
        Pt.col_indices.resize(n);
        Pt.values.resize(n);
        for (int I = 0; I < n / bs; ++I) {
            for (int c = 0; c < bs; ++c) {
                const int row = I * bs + c;
                Pt.row_offsets[row] = row;
                Pt.col_indices[row] = agg[I] * bs + c;
                Pt.values[row] = 1.0 / std::sqrt(static_cast<double>(agg_size[agg[I]]));
            }
        }
        Pt.row_offsets[n] = n;

        // Glaettung: P = P_tent - omega_s D^-1 A P_tent,  omega_s = 4/(3 lambda)
        const double omega_s = 4.0 / (3.0 * lam);
        HostCsr AP = spmm(lv.A, Pt);
        HostCsr P;
        P.num_rows = n; P.num_cols = Pt.num_cols;
        P.row_offsets.assign(n + 1, 0);
        {
            std::vector<double> scratch(static_cast<size_t>(P.num_cols), 0.0);
            std::vector<char>   hit(static_cast<size_t>(P.num_cols), 0);
            std::vector<int>    touched;
            for (int i = 0; i < n; ++i) {
                P.row_offsets[i] = static_cast<int>(P.col_indices.size());
                for (int k = Pt.row_offsets[i]; k < Pt.row_offsets[i+1]; ++k) {
                    const int c = Pt.col_indices[k];
                    if (!hit[c]) { hit[c]=1; touched.push_back(c); }
                    scratch[c] += Pt.values[k];
                }
                const double f = -omega_s * lv.inv_diag[i];
                for (int k = AP.row_offsets[i]; k < AP.row_offsets[i+1]; ++k) {
                    const int c = AP.col_indices[k];
                    if (!hit[c]) { hit[c]=1; touched.push_back(c); }
                    scratch[c] += f * AP.values[k];
                }
                std::sort(touched.begin(), touched.end());
                for (int c : touched) {
                    if (scratch[c] != 0.0) { P.col_indices.push_back(c); P.values.push_back(scratch[c]); }
                    scratch[c] = 0.0; hit[c] = 0;
                }
                touched.clear();
            }
            P.row_offsets[n] = static_cast<int>(P.col_indices.size());
        }

        // Galerkin
        HostCsr Pt_T = transpose(P);
        HostCsr Ac   = spmm(Pt_T, spmm(lv.A, P));

        lv.P = P;
        lv.r.assign(static_cast<size_t>(n), 0.0);
        lv.rc.assign(static_cast<size_t>(Ac.num_rows), 0.0);
        lv.xc.assign(static_cast<size_t>(Ac.num_rows), 0.0);

        m_levels.emplace_back();
        m_levels.back().A = std::move(Ac);
    }

    // Groebste Ebene: dichte LU mit Diagonalshift
    Level& last = m_levels.back();
    const int nc = last.A.num_rows;
    if (last.inv_diag.empty()) {
        double lam = 0.0;
        m_num_tiny_diag += fill_inv_diag(last.A, last.inv_diag, lam);
        last.omega = (lam > 0.0) ? 1.0/lam : 0.5;
    }

    if (nc > 0 && nc <= 4000) {
        m_coarse_lu.assign(static_cast<size_t>(nc) * nc, 0.0);
        double dmax = 0.0;
        for (int i = 0; i < nc; ++i)
            for (int k = last.A.row_offsets[i]; k < last.A.row_offsets[i+1]; ++k) {
                m_coarse_lu[static_cast<size_t>(last.A.col_indices[k]) * nc + i] = last.A.values[k];
                if (last.A.col_indices[k] == i) dmax = std::fmax(dmax, std::fabs(last.A.values[k]));
            }
        const double eps = m_opt.diagonal_shift * (dmax > 0.0 ? dmax : 1.0);
        for (int i = 0; i < nc; ++i) m_coarse_lu[static_cast<size_t>(i)*nc + i] += eps;

        // LU mit Spaltenpivotisierung
        m_coarse_piv.assign(static_cast<size_t>(nc), 0);
        double lu_max = 0.0;
        for (double v : m_coarse_lu) lu_max = std::fmax(lu_max, std::fabs(v));
        const double coarse_pivot_floor = 1e-14 * (lu_max > 0.0 ? lu_max : 1.0);

        m_coarse_direct_ok = true;
        for (int k = 0; k < nc && m_coarse_direct_ok; ++k) {
            int piv = k; double best = std::fabs(m_coarse_lu[static_cast<size_t>(k)*nc + k]);
            for (int i = k+1; i < nc; ++i) {
                const double v = std::fabs(m_coarse_lu[static_cast<size_t>(k)*nc + i]);
                if (v > best) { best = v; piv = i; }
            }
            // Siehe DenseLinalg: "== 0.0" laesst winzige und nicht endliche
            // Pivots durch. Faellt die direkte Loesung aus, uebernimmt der
            // Glaetter auf der groebsten Ebene -- das ist zulaessig und wird
            // unten ueber m_coarse_direct_ok abgefangen.
            if (!std::isfinite(best) || best <= coarse_pivot_floor) {
                m_coarse_direct_ok = false; break;
            }
            m_coarse_piv[k] = piv;
            if (piv != k) for (int j = 0; j < nc; ++j)
                std::swap(m_coarse_lu[static_cast<size_t>(j)*nc + k],
                          m_coarse_lu[static_cast<size_t>(j)*nc + piv]);
            const double ip = 1.0 / m_coarse_lu[static_cast<size_t>(k)*nc + k];
            for (int i = k+1; i < nc; ++i) m_coarse_lu[static_cast<size_t>(k)*nc + i] *= ip;
            for (int j = k+1; j < nc; ++j) {
                const double akj = m_coarse_lu[static_cast<size_t>(j)*nc + k];
                if (akj == 0.0) continue;
                for (int i = k+1; i < nc; ++i)
                    m_coarse_lu[static_cast<size_t>(j)*nc + i] -=
                        m_coarse_lu[static_cast<size_t>(k)*nc + i] * akj;
            }
        }
        if (!m_coarse_direct_ok) { m_coarse_lu.clear(); m_coarse_piv.clear(); }
    }
    m_coarse_rhs.assign(static_cast<size_t>(nc), 0.0);
}

void AlgebraicMultigrid::smooth(const Level& lv, const double* b, double* x, int sweeps) const {
    const int n = lv.A.num_rows;
    std::vector<double> t(static_cast<size_t>(n));
    for (int s = 0; s < sweeps; ++s) {
        apply_host_csr(lv.A, x, t.data());
        for (int i = 0; i < n; ++i) x[i] += lv.omega * lv.inv_diag[i] * (b[i] - t[i]);
    }
}

void AlgebraicMultigrid::vcycle(int level, const double* b, double* x) const {
    const Level& lv = m_levels[level];
    const int n = lv.A.num_rows;

    if (level + 1 == static_cast<int>(m_levels.size())) {
        if (m_coarse_direct_ok) {
            std::copy(b, b + n, m_coarse_rhs.begin());
            double* y = m_coarse_rhs.data();
            for (int k = 0; k < n; ++k) { const int p = m_coarse_piv[k]; if (p != k) std::swap(y[k], y[p]); }
            for (int i = 1; i < n; ++i) {
                double s = y[i];
                for (int k = 0; k < i; ++k) s -= m_coarse_lu[static_cast<size_t>(k)*n + i] * y[k];
                y[i] = s;
            }
            for (int i = n-1; i >= 0; --i) {
                double s = y[i];
                for (int k = i+1; k < n; ++k) s -= m_coarse_lu[static_cast<size_t>(k)*n + i] * y[k];
                y[i] = s / m_coarse_lu[static_cast<size_t>(i)*n + i];
            }
            std::copy(m_coarse_rhs.begin(), m_coarse_rhs.begin() + n, x);
        } else {
            std::fill(x, x + n, 0.0);
            smooth(lv, b, x, 20);
        }
        return;
    }

    std::fill(x, x + n, 0.0);
    smooth(lv, b, x, m_opt.pre_smooth);

    apply_host_csr(lv.A, x, lv.r.data());
    for (int i = 0; i < n; ++i) lv.r[i] = b[i] - lv.r[i];

    const int nc = lv.P.num_cols;
    std::fill(lv.rc.begin(), lv.rc.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        const double ri = lv.r[i];
        if (ri == 0.0) continue;
        for (int k = lv.P.row_offsets[i]; k < lv.P.row_offsets[i+1]; ++k)
            lv.rc[lv.P.col_indices[k]] += lv.P.values[k] * ri;
    }

    std::fill(lv.xc.begin(), lv.xc.begin() + nc, 0.0);
    vcycle(level + 1, lv.rc.data(), lv.xc.data());

    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int k = lv.P.row_offsets[i]; k < lv.P.row_offsets[i+1]; ++k)
            acc += lv.P.values[k] * lv.xc[lv.P.col_indices[k]];
        x[i] += acc;
    }

    smooth(lv, b, x, m_opt.post_smooth);
}

void AlgebraicMultigrid::apply(const double* b, double* x) const {
    if (m_levels.empty()) return;
    vcycle(0, b, x);
}
