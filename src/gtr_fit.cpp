/*
 * gtr_fit.cpp
 *
 * The GTR+GC fit (gtr_fit.hpp): in order, the model's ten cell probabilities, one pair's distance, the objective, BFGS,
 * Brent's method, the fit, and the pairs it uses.
 * The model, the pair's distance, the objective and BFGS keep the order of operations of the paper's code, and the
 * numerical constants below its values (no reason for them is recorded); the search over alpha is new (gtr_fit.hpp).
 */
#include "gtr_fit.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "random.hpp"
#include "timing.hpp"

namespace p2vbme {

namespace {

// ---- Constants ---------------------------------------------------------------------------------------------

// The ten cells i <= j of a symmetric 4 x 4 table, and the three non-trivial modes of the rate matrix.
constexpr int N_CELLS = N_BASES * (N_BASES + 1) / 2;
constexpr int N_MODES = N_BASES - 1;
// The parameters theta: the five free log exchangeability ratios log(S_k / S_GT), then log alpha.
constexpr int N_FREE_RATES = N_GTR_RATES - 1;
constexpr int N_PARAMS = N_FREE_RATES + 1;
constexpr int LOG_ALPHA = N_FREE_RATES; // index of log alpha in theta

// The cells' bases (i, j), and the exchangeabilities' (AC, AG, AT, CG, CT, GT).
constexpr int CELL_I[N_CELLS] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3}, CELL_J[N_CELLS] = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
constexpr int RATE_I[N_GTR_RATES] = {0, 0, 0, 1, 1, 2}, RATE_J[N_GTR_RATES] = {1, 2, 3, 2, 3, 3};

// Jacobi eigensolver: sweeps at most, and the squared off-diagonal norm (and the single element) treated as zero.
constexpr int JACOBI_MAX_SWEEPS = 60;
constexpr double JACOBI_OFF_TOL = 1e-30;
constexpr double JACOBI_ELEMENT_TOL = 1e-300;
// Floor of the mean rate before the rate matrix is divided by it, and of a cell probability before its log.
constexpr double TINY = 1e-300;

// One pair's distance: Newton iterations at most, their relative stopping tolerance, the start when the previous
// distance is out of range, and the Jukes-Cantor start of the first evaluation (p below JC_START_P_MAX, else
// JC_START_FAR; at least JC_START_MIN).
constexpr int NEWTON_MAX_ITERATIONS = 100;
constexpr double NEWTON_TOL = 1e-12;
constexpr double NEWTON_D_START = 0.1;
constexpr double JC_START_P_MAX = 0.7;
constexpr double JC_START_FAR = 3.0;
constexpr double JC_START_MIN = 1e-4;
// Floor of each pooled base frequency.
constexpr double PI_MIN = 1e-12;

// BFGS: the central-difference step of the gradient, the Armijo constant and halvings of the line search, the
// curvature condition's relative tolerance, and the stop: STALL_ITERATIONS successive gains below
// STALL_REL |f| + STALL_ABS, once the curvature estimate holds nd + 2 updates.
constexpr double GRADIENT_STEP = 1e-4;
constexpr double ARMIJO = 1e-4;
constexpr int LINE_SEARCH_HALVINGS = 40;
constexpr double CURVATURE_TOL = 1e-10;
constexpr double STALL_REL = 1e-12;
constexpr double STALL_ABS = 1e-9;
constexpr int STALL_ITERATIONS = 2;

// The start: the alpha values scanned, BFGS iterations per value, and the tables the scan uses (the first ones).
constexpr double START_ALPHAS[] = {148.0, 16.0, 6.0, 3.0, 1.6, 0.9, 0.5, 0.25, 0.1};
constexpr int START_MAX_ITERATIONS = 60;
constexpr int32_t START_PAIRS = 20000;
// Pairs per thread, at least, in each parallel loop.
// Measured on a 64-core machine (128 hardware threads) under load, all threads allowed: a fit of 100,000 pairs took
// 0.48 s every time with 1,000, 0.37-0.74 s with 250 or 500, and 0.77-3 s with no minimum.
// Small inputs would gain from a lower minimum (4,000 pairs: 0.13 s with 250, 0.43 s with 1,000), but stay below 0.5 s.
constexpr int64_t MIN_PAIRS_PER_THREAD = 1000;

// BFGS iterations of each search of the exchangeabilities, at most.
constexpr int MAX_ITERATIONS = 200;

// The search over log alpha (Brent's method): the width of the final bracket (0.1% in alpha, the paper's check's
// tolerance), its iterations at most, the step that widens the start bracket, and the golden-section ratio (3 - sqrt 5)
// / 2 of its fallback steps.
constexpr double LOG_ALPHA_TOL = 1e-3;
constexpr int BRENT_MAX_ITERATIONS = 100;
constexpr double LOG_ALPHA_WIDEN = 1.0;
constexpr double GOLDEN_SECTION = 0.3819660112501051;

// ---- The model's cell probabilities ----------------------------------------------------------------------

// The model at one theta: F_c(d) = base[c] + sum_k w[c][k] m_k(d), m_k(d) = (1 - lam[k] d / alpha)^(-alpha).
struct CellModel {
    double lam[N_MODES], w[N_CELLS][N_MODES], base[N_CELLS], alpha;
};

// Diagonalises a symmetric 4 x 4 matrix by cyclic Jacobi rotations: a -> (eval, v), v's columns the eigenvectors.
// a is overwritten.
void jacobi4(double a[16], double eval[4], double v[16]) {
    for (int i = 0; i < 16; ++i)
        v[i] = (i % (N_BASES + 1) == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < JACOBI_MAX_SWEEPS; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q)
                off += a[p * 4 + q] * a[p * 4 + q];
        if (off < JACOBI_OFF_TOL)
            break;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q) {
                const double apq = a[p * 4 + q];
                if (std::fabs(apq) < JACOBI_ELEMENT_TOL)
                    continue;
                const double tau = (a[q * 4 + q] - a[p * 4 + p]) / (2.0 * apq);
                const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                for (int r = 0; r < 4; ++r) {
                    const double arp = a[r * 4 + p], arq = a[r * 4 + q];
                    a[r * 4 + p] = c * arp - s * arq;
                    a[r * 4 + q] = s * arp + c * arq;
                }
                for (int r = 0; r < 4; ++r) {
                    const double apr = a[p * 4 + r], aqr = a[q * 4 + r];
                    a[p * 4 + r] = c * apr - s * aqr;
                    a[q * 4 + r] = s * apr + c * aqr;
                }
                for (int r = 0; r < 4; ++r) {
                    const double vrp = v[r * 4 + p], vrq = v[r * 4 + q];
                    v[r * 4 + p] = c * vrp - s * vrq;
                    v[r * 4 + q] = s * vrp + c * vrq;
                }
            }
    }
    for (int i = 0; i < 4; ++i)
        eval[i] = a[i * (N_BASES + 1)];
}

// Reads the parameters: theta -> (s, alpha), each clamped to its range and the exchangeabilities scaled to mean 1.
void unpack(const double theta[N_PARAMS], double s[N_GTR_RATES], double *alpha) {
    double sum = 0.0;
    for (int k = 0; k < N_GTR_RATES; ++k) {
        double t = (k < N_FREE_RATES) ? theta[k] : 0.0; // S_GT is the reference, log ratio 0
        if (t > GTR_FIT_LOG_RATE_MAX)
            t = GTR_FIT_LOG_RATE_MAX;
        if (t < -GTR_FIT_LOG_RATE_MAX)
            t = -GTR_FIT_LOG_RATE_MAX;
        s[k] = std::exp(t);
        sum += s[k];
    }
    for (int k = 0; k < N_GTR_RATES; ++k)
        s[k] *= (double)N_GTR_RATES / sum;
    double la = theta[LOG_ALPHA];
    if (la > GTR_FIT_LOG_ALPHA_MAX)
        la = GTR_FIT_LOG_ALPHA_MAX;
    if (la < GTR_FIT_LOG_ALPHA_MIN)
        la = GTR_FIT_LOG_ALPHA_MIN;
    *alpha = std::exp(la);
}

// Builds the model at theta: (theta, pi) -> md, from the eigenpairs of the unit-rate Pi^(1/2) Q Pi^(-1/2).
void build_model(const double theta[N_PARAMS], const double pi[N_BASES], CellModel *md) {
    double s[N_GTR_RATES], b[16], eval[4], u[16], mu = 0.0;
    unpack(theta, s, &md->alpha);
    std::memset(b, 0, sizeof(b));
    for (int k = 0; k < N_GTR_RATES; ++k) {
        const int i = RATE_I[k], j = RATE_J[k];
        b[i * 4 + j] = b[j * 4 + i] = s[k] * std::sqrt(pi[i] * pi[j]);
        b[i * (N_BASES + 1)] -= s[k] * pi[j];
        b[j * (N_BASES + 1)] -= s[k] * pi[i];
        mu += 2.0 * s[k] * pi[i] * pi[j];
    }
    if (mu < TINY)
        mu = TINY;
    for (int i = 0; i < 16; ++i)
        b[i] /= mu;
    jacobi4(b, eval, u);
    int triv = 0; // the zero eigenvalue: the largest
    for (int k = 1; k < 4; ++k)
        if (eval[k] > eval[triv])
            triv = k;
    for (int c = 0; c < N_CELLS; ++c)
        md->base[c] = pi[CELL_I[c]] * pi[CELL_J[c]];
    for (int k = 0, m = 0; k < 4; ++k) {
        if (k == triv)
            continue;
        md->lam[m] = (eval[k] < 0.0) ? eval[k] : 0.0;
        for (int c = 0; c < N_CELLS; ++c)
            md->w[c][m] = std::sqrt(md->base[c]) * u[CELL_I[c] * 4 + k] * u[CELL_J[c] * 4 + k];
        ++m;
    }
}

// Evaluates the ten cells at distance d: (md, d) -> f, and, if f1 is given, the first and second derivatives in d.
void cells(const CellModel *md, double d, double f[N_CELLS], double f1[N_CELLS], double f2[N_CELLS]) {
    double m[N_MODES], m1[N_MODES], m2[N_MODES];
    for (int k = 0; k < N_MODES; ++k) {
        const double base = 1.0 - md->lam[k] * d / md->alpha;
        m[k] = std::pow(base, -md->alpha);
        m1[k] = md->lam[k] * m[k] / base;
        m2[k] = md->lam[k] * md->lam[k] * (1.0 + 1.0 / md->alpha) * m[k] / (base * base);
    }
    for (int c = 0; c < N_CELLS; ++c) {
        const double *w = md->w[c];
        const double v = md->base[c] + w[0] * m[0] + w[1] * m[1] + w[2] * m[2];
        f[c] = (v > TINY) ? v : TINY;
        if (f1) {
            f1[c] = w[0] * m1[0] + w[1] * m1[1] + w[2] * m1[2];
            f2[c] = w[0] * m2[0] + w[1] * m2[1] + w[2] * m2[2];
        }
    }
}

// ---- One pair ----------------------------------------------------------------------------------------------

// Computes one pair's log-likelihood: (md, cell counts c, d) -> sum_c c ln F_c(d).
double pair_loglik(const CellModel *md, const double *c, double d) {
    double f[N_CELLS], ll = 0.0;
    cells(md, d, f, nullptr, nullptr);
    for (int k = 0; k < N_CELLS; ++k)
        if (c[k] > 0.0)
            ll += c[k] * std::log(f[k]);
    return ll;
}

// Maximises one pair's log-likelihood over d: (md, c, *d_io) -> the maximum, and its d in *d_io (also the start).
// Newton steps, bisected (geometric mean) when they leave the bracket the gradient's sign keeps.
// A pair without a difference sits at GTR_FIT_D_MIN.
double pair_solve(const CellModel *md, const double *c, double *d_io) {
    double lo = GTR_FIT_D_MIN, hi = GTR_FIT_D_MAX, d = *d_io;
    if (c[1] + c[2] + c[3] + c[5] + c[6] + c[8] <= 0.0) { // the off-diagonal cells
        *d_io = lo;
        return pair_loglik(md, c, lo);
    }
    if (!(d > lo && d < hi))
        d = NEWTON_D_START;
    for (int it = 0; it < NEWTON_MAX_ITERATIONS; ++it) {
        double f[N_CELLS], f1[N_CELLS], f2[N_CELLS], g = 0.0, h = 0.0;
        cells(md, d, f, f1, f2);
        for (int k = 0; k < N_CELLS; ++k)
            if (c[k] > 0.0) {
                const double r = f1[k] / f[k];
                g += c[k] * r;
                h += c[k] * (f2[k] / f[k] - r * r);
            }
        if (g > 0.0)
            lo = d;
        else
            hi = d;
        double dn = (h < 0.0) ? d - g / h : -1.0;
        if (!(dn > lo && dn < hi))
            dn = std::sqrt(lo * hi);
        const double step = std::fabs(dn - d);
        d = dn;
        if (step <= NEWTON_TOL * d || hi - lo <= NEWTON_TOL * hi)
            break;
    }
    *d_io = d;
    return pair_loglik(md, c, d);
}

// ---- The objective and its search ---------------------------------------------------------------------------

// Threads for a loop over `items` pairs: at most `threads`, and at least MIN_PAIRS_PER_THREAD pairs each (as
// device_cpu.cpp's threads_for).
int threads_for(int threads, int64_t items) {
    return (int)std::max<int64_t>(1, std::min<int64_t>(threads, items / MIN_PAIRS_PER_THREAD));
}

// The pairs' data: their ten cell counts, their current distances and a buffer for their log-likelihoods, and the
// threads the fit may use.
struct Pairs {
    std::vector<double> cells; // 10 per pair
    std::vector<double> dist;
    std::vector<double> ll;
    int threads = 1;
};

// Computes the penalised composite log-likelihood: (theta, pi, the first n pairs) -> objective.
// solve: re-optimise every distance (warm start from the current ones); otherwise they stay fixed.
// The pairs' terms are added in index order, so the value does not depend on the number of threads.
double objective(const double theta[N_PARAMS], const double pi[N_BASES], Pairs &pairs, int32_t n, bool solve) {
    CellModel md;
    double total = 0.0;
    build_model(theta, pi, &md);
    const double *all_cells = pairs.cells.data();
    double *dist = pairs.dist.data(), *ll = pairs.ll.data();
#pragma omp parallel for schedule(static) num_threads(threads_for(pairs.threads, n))
    for (int32_t p = 0; p < n; ++p) {
        const double *c = &all_cells[(size_t)p * N_CELLS];
        ll[p] = solve ? pair_solve(&md, c, &dist[p]) : pair_loglik(&md, c, dist[p]);
    }
    for (int32_t p = 0; p < n; ++p)
        total += ll[p];
    return total - GTR_FIT_RATE_VARIANCE_PENALTY / md.alpha;
}

// Computes the gradient in the first nd parameters by central differences at fixed distances: theta -> grad.
void gradient(const double theta[N_PARAMS], int nd, const double pi[N_BASES], Pairs &pairs, int32_t n,
              double grad[N_PARAMS]) {
    for (int k = 0; k < nd; ++k) {
        double tp[N_PARAMS], tm[N_PARAMS];
        std::memcpy(tp, theta, sizeof(tp));
        std::memcpy(tm, theta, sizeof(tm));
        tp[k] += GRADIENT_STEP;
        tm[k] -= GRADIENT_STEP;
        grad[k] = (objective(tp, pi, pairs, n, false) - objective(tm, pi, pairs, n, false)) / (2.0 * GRADIENT_STEP);
    }
}

// Maximises the objective over the first nd parameters by BFGS ascent: theta -> theta (in place), *f_out its value.
// The fit uses nd = N_FREE_RATES: the exchangeabilities at fixed alpha.
// Stops after STALL_ITERATIONS successive iterations that gain less than STALL_REL |f| + STALL_ABS, counted only once
// the curvature estimate holds nd + 2 updates (before that, a weakly determined alpha moves too slowly to tell flat
// from converged), or when no ascent step exists.
// Returns true if it stopped that way, false if it used all max_iters iterations.
bool bfgs(double theta[N_PARAMS], int nd, const double pi[N_BASES], Pairs &pairs, int32_t n, int max_iters,
          double *f_out, int *iters_out) {
    double grad[N_PARAMS], H[N_PARAMS * N_PARAMS], f = objective(theta, pi, pairs, n, true);
    int fresh = 1, small = 0, iter = 0, done = 0, updates = 0;
    gradient(theta, nd, pi, pairs, n, grad);
    for (; iter < max_iters && !done; ++iter) {
        double p[N_PARAMS], dir = 0.0, pmax = 0.0, gmax = 0.0;
        for (int i = 0; i < nd; ++i)
            if (std::fabs(grad[i]) > gmax)
                gmax = std::fabs(grad[i]);
        if (gmax <= 0.0) {
            done = 1;
            break;
        }
        if (fresh) { // steepest ascent, scaled so that the largest step is 0.5
            updates = 0;
            for (int i = 0; i < N_PARAMS * N_PARAMS; ++i)
                H[i] = (i % (N_PARAMS + 1) == 0) ? 0.5 / gmax : 0.0;
        }
        for (int i = 0; i < nd; ++i) {
            p[i] = 0.0;
            for (int j = 0; j < nd; ++j)
                p[i] += H[i * N_PARAMS + j] * grad[j];
            dir += p[i] * grad[i];
            if (std::fabs(p[i]) > pmax)
                pmax = std::fabs(p[i]);
        }
        if (!(dir > 0.0)) {
            if (fresh) {
                done = 1;
                break;
            }
            fresh = 1;
            continue;
        }
        double step = (pmax > 1.0) ? 1.0 / pmax : 1.0, tn[N_PARAMS], fn = f;
        int ok = 0;
        std::memcpy(tn, theta, sizeof(tn));
        for (int ls = 0; ls < LINE_SEARCH_HALVINGS && !ok; ++ls, step *= 0.5) {
            for (int i = 0; i < nd; ++i)
                tn[i] = theta[i] + step * p[i];
            fn = objective(tn, pi, pairs, n, true);
            ok = (fn >= f + ARMIJO * step * dir);
        }
        if (!ok) { // no ascent along this direction: retry once from steepest ascent, then stop
            f = objective(theta, pi, pairs, n, true);
            if (fresh) {
                done = 1;
                break;
            }
            fresh = 1;
            continue;
        }
        double gn[N_PARAMS], s[N_PARAMS], y[N_PARAMS], sy = 0.0, yy = 0.0, ss = 0.0;
        gradient(tn, nd, pi, pairs, n, gn);
        for (int i = 0; i < nd; ++i) {
            s[i] = tn[i] - theta[i];
            y[i] = grad[i] - gn[i];
            sy += s[i] * y[i];
            yy += y[i] * y[i];
            ss += s[i] * s[i];
        }
        if (sy > CURVATURE_TOL * std::sqrt(ss * yy)) { // curvature condition; otherwise H is kept
            double Hy[N_PARAMS], yHy = 0.0;
            if (fresh)
                for (int i = 0; i < N_PARAMS * N_PARAMS; ++i)
                    H[i] = (i % (N_PARAMS + 1) == 0) ? sy / yy : 0.0;
            for (int i = 0; i < nd; ++i) {
                Hy[i] = 0.0;
                for (int j = 0; j < nd; ++j)
                    Hy[i] += H[i * N_PARAMS + j] * y[j];
                yHy += y[i] * Hy[i];
            }
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j)
                    H[i * N_PARAMS + j] += (sy + yHy) / (sy * sy) * s[i] * s[j] - (Hy[i] * s[j] + s[i] * Hy[j]) / sy;
            fresh = 0;
            ++updates;
        }
        small = (fn - f <= STALL_REL * std::fabs(f) + STALL_ABS && updates >= nd + 2) ? small + 1 : 0;
        std::memcpy(theta, tn, sizeof(tn));
        std::memcpy(grad, gn, sizeof(grad));
        f = fn;
        if (small >= STALL_ITERATIONS)
            done = 1;
    }
    *f_out = f;
    if (iters_out)
        *iters_out = iter;
    return done != 0;
}

// Maximises g over [lo, hi] by Brent's method: (g, lo, x, hi) -> the maximum's position, to within tol / 2.
// x is inside [lo, hi] with g(x) = gx.
// Steps are parabolic through the three best points when that is safe, golden-section otherwise.
// *converged is false if BRENT_MAX_ITERATIONS ran out first.
template <class G> double brent_max(G &&g, double lo, double x, double gx, double hi, double tol, bool *converged) {
    double w = x, v = x, gw = gx, gv = gx, d = 0.0, e = 0.0;
    const double tol1 = 0.5 * tol, tol2 = 2.0 * tol1;
    *converged = false;
    for (int it = 0; it < BRENT_MAX_ITERATIONS; ++it) {
        const double xm = 0.5 * (lo + hi);
        if (std::fabs(x - xm) <= tol2 - 0.5 * (hi - lo)) {
            *converged = true;
            break;
        }
        bool golden = true;
        if (std::fabs(e) > tol1) { // the vertex of the parabola through (x, gx), (w, gw), (v, gv)
            const double r = (x - w) * (gx - gv);
            double q = (x - v) * (gx - gw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) // then the step p / q is the vertex's offset from x (the same for a maximum as a minimum)
                p = -p;
            q = std::fabs(q);
            const double e_prev = e;
            e = d;
            if (std::fabs(p) < std::fabs(0.5 * q * e_prev) && p > q * (lo - x) && p < q * (hi - x)) {
                d = p / q;
                const double u = x + d;
                if (u - lo < tol2 || hi - u < tol2)
                    d = (xm >= x) ? tol1 : -tol1;
                golden = false;
            }
        }
        if (golden) {
            e = (x >= xm) ? lo - x : hi - x;
            d = GOLDEN_SECTION * e;
        }
        const double u = (std::fabs(d) >= tol1) ? x + d : x + (d >= 0.0 ? tol1 : -tol1);
        const double gu = g(u);
        if (gu >= gx) {
            if (u >= x)
                lo = x;
            else
                hi = x;
            v = w, gv = gw, w = x, gw = gx, x = u, gx = gu;
        } else {
            if (u < x)
                lo = u;
            else
                hi = u;
            if (gu >= gw || w == x) {
                v = w, gv = gw, w = u, gw = gu;
            } else if (gu >= gv || v == x || v == w) {
                v = u, gv = gu;
            }
        }
    }
    return x;
}

} // namespace

// ---- The fit -----------------------------------------------------------------------------------------------

GtrFit fit_gtr_gc_counts(const std::vector<PairCounts> &tables, int threads) {
    GtrFit out;
    const int32_t n_tables = (int32_t)tables.size();
    Pairs pairs;
    pairs.threads = threads > 0 ? threads : omp_get_max_threads();
    pairs.cells.resize((size_t)n_tables * N_CELLS);
    pairs.dist.resize(n_tables);
    pairs.ll.resize(n_tables);
    double pi[N_BASES] = {0.0, 0.0, 0.0, 0.0}, pi_sum = 0.0;
    int32_t n = 0;
    for (int32_t t = 0; t < n_tables; ++t) {
        const uint32_t *x = tables[t].n;
        double tot = 0.0, off = 0.0;
        for (int k = 0; k < N_BASES * N_BASES; ++k)
            tot += (double)x[k];
        if (tot < GTR_FIT_MIN_SITES)
            continue;
        double *c = &pairs.cells[(size_t)n * N_CELLS];
        for (int k = 0; k < N_CELLS; ++k) { // fold the table: c(i, j) = x(i, j) + x(j, i); pi gets both ends
            const int i = CELL_I[k], j = CELL_J[k];
            c[k] = (i == j) ? (double)x[i * (N_BASES + 1)] : (double)x[i * 4 + j] + (double)x[j * 4 + i];
            pi[i] += c[k];
            pi[j] += c[k];
            if (i != j)
                off += c[k];
        }
        const double p = off / tot; // the Jukes-Cantor distance starts the pair's Newton search
        pairs.dist[n] = (p < JC_START_P_MAX) ? -0.75 * std::log(1.0 - 4.0 * p / 3.0) : JC_START_FAR;
        if (pairs.dist[n] < JC_START_MIN)
            pairs.dist[n] = JC_START_MIN;
        ++n;
    }
    if (n == 0)
        return out;
    for (int i = 0; i < N_BASES; ++i)
        pi_sum += pi[i];
    for (int i = 0; i < N_BASES; ++i) {
        pi[i] /= pi_sum;
        if (pi[i] < PI_MIN)
            pi[i] = PI_MIN;
    }

    // The start.
    // The likelihood can have a second, lower maximum at "no rate variation", where distorted exchangeabilities stand
    // in for the missing variation.
    // So alpha is first scanned over START_ALPHAS with the exchangeabilities optimised at each value (on at most
    // START_PAIRS tables, warm-started from the previous value); the best value and its neighbours bracket the maximum.
    const int32_t n_start = (n < START_PAIRS) ? n : START_PAIRS;
    constexpr int n_grid = sizeof(START_ALPHAS) / sizeof(START_ALPHAS[0]);
    double theta[N_PARAMS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, scan[N_PARAMS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double f_scan = -std::numeric_limits<double>::infinity(), fg;
    int g_best = 0;
    for (int g = 0; g < n_grid; ++g) {
        scan[LOG_ALPHA] = std::log(START_ALPHAS[g]);
        bfgs(scan, N_FREE_RATES, pi, pairs, n_start, START_MAX_ITERATIONS, &fg, nullptr);
        if (fg > f_scan) {
            f_scan = fg;
            g_best = g;
            std::memcpy(theta, scan, sizeof(theta));
        }
    }

    // The profile: the objective maximised over the exchangeabilities at a given log alpha, on all tables, each search
    // warm-started from the previous one's exchangeabilities and distances.
    // A joint search over all six parameters can stop short in alpha, which pairs determine far less well than the
    // exchangeabilities, so alpha is searched on its own.
    double best_theta[N_PARAMS], f = -std::numeric_limits<double>::infinity();
    bool all_stopped = true;
    auto profile = [&](double log_alpha) {
        theta[LOG_ALPHA] = log_alpha;
        double fp;
        int iters = 0;
        all_stopped = bfgs(theta, N_FREE_RATES, pi, pairs, n, MAX_ITERATIONS, &fp, &iters) && all_stopped;
        out.n_iterations += iters;
        if (fp > f) {
            f = fp;
            std::memcpy(best_theta, theta, sizeof(theta));
        }
        return fp;
    };
    // The bracket a < b < c: the scan's best alpha and its neighbours on the grid (the grid decreases), or the bounds;
    // widened by LOG_ALPHA_WIDEN towards the higher end while the profile rises towards an end that is not a bound.
    // The best point evaluated is kept throughout, so a higher end left behind is not lost.
    double a = (g_best + 1 < n_grid) ? std::log(START_ALPHAS[g_best + 1]) : GTR_FIT_LOG_ALPHA_MIN;
    double b = std::log(START_ALPHAS[g_best]);
    double c = (g_best > 0) ? std::log(START_ALPHAS[g_best - 1]) : GTR_FIT_LOG_ALPHA_MAX;
    double fb = profile(b), fa = profile(a), fc = profile(c);
    bool at_bound = false;
    while (fa > fb || fc > fb) {
        const bool left = fa > fb && a > GTR_FIT_LOG_ALPHA_MIN, right = fc > fb && c < GTR_FIT_LOG_ALPHA_MAX;
        if (left && (!right || fa >= fc)) {
            c = b, fc = fb, b = a, fb = fa;
            a = std::fmax(GTR_FIT_LOG_ALPHA_MIN, b - LOG_ALPHA_WIDEN);
            fa = profile(a);
        } else if (right) {
            a = b, fa = fb, b = c, fb = fc;
            c = std::fmin(GTR_FIT_LOG_ALPHA_MAX, b + LOG_ALPHA_WIDEN);
            fc = profile(c);
        } else { // the maximum is at a bound, already evaluated
            at_bound = true;
            break;
        }
    }
    bool brent_converged = true;
    if (!at_bound)
        brent_max(profile, a, b, fb, c, LOG_ALPHA_TOL, &brent_converged);
    out.converged = brent_converged && all_stopped;
    std::memcpy(theta, best_theta, sizeof(theta));
    unpack(theta, out.rates, &out.alpha);
    std::memcpy(out.base_freqs, pi, sizeof(pi));
    out.log_lik = f + GTR_FIT_RATE_VARIANCE_PENALTY / out.alpha; // f: the objective at theta
    out.n_pairs = n;
    return out;
}

// ---- The pairs -----------------------------------------------------------------------------------------------

std::vector<std::pair<int32_t, int32_t>> draw_fit_pairs(int32_t n_taxa, int32_t n_pairs, uint64_t seed) {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    if (n_taxa < 2 || n_pairs < 1)
        return pairs;
    uint64_t state = seed;
    const int64_t all = (int64_t)n_taxa * (n_taxa - 1) / 2;
    if (all > n_pairs) { // the paper's draw
        pairs.resize(n_pairs);
        for (auto &ab : pairs) {
            const int32_t a = (int32_t)uniform_below(state, (uint64_t)n_taxa);
            int32_t b = (int32_t)uniform_below(state, (uint64_t)(n_taxa - 1));
            if (b >= a)
                ++b;
            ab = {a, b};
        }
        return pairs;
    }
    pairs.reserve(all);
    for (int32_t a = 0; a < n_taxa; ++a)
        for (int32_t b = a + 1; b < n_taxa; ++b)
            pairs.push_back({a, b});
    shuffle(pairs.begin(), pairs.end(), state); // the start scan uses the first tables
    return pairs;
}

GtrFit fit_gtr_gc(const Alignment &aln, int32_t n_pairs, uint64_t seed, int threads) {
    if (n_pairs < 1)
        throw std::invalid_argument("GTR+GC fit: the number of pairs must be >= 1");
    const double t0 = wall_sec();
    const auto pairs = draw_fit_pairs(aln.n_taxa, n_pairs, seed);
    const size_t row_words = (size_t)N_PLANES * aln.n_words;
    std::vector<PairCounts> tables(pairs.size());
    const int max_threads = threads > 0 ? threads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(threads_for(max_threads, (int64_t)pairs.size()))
    for (int64_t m = 0; m < (int64_t)pairs.size(); ++m)
        tables[m] = count_pair(&aln.packed[(size_t)pairs[m].first * row_words],
                               &aln.packed[(size_t)pairs[m].second * row_words], aln.n_words);
    GtrFit fit = fit_gtr_gc_counts(tables, threads);
    fit.time = wall_sec() - t0;
    return fit;
}

} // namespace p2vbme
