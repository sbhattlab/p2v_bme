/*
 * simulations.cpp
 *
 * GTR(+GC) sequence simulation along a tree. P(t) = exp(Q t) comes from the eigendecomposition of the
 * symmetrised rate matrix Pi^(1/2) Q Pi^(-1/2) (Jacobi rotations); per-site gamma rates are drawn with
 * Marsaglia & Tsang's method.
 *
 * Reference
 *   Marsaglia, G. & Tsang, W. W. (2000). A simple method for generating gamma variables. ACM Transactions on
 *     Mathematical Software 26, 363-372. doi:10.1145/358407.358414
 */
#include "simulations.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eigen_dist.hpp"
#include "random.hpp"

namespace p2vbme {

namespace {

// Uniform in (0, 1].
double uniform_open0(uint64_t &state) { return (double)((splitmix64(state) >> 11) + 1) * 0x1.0p-53; }

double standard_normal(uint64_t &state) { // Box-Muller
    const double u1 = uniform_open0(state), u2 = uniform_open0(state);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// Gamma(shape, 1): Marsaglia & Tsang (2000) for shape >= 1; shape < 1 via Gamma(shape + 1) U^(1 / shape).
double gamma_draw(double shape, uint64_t &state) {
    if (shape < 1.0) {
        const double u = uniform_open0(state);
        return gamma_draw(shape + 1.0, state) * std::pow(u, 1.0 / shape);
    }
    const double d = shape - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = standard_normal(state);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = uniform_open0(state);
        if (u < 1.0 - 0.0331 * x * x * x * x)
            return d * v;
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v)))
            return d * v;
    }
}

// Q = U diag(lambda) Uinv for the normalised GTR rate matrix.
struct Eigen4 {
    double U[4][4], Uinv[4][4], lambda[4];
    double pi[4];
};

// Cyclic Jacobi eigenvalue algorithm for a symmetric 4 x 4 matrix: A = V diag(d) V^T (A is overwritten).
void jacobi4(double A[4][4], double V[4][4], double d[4]) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            V[i][j] = i == j ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q)
                off += A[p][q] * A[p][q];
        if (off < 1e-32)
            break;
        for (int p = 0; p < 4; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                if (A[p][q] == 0.0)
                    continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 4; ++k) { // A <- A J
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 4; ++k) { // A <- J^T A
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 4; ++k) { // V <- V J
                    const double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < 4; ++i)
        d[i] = A[i][i];
}

Eigen4 decompose(const SimulationModel &m) {
    Eigen4 e;
    double total = 0.0;
    for (int i = 0; i < N_BASES; ++i) {
        if (!(m.freqs[i] > 0.0))
            throw std::invalid_argument("simulation: base frequencies must be positive");
        total += m.freqs[i];
    }
    for (int i = 0; i < N_BASES; ++i)
        e.pi[i] = m.freqs[i] / total;
    for (int k = 0; k < N_GTR_RATES; ++k)
        if (!(m.rates[k] > 0.0))
            throw std::invalid_argument("simulation: GTR rates must be positive");
    if (!(m.alpha >= 0.0))
        throw std::invalid_argument("simulation: alpha must be >= 0");

    // S = Pi^(1/2) Q Pi^(-1/2), Q normalised to one substitution per unit time (eigen_dist.hpp)
    double R[N_BASES * N_BASES], S[N_BASES][N_BASES];
    gtr_symmetric_rate_matrix(m.rates, e.pi, R);
    for (int i = 0; i < N_BASES; ++i)
        for (int j = 0; j < N_BASES; ++j)
            S[i][j] = R[N_BASES * i + j];
    double W[N_BASES][N_BASES];
    jacobi4(S, W, e.lambda);
    for (int i = 0; i < N_BASES; ++i)
        for (int k = 0; k < N_BASES; ++k) {
            e.U[i][k] = W[i][k] / std::sqrt(e.pi[i]);
            e.Uinv[k][i] = W[i][k] * std::sqrt(e.pi[i]);
        }
    return e;
}

// Cumulative transition probabilities of row `from` at time t: cum[b] = P(from -> 0..b).
void cumulative_row(const Eigen4 &e, int from, double t, double cum[N_BASES]) {
    double ex[N_BASES];
    for (int k = 0; k < N_BASES; ++k)
        ex[k] = std::exp(e.lambda[k] * t);
    double acc = 0.0;
    for (int b = 0; b < N_BASES; ++b) {
        double p = 0.0;
        for (int k = 0; k < N_BASES; ++k)
            p += e.U[from][k] * ex[k] * e.Uinv[k][b];
        acc += p > 0.0 ? p : 0.0;
        cum[b] = acc;
    }
}

int draw_base(const double cum[N_BASES], uint64_t &state) {
    const double u = (1.0 - uniform_open0(state)) * cum[3]; // in [0, total)
    for (int b = 0; b < 3; ++b)
        if (u < cum[b])
            return b;
    return 3;
}

} // namespace

void transition_matrix(const SimulationModel &model, double t, double P[N_BASES][N_BASES]) {
    const Eigen4 e = decompose(model);
    for (int i = 0; i < N_BASES; ++i)
        for (int j = 0; j < N_BASES; ++j) {
            double p = 0.0;
            for (int k = 0; k < N_BASES; ++k)
                p += e.U[i][k] * std::exp(e.lambda[k] * t) * e.Uinv[k][j];
            P[i][j] = p;
        }
}

Tree random_tree(int32_t n, double min_length, double max_length, uint64_t seed) {
    if (n < 2)
        throw std::invalid_argument("random_tree: need at least 2 leaves");
    if (!(min_length >= 0.0 && max_length >= min_length))
        throw std::invalid_argument("random_tree: need 0 <= min_length <= max_length");
    uint64_t st = seed;
    Tree t;
    t.c1.assign(2 * n - 1, -1);
    t.c2.assign(2 * n - 1, -1);
    t.branch_lengths.assign(2 * n - 1, 0.0f);
    std::vector<int32_t> pool(n);
    for (int32_t i = 0; i < n; ++i)
        pool[i] = i;
    for (int32_t m = n; m < 2 * n - 1; ++m) {
        int32_t picked[2];
        for (int k = 0; k < 2; ++k) {
            const size_t a = (size_t)uniform_below(st, (uint64_t)pool.size());
            std::swap(pool[a], pool.back());
            picked[k] = pool.back();
            pool.pop_back();
        }
        t.c1[m] = picked[0];
        t.c2[m] = picked[1];
        pool.push_back(m);
    }
    for (int32_t u = 0; u < 2 * n - 2; ++u)
        t.branch_lengths[u] = (float)(min_length + (max_length - min_length) * (1.0 - uniform_open0(st)));
    return t;
}

Alignment simulate_alignment(const Tree &tree, int32_t n_sites, const SimulationModel &model, uint64_t seed) {
    const int32_t n_nodes = (int32_t)tree.c1.size();
    if (n_nodes < 3 || n_nodes % 2 == 0 || tree.c2.size() != tree.c1.size() ||
        tree.branch_lengths.size() != tree.c1.size())
        throw std::invalid_argument("simulate_alignment: the tree needs c1, c2 and branch_lengths of size 2N - 1");
    if (n_sites < 1)
        throw std::invalid_argument("simulate_alignment: need at least one site");
    const int32_t N = (n_nodes + 1) / 2, root = n_nodes - 1;
    const Eigen4 e = decompose(model);
    uint64_t st = seed;

    std::vector<double> rate(n_sites, 1.0);
    if (model.alpha > 0.0)
        for (int32_t s = 0; s < n_sites; ++s)
            rate[s] = gamma_draw(model.alpha, st) / model.alpha;

    Alignment aln;
    aln.n_taxa = N;
    aln.n_sites = n_sites;
    aln.n_words = (n_sites + WORD_BITS - 1) / WORD_BITS;
    aln.packed.assign((size_t)N * N_PLANES * aln.n_words, 0);
    for (int32_t i = 0; i < N; ++i)
        aln.names.push_back("t" + std::to_string(i));

    // Sequences of internal nodes live until their two children are drawn (a depth-first stack).
    std::vector<std::vector<uint8_t>> seq(n_nodes);
    seq[root].resize(n_sites);
    double cum_pi[N_BASES], acc = 0.0;
    for (int b = 0; b < N_BASES; ++b)
        cum_pi[b] = (acc += e.pi[b]);
    for (int32_t s = 0; s < n_sites; ++s)
        seq[root][s] = (uint8_t)draw_base(cum_pi, st);

    auto draw_child = [&](int32_t parent, int32_t child) {
        std::vector<uint8_t> out(n_sites);
        const double len = tree.branch_lengths[child];
        double cum[N_BASES][N_BASES];
        if (model.alpha == 0.0)
            for (int a = 0; a < N_BASES; ++a)
                cumulative_row(e, a, len, cum[a]);
        for (int32_t s = 0; s < n_sites; ++s) {
            const int a = seq[parent][s];
            if (model.alpha > 0.0)
                cumulative_row(e, a, rate[s] * len, cum[a]);
            out[s] = (uint8_t)draw_base(cum[a], st);
        }
        if (child < N) {
            uint64_t *row = &aln.packed[(size_t)child * N_PLANES * aln.n_words];
            for (int32_t s = 0; s < n_sites; ++s) {
                const uint64_t bit = 1ULL << (s % WORD_BITS);
                row[(size_t)out[s] * aln.n_words + s / WORD_BITS] |= bit;
                row[(size_t)VALID_PLANE * aln.n_words + s / WORD_BITS] |= bit;
            }
        } else {
            seq[child] = std::move(out);
        }
    };
    std::vector<int32_t> stack = {root};
    while (!stack.empty()) {
        const int32_t u = stack.back();
        stack.pop_back();
        draw_child(u, tree.c1[u]);
        draw_child(u, tree.c2[u]);
        std::vector<uint8_t>().swap(seq[u]);
        if (tree.c2[u] >= N)
            stack.push_back(tree.c2[u]);
        if (tree.c1[u] >= N)
            stack.push_back(tree.c1[u]);
    }
    return aln;
}

} // namespace p2vbme
