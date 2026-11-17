/*
 * simulations.cpp
 *
 * GTR(+GC) sequence simulation along a tree. P(t) = exp(Q t) comes from the eigendecomposition of the
 * symmetrised rate matrix Pi^(1/2) Q Pi^(-1/2) (Jacobi rotations); per-site gamma rates are drawn with
 * Marsaglia & Tsang's method.
 */
#include "simulations.hpp"

#include <math.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "random.hpp"

namespace p2vbme {

namespace {

// Uniform in (0, 1].
double uniform_open0(uint64_t &state) { return (double)((splitmix64(state) >> 11) + 1) * 0x1.0p-53; }

double standard_normal(uint64_t &state) { // Box-Muller
    const double u1 = uniform_open0(state), u2 = uniform_open0(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Gamma(shape, 1): Marsaglia & Tsang (2000) for shape >= 1; shape < 1 via Gamma(shape + 1) U^(1 / shape).
double gamma_draw(double shape, uint64_t &state) {
    if (shape < 1.0) {
        const double u = uniform_open0(state);
        return gamma_draw(shape + 1.0, state) * pow(u, 1.0 / shape);
    }
    const double d = shape - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
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
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
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
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
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
    for (int i = 0; i < 4; ++i) {
        if (!(m.freqs[i] > 0.0))
            throw std::invalid_argument("simulation: base frequencies must be positive");
        total += m.freqs[i];
    }
    for (int i = 0; i < 4; ++i)
        e.pi[i] = m.freqs[i] / total;
    for (int k = 0; k < 6; ++k)
        if (!(m.rates[k] > 0.0))
            throw std::invalid_argument("simulation: GTR rates must be positive");
    if (!(m.alpha >= 0.0))
        throw std::invalid_argument("simulation: alpha must be >= 0");

    // exchangeabilities r_ij in the order AC, AG, AT, CG, CT, GT
    double r[4][4] = {};
    const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    for (int k = 0; k < 6; ++k)
        r[pairs[k][0]][pairs[k][1]] = r[pairs[k][1]][pairs[k][0]] = m.rates[k];
    // mean rate mu = sum_i pi_i sum_{j != i} r_ij pi_j; Q = R Pi / mu has one substitution per unit time
    double mu = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j)
                mu += e.pi[i] * r[i][j] * e.pi[j];
    // S = Pi^(1/2) Q Pi^(-1/2): S_ij = r_ij sqrt(pi_i pi_j) / mu (i != j), S_ii = Q_ii
    double S[4][4];
    for (int i = 0; i < 4; ++i) {
        double row = 0.0;
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                continue;
            S[i][j] = r[i][j] * sqrt(e.pi[i] * e.pi[j]) / mu;
            row += r[i][j] * e.pi[j] / mu;
        }
        S[i][i] = -row;
    }
    double W[4][4];
    jacobi4(S, W, e.lambda);
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) {
            e.U[i][k] = W[i][k] / sqrt(e.pi[i]);
            e.Uinv[k][i] = W[i][k] * sqrt(e.pi[i]);
        }
    return e;
}

// Cumulative transition probabilities of row `from` at time t: cum[b] = P(from -> 0..b).
void cumulative_row(const Eigen4 &e, int from, double t, double cum[4]) {
    double ex[4];
    for (int k = 0; k < 4; ++k)
        ex[k] = exp(e.lambda[k] * t);
    double acc = 0.0;
    for (int b = 0; b < 4; ++b) {
        double p = 0.0;
        for (int k = 0; k < 4; ++k)
            p += e.U[from][k] * ex[k] * e.Uinv[k][b];
        acc += p > 0.0 ? p : 0.0;
        cum[b] = acc;
    }
}

int draw_base(const double cum[4], uint64_t &state) {
    const double u = (1.0 - uniform_open0(state)) * cum[3]; // in [0, total)
    for (int b = 0; b < 3; ++b)
        if (u < cum[b])
            return b;
    return 3;
}

} // namespace

void transition_matrix(const SimulationModel &model, double t, double P[4][4]) {
    const Eigen4 e = decompose(model);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double p = 0.0;
            for (int k = 0; k < 4; ++k)
                p += e.U[i][k] * exp(e.lambda[k] * t) * e.Uinv[k][j];
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
    aln.n_words = (n_sites + 63) / 64;
    aln.packed.assign((size_t)N * 5 * aln.n_words, 0);
    for (int32_t i = 0; i < N; ++i)
        aln.names.push_back("t" + std::to_string(i));

    // Sequences of internal nodes live until their two children are drawn (a depth-first stack).
    std::vector<std::vector<uint8_t>> seq(n_nodes);
    seq[root].resize(n_sites);
    double cum_pi[4], acc = 0.0;
    for (int b = 0; b < 4; ++b)
        cum_pi[b] = (acc += e.pi[b]);
    for (int32_t s = 0; s < n_sites; ++s)
        seq[root][s] = (uint8_t)draw_base(cum_pi, st);

    auto draw_child = [&](int32_t parent, int32_t child) {
        std::vector<uint8_t> out(n_sites);
        const double len = tree.branch_lengths[child];
        double cum[4][4];
        if (model.alpha == 0.0)
            for (int a = 0; a < 4; ++a)
                cumulative_row(e, a, len, cum[a]);
        for (int32_t s = 0; s < n_sites; ++s) {
            const int a = seq[parent][s];
            if (model.alpha > 0.0)
                cumulative_row(e, a, rate[s] * len, cum[a]);
            out[s] = (uint8_t)draw_base(cum[a], st);
        }
        if (child < N) {
            uint64_t *row = &aln.packed[(size_t)child * 5 * aln.n_words];
            for (int32_t s = 0; s < n_sites; ++s) {
                const uint64_t bit = 1ULL << (s % 64);
                row[(size_t)out[s] * aln.n_words + s / 64] |= bit;
                row[(size_t)4 * aln.n_words + s / 64] |= bit;
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

double path_length(const Tree &tree, int32_t i, int32_t j) {
    const int32_t n_nodes = (int32_t)tree.c1.size();
    std::vector<int32_t> parent(n_nodes, -1);
    for (int32_t m = (n_nodes + 1) / 2; m < n_nodes; ++m)
        parent[tree.c1[m]] = parent[tree.c2[m]] = m;
    std::vector<double> up(n_nodes, -1.0); // distance from i up to each of its ancestors
    double d = 0.0;
    for (int32_t u = i; u != -1; u = parent[u]) {
        up[u] = d;
        if (parent[u] != -1)
            d += tree.branch_lengths[u];
    }
    d = 0.0;
    for (int32_t u = j; u != -1; u = parent[u]) {
        if (up[u] >= 0.0)
            return d + up[u];
        d += tree.branch_lengths[u];
    }
    throw std::logic_error("path_length: no common ancestor");
}

} // namespace p2vbme
