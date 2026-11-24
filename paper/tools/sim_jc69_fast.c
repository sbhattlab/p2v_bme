/*
 * sim_jc69_fast.c
 *
 * Ultra-fast OpenMP bit-parallel Jukes-Cantor (JC69) sequence simulator and
 * bitplane packer for Phylo2Vec-BME benchmarks (N = 1,000 up to N = 1,000,000+ taxa,
 * L = 1,000 to L = 10,000+ sites).
 *
 * Input binary tree file (<tree_in.bin>):
 *   int32_t N
 *   int32_t true_c1[2*N - 1]
 *   int32_t true_c2[2*N - 1]
 *   float   branch_len[2*N - 1]   (in substitutions per site)
 *
 * Modes:
 *   1. Simulate packed bitplanes + optional FASTA/PHYLIP:
 *      ./sim_jc69_fast --sim <tree_in.bin> <L> <W> <seed> <packed_out.raw> [fasta_out.fa] [phylip_out.phy]
 */

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline double rand_u01(uint64_t *state) { return (double)(splitmix64_next(state) >> 11) * 0x1.0p-53; }

int main(int argc, char **argv) {
    if (argc < 7 || strcmp(argv[1], "--sim") != 0) {
        fprintf(stderr,
                "Usage: %s --sim <tree_in.bin> <L> <W> <seed> <packed_out.raw> [fasta_out.fa] [phylip_out.phy]\n",
                argv[0]);
        return 1;
    }

    const char *tree_bin_path = argv[2];
    int32_t L = atoi(argv[3]);
    int32_t W = atoi(argv[4]);
    uint64_t base_seed = (uint64_t)strtoull(argv[5], NULL, 10);
    const char *packed_out_path = argv[6];
    const char *fasta_out_path = (argc > 7 && strcmp(argv[7], "none") != 0) ? argv[7] : NULL;
    const char *phylip_out_path = (argc > 8 && strcmp(argv[8], "none") != 0) ? argv[8] : NULL;

    FILE *fp = fopen(tree_bin_path, "rb");
    if (!fp) {
        perror("fopen tree_in.bin");
        return 1;
    }
    int32_t N = 0;
    if (fread(&N, sizeof(int32_t), 1, fp) != 1)
        return 1;
    int32_t n_nodes = 2 * N - 1;
    int32_t root_r = n_nodes - 1;

    int32_t *c1 = (int32_t *)malloc((size_t)n_nodes * sizeof(int32_t));
    int32_t *c2 = (int32_t *)malloc((size_t)n_nodes * sizeof(int32_t));
    float *blen = (float *)malloc((size_t)n_nodes * sizeof(float));
    if (fread(c1, sizeof(int32_t), (size_t)n_nodes, fp) != (size_t)n_nodes)
        return 1;
    if (fread(c2, sizeof(int32_t), (size_t)n_nodes, fp) != (size_t)n_nodes)
        return 1;
    if (fread(blen, sizeof(float), (size_t)n_nodes, fp) != (size_t)n_nodes)
        return 1;
    fclose(fp);

    /* Precompute per-node JC69 probability of changing to a DIFFERENT specific base:
     * Under JC69, for edge length t (expected substitutions/site):
     *   P(same) = 1/4 + 3/4 * exp(-4*t/3)
     *   P(any change to one of the 4 uniform bases, which changes base w.p. 3/4) = 1 - exp(-4*t/3).
     * That is: with probability p_resample = 1 - exp(-4*t/3), replace nucleotide with a uniform random base in
     * {0,1,2,3}! Let's verify: P(end up at same base) = (1 - p_resample) + p_resample * (1/4) = 1 - (3/4)*(1 -
     * exp(-4*t/3)) = 1/4 + 3/4 * exp(-4*t/3). P(end up at specific j != i) = p_resample * (1/4) = 1/4 - 1/4 *
     * exp(-4*t/3). Exact! And resampling a uniform base in {0,1,2,3} takes just 2 random bits! */
    float *p_resamp = (float *)malloc((size_t)n_nodes * sizeof(float));
    for (int32_t u = 0; u < n_nodes; ++u) {
        double t = (blen[u] > 0.0f) ? (double)blen[u] : 0.0;
        p_resamp[u] = (float)(1.0 - exp(-4.0 * t / 3.0));
    }

    size_t total_words = (size_t)N * 5 * (size_t)W;
    uint64_t *packed = (uint64_t *)calloc(total_words, sizeof(uint64_t));
    if (!packed) {
        fprintf(stderr, "Failed to allocate %.2f GB for packed sequences\n", (double)(total_words * 8) / 1e9);
        return 1;
    }

    int32_t W_active = (L + 63) / 64;

#pragma omp parallel
    {
        uint64_t *node_b0 = (uint64_t *)malloc((size_t)n_nodes * sizeof(uint64_t));
        uint64_t *node_b1 = (uint64_t *)malloc((size_t)n_nodes * sizeof(uint64_t));

#pragma omp for schedule(static)
        for (int32_t w = 0; w < W_active; ++w) {
            uint64_t rng = base_seed ^ ((uint64_t)(w + 1) * 0x9E3779B97F4A7C15ULL);
            int32_t start_site = w * 64;
            int32_t rem = L - start_site;
            int32_t n_bits = (rem < 64) ? rem : 64;
            uint64_t valid_mask = (n_bits == 64) ? ~0ULL : ((1ULL << n_bits) - 1ULL);

            /* Root state: 64 uniform random nucleotides (2 bits per site: b0, b1) */
            node_b0[root_r] = splitmix64_next(&rng) & valid_mask;
            node_b1[root_r] = splitmix64_next(&rng) & valid_mask;

            /* Top-down traversal: internal nodes are in post-order N .. 2N-2,
             * so iterating u from 2N-2 down to N visits every parent before its children! */
            for (int32_t u = root_r; u >= N; --u) {
                uint64_t pb0 = node_b0[u];
                uint64_t pb1 = node_b1[u];
                int32_t ch[2] = {c1[u], c2[u]};
                for (int k = 0; k < 2; ++k) {
                    int32_t c = ch[k];
                    float pr = p_resamp[c];
                    if (pr <= 0.0f) {
                        node_b0[c] = pb0;
                        node_b1[c] = pb1;
                    } else {
                        uint64_t resamp_mask = 0ULL;
                        /* Fast binomial/Bernoulli mask generation across n_bits sites */
                        for (int b = 0; b < n_bits; ++b) {
                            /* 24-bit uniform float comparison */
                            uint32_t r24 = (uint32_t)(splitmix64_next(&rng) >> 40);
                            if ((float)r24 < pr * 16777216.0f) {
                                resamp_mask |= (1ULL << b);
                            }
                        }
                        if (resamp_mask == 0ULL) {
                            node_b0[c] = pb0;
                            node_b1[c] = pb1;
                        } else {
                            uint64_t nb0 = splitmix64_next(&rng);
                            uint64_t nb1 = splitmix64_next(&rng);
                            node_b0[c] = ((pb0 & ~resamp_mask) | (nb0 & resamp_mask)) & valid_mask;
                            node_b1[c] = ((pb1 & ~resamp_mask) | (nb1 & resamp_mask)) & valid_mask;
                        }
                    }
                }
            }

            /* Pack all N leaves for word w into the 5 bitplanes:
             *   0=A (~b1 & ~b0), 1=C (~b1 & b0), 2=G (b1 & ~b0), 3=T (b1 & b0), 4=Valid */
            for (int32_t x = 0; x < N; ++x) {
                uint64_t b0 = node_b0[x];
                uint64_t b1 = node_b1[x];
                uint64_t is_A = (~b1) & (~b0) & valid_mask;
                uint64_t is_C = (~b1) & (b0)&valid_mask;
                uint64_t is_G = (b1) & (~b0) & valid_mask;
                uint64_t is_T = (b1) & (b0)&valid_mask;
                uint64_t *dst = &packed[(size_t)x * 5 * (size_t)W];
                dst[0 * W + w] = is_A;
                dst[1 * W + w] = is_C;
                dst[2 * W + w] = is_G;
                dst[3 * W + w] = is_T;
                dst[4 * W + w] = valid_mask;
            }
        }

        free(node_b0);
        free(node_b1);
    }

    FILE *out_fp = fopen(packed_out_path, "wb");
    if (!out_fp) {
        perror("fopen packed_out.raw");
        return 1;
    }
    if (fwrite(packed, sizeof(uint64_t), total_words, out_fp) != total_words) {
        fprintf(stderr, "Short write on %s\n", packed_out_path);
        return 1;
    }
    fclose(out_fp);

    if (fasta_out_path || phylip_out_path) {
        FILE *fa_fp = fasta_out_path ? fopen(fasta_out_path, "wb") : NULL;
        FILE *phy_fp = phylip_out_path ? fopen(phylip_out_path, "wb") : NULL;
        if (phy_fp) {
            fprintf(phy_fp, "%d %d\n", N, L);
        }
        char *seq_buf = (char *)malloc((size_t)L + 2);
        seq_buf[L] = '\n';
        seq_buf[L + 1] = '\0';
        for (int32_t x = 0; x < N; ++x) {
            const uint64_t *src = &packed[(size_t)x * 5 * (size_t)W];
            for (int32_t w = 0; w < W_active; ++w) {
                uint64_t mA = src[0 * W + w];
                uint64_t mC = src[1 * W + w];
                uint64_t mG = src[2 * W + w];
                int32_t base = w * 64;
                int32_t lim = (L - base < 64) ? (L - base) : 64;
                for (int32_t b = 0; b < lim; ++b) {
                    uint64_t bit = 1ULL << b;
                    char ch = (mA & bit) ? 'A' : ((mC & bit) ? 'C' : ((mG & bit) ? 'G' : 'T'));
                    seq_buf[base + b] = ch;
                }
            }
            if (fa_fp) {
                fprintf(fa_fp, ">T%d\n", x);
                fwrite(seq_buf, 1, (size_t)L + 1, fa_fp);
            }
            if (phy_fp) {
                fprintf(phy_fp, "T%-9d ", x);
                fwrite(seq_buf, 1, (size_t)L + 1, phy_fp);
            }
        }
        free(seq_buf);
        if (fa_fp)
            fclose(fa_fp);
        if (phy_fp)
            fclose(phy_fp);
    }

    free(packed);
    free(p_resamp);
    free(blen);
    free(c2);
    free(c1);
    return 0;
}
