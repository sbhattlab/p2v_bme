// Writes a simulated alignment for the command-line tests: make_test_fasta OUT N SITES SEED writes OUT.fa (JC69
// along a random tree; sequence names t0..t{N-1}; a few missing sites), OUT_true.nwk (the tree, unrooted, with
// branch lengths), OUT.maple (the same alignment as differences from t0) and OUT_dups.fa (OUT.fa plus three copies
// of t1 and two of t2, named t1_copy1 ...).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "newick.hpp"
#include "simulations.hpp"

static bool write_fasta(const std::string &path, const std::vector<std::string> &names,
                        const std::vector<std::string> &seqs) {
    FILE *f = fopen(path.c_str(), "w");
    if (!f)
        return false;
    for (size_t i = 0; i < seqs.size(); ++i)
        fprintf(f, ">%s\n%s\n", names[i].c_str(), seqs[i].c_str());
    return fclose(f) == 0;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: make_test_fasta OUT N SITES SEED\n");
        return 2;
    }
    const std::string out = argv[1];
    const int32_t n = atoi(argv[2]), sites = atoi(argv[3]);
    const uint64_t seed = strtoull(argv[4], NULL, 10);
    const p2vbme::Tree tree = p2vbme::random_tree(n, 0.01, 0.1, seed);
    const p2vbme::Alignment aln = p2vbme::simulate_alignment(tree, sites, p2vbme::SimulationModel(), seed + 1);
    const char bases[4] = {'A', 'C', 'G', 'T'};
    std::vector<std::string> seqs(n, std::string(sites, '-'));
    for (int32_t i = 0; i < n; ++i) {
        const uint64_t *row = &aln.packed[(size_t)i * 5 * aln.n_words];
        for (int32_t s = 0; s < sites; ++s)
            for (int b = 0; b < 4; ++b)
                if ((row[(size_t)b * aln.n_words + s / 64] >> (s % 64)) & 1ULL)
                    seqs[i][s] = bases[b];
    }
    seqs[3][5] = 'N'; /* a few missing sites, so the .maple file has runs of n and - */
    for (int32_t s = 10; s < 14; ++s)
        seqs[4][s] = '-';
    if (!write_fasta(out + ".fa", aln.names, seqs))
        return 1;

    std::vector<std::string> names_dups = aln.names, seqs_dups = seqs;
    for (int c = 1; c <= 3; ++c) {
        names_dups.push_back("t1_copy" + std::to_string(c));
        seqs_dups.push_back(seqs[1]);
    }
    for (int c = 1; c <= 2; ++c) {
        names_dups.push_back("t2_copy" + std::to_string(c));
        seqs_dups.push_back(seqs[2]);
    }
    if (!write_fasta(out + "_dups.fa", names_dups, seqs_dups))
        return 1;

    FILE *f = fopen((out + ".maple").c_str(), "w"); /* reference = t0 */
    if (!f)
        return 1;
    fprintf(f, ">reference\n%s\n", seqs[0].c_str());
    for (int32_t i = 0; i < n; ++i) {
        fprintf(f, ">%s\n", aln.names[i].c_str());
        for (int32_t s = 0; s < sites;) {
            if (seqs[i][s] == seqs[0][s]) {
                ++s;
                continue;
            }
            int32_t e = s + 1;
            if (seqs[i][s] == '-' || seqs[i][s] == 'N')
                while (e < sites && seqs[i][e] == seqs[i][s] && seqs[0][e] != seqs[i][s])
                    ++e;
            fprintf(f, "%c %d %d\n", seqs[i][s] == 'N' ? 'n' : seqs[i][s], s + 1, e - s);
            s = e;
        }
    }
    if (fclose(f) != 0)
        return 1;

    p2vbme::write_newick_unrooted(out + "_true.nwk", aln.names, tree);
    return 0;
}
