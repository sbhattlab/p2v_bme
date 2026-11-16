/*
 * io_driver.cpp
 *
 * Driver for tests/python/test_io.py: exercises p2vbme::read_fasta, read_newick and write_newick_unrooted.
 *   io_driver pack  <in.fa> <out.bin>              header int32 [N, L, W] + packed words (bundle layout)
 *   io_driver tree  <in.fa> <in.nwk> <out.nwk>     reads a start tree, writes it back unrooted
 *   io_driver write <in.fa> <tree.bin> <out.nwk>   tree.bin: int32 c1[2N-1], c2[2N-1], float bl[2N-1]
 */
#include <exception>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "io.hpp"

int main(int argc, char **argv) {
    try {
        if (argc == 4 && !strcmp(argv[1], "pack")) {
            const p2vbme::Alignment aln = p2vbme::read_fasta(argv[2]);
            FILE *f = fopen(argv[3], "wb");
            const int32_t hdr[3] = {aln.n_taxa, aln.n_sites, aln.n_words};
            fwrite(hdr, sizeof(int32_t), 3, f);
            fwrite(aln.packed.data(), sizeof(uint64_t), aln.packed.size(), f);
            fclose(f);
            return 0;
        }
        if (argc == 5 && !strcmp(argv[1], "tree")) {
            const p2vbme::Alignment aln = p2vbme::read_fasta(argv[2]);
            const p2vbme::Tree t = p2vbme::read_newick(argv[3], aln.names);
            p2vbme::write_newick_unrooted(argv[4], aln.names, t);
            return 0;
        }
        if (argc == 5 && !strcmp(argv[1], "write")) {
            const p2vbme::Alignment aln = p2vbme::read_fasta(argv[2]);
            const size_t n_nodes = 2 * (size_t)aln.n_taxa - 1;
            p2vbme::Tree t;
            t.c1.resize(n_nodes);
            t.c2.resize(n_nodes);
            std::vector<float> lengths(n_nodes); /* float32 in the file */
            FILE *f = fopen(argv[3], "rb");
            if (!f || fread(t.c1.data(), sizeof(int32_t), n_nodes, f) != n_nodes ||
                fread(t.c2.data(), sizeof(int32_t), n_nodes, f) != n_nodes ||
                fread(lengths.data(), sizeof(float), n_nodes, f) != n_nodes) {
                fprintf(stderr, "io_driver: cannot read %s\n", argv[3]);
                return 1;
            }
            fclose(f);
            t.branch_lengths.assign(lengths.begin(), lengths.end());
            p2vbme::write_newick_unrooted(argv[4], aln.names, t);
            return 0;
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "io_driver: error: %s\n", e.what());
        return 1;
    }
    fprintf(stderr, "usage: io_driver pack|tree|write ...\n");
    return 2;
}
