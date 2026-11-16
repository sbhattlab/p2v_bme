// Unit tests for the FASTA reader: packing into bitplanes, input errors.
#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include "fasta.hpp"
#include "io_test_helpers.hpp"

namespace {

bool bit(const p2vbme::Alignment &a, int taxon, int plane, int site) {
    return (a.packed[((size_t)taxon * 5 + plane) * a.n_words + site / 64] >> (site % 64)) & 1ULL;
}

} // namespace

TEST(Fasta, PacksPlanesAsTheBundles) {
    const std::string p = temp_path("pack.fa");
    write_file(p, ">a desc\nACGTU-\nNnRY\n>b\nacgtac\ngtac\n");
    const auto aln = p2vbme::read_fasta(p);
    ASSERT_EQ(aln.n_taxa, 2);
    ASSERT_EQ(aln.n_sites, 10);
    ASSERT_EQ(aln.n_words, 1);
    EXPECT_EQ(aln.names, (std::vector<std::string>{"a", "b"}));
    // taxon a: A C G T U - N n R Y
    EXPECT_TRUE(bit(aln, 0, 0, 0));
    EXPECT_TRUE(bit(aln, 0, 1, 1));
    EXPECT_TRUE(bit(aln, 0, 2, 2));
    EXPECT_TRUE(bit(aln, 0, 3, 3));
    EXPECT_TRUE(bit(aln, 0, 3, 4)); // U counts as T
    for (int site = 5; site < 10; ++site)
        for (int plane = 0; plane < 5; ++plane)
            EXPECT_FALSE(bit(aln, 0, plane, site)) << site << " " << plane;
    for (int site = 0; site < 5; ++site)
        EXPECT_TRUE(bit(aln, 0, 4, site));
    // taxon b: lowercase is read as uppercase
    EXPECT_TRUE(bit(aln, 1, 0, 0));
    EXPECT_TRUE(bit(aln, 1, 1, 9));
}

TEST(Fasta, RejectsBadInput) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {">a\nACGT\n>b\nACG\n", "must be aligned"},
        {">a\nACGT\n>a\nACGT\n", "duplicate name"},
        {"ACGT\n>a\nACGT\n", "before the first"},
        {"", "no sequences"},
    };
    for (const auto &[text, message] : cases) {
        const std::string p = temp_path("bad.fa");
        write_file(p, text);
        try {
            p2vbme::read_fasta(p);
            ADD_FAILURE() << "no error for: " << text;
        } catch (const std::runtime_error &e) {
            EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
        }
    }
}

// Random sequences over a mixed alphabet, wrapped at 60 columns, with a description after each name: the bitplanes
// equal those of a plain letter-by-letter loop, including the zero padding of the last, partial word.
TEST(Fasta, RandomSequencesPackAsALetterByLetterLoop) {
    const std::string alphabet = "ACGTacgtUu-NnRY?";
    const int n = 40, length = 150, n_words = 3; // the last word partial
    std::mt19937 rng(0);
    std::vector<std::string> seqs(n);
    std::string text;
    for (int x = 0; x < n; ++x) {
        for (int i = 0; i < length; ++i)
            seqs[x].push_back(alphabet[rng() % alphabet.size()]);
        text += ">seq" + std::to_string(x) + " some description\n";
        for (int k = 0; k < length; k += 60)
            text += seqs[x].substr(k, 60) + "\n";
    }
    const std::string p = temp_path("random.fa");
    write_file(p, text);
    const auto aln = p2vbme::read_fasta(p);
    ASSERT_EQ(aln.n_taxa, n);
    ASSERT_EQ(aln.n_sites, length);
    ASSERT_EQ(aln.n_words, n_words);
    std::vector<uint64_t> expected((size_t)n * 5 * n_words, 0);
    for (int x = 0; x < n; ++x) {
        EXPECT_EQ(aln.names[x], "seq" + std::to_string(x));
        for (int i = 0; i < length; ++i) {
            const char c = seqs[x][i];
            int plane = -1;
            if (c == 'A' || c == 'a')
                plane = 0;
            else if (c == 'C' || c == 'c')
                plane = 1;
            else if (c == 'G' || c == 'g')
                plane = 2;
            else if (c == 'T' || c == 't' || c == 'U' || c == 'u')
                plane = 3;
            if (plane < 0)
                continue;
            uint64_t *row = &expected[(size_t)x * 5 * n_words];
            row[(size_t)plane * n_words + i / 64] |= 1ULL << (i % 64);
            row[(size_t)4 * n_words + i / 64] |= 1ULL << (i % 64);
        }
    }
    EXPECT_EQ(aln.packed, expected);
}
