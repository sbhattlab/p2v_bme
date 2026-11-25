// Unit tests for the .maple reader: the same alignment as FASTA, input errors.
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "fasta.hpp"
#include "io.hpp"
#include "io_test_helpers.hpp"
#include "maple.hpp"

// The same alignment as FASTA and as .maple (reference + differences) packs to the same bitplanes.
TEST(Maple, SameAlignmentAsFasta) {
    const std::string fa = temp_path("m.fa"), mp = temp_path("m.maple");
    write_file(fa, ">s1\nACGTACGTAC\n>s2 descr\nACGTnnnTAC\n>s3\nTCGTAC-tAR\n>s4\nACGTACGTAC\n");
    write_file(mp, ">ref\nACGTAC\nGTAC\n"
                   ">s1\n"
                   ">s2 descr\nn 5 3\n"
                   ">s3\nT 1\n- 7\nr 10\nt 8\n\n"
                   ">s4\n");
    const vecme::Alignment a = vecme::read_fasta(fa), b = vecme::read_alignment(mp);
    EXPECT_EQ(b.n_taxa, 4);
    EXPECT_EQ(b.n_sites, 10);
    EXPECT_EQ(b.names, a.names);
    EXPECT_EQ(b.packed, a.packed);
}

TEST(Maple, RejectsBadInput) {
    const std::string p = temp_path("bad.maple");
    write_file(p, "A 3\n");
    EXPECT_THROW(vecme::read_maple(p), std::runtime_error); // no reference
    write_file(p, ">ref\nACGT\n>s\nA 5\n");
    EXPECT_THROW(vecme::read_maple(p), std::runtime_error); // outside the reference
    write_file(p, ">ref\nACGT\n>s\nn 3 3\n");
    EXPECT_THROW(vecme::read_maple(p), std::runtime_error); // runs past the end
    write_file(p, ">ref\nACGT\n>s\nn 2 9223372036854775807\n");
    EXPECT_THROW(vecme::read_maple(p), std::runtime_error); // a length that would overflow
    write_file(p, ">ref\nACGT\n>s\nA x\n");
    EXPECT_THROW(vecme::read_maple(p), std::runtime_error); // not a position
    write_file(p, ">ref\nACGT\n>s\n>s\n");
    EXPECT_THROW(vecme::read_maple(p), std::runtime_error); // duplicate name
    write_file(p, ">ref\nACGT\n");
    EXPECT_THROW(vecme::read_maple(p), std::runtime_error); // no sequences
}
