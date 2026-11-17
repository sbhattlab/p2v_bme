/*
 * fasta.cpp
 *
 * Aligned DNA FASTA input, packed as in the .bin bundles.
 */
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "fasta.hpp"

namespace p2vbme {

Alignment read_fasta(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    // One pass over the file: a '>' line starts a record, and the lines after it are its sequence.
    // A record is packed when the next one starts, or at the end of the file.
    Alignment aln;
    std::unordered_map<std::string, int32_t> seen;
    std::string name, seq;
    bool have_record = false;
    // Packs the record read so far (name, seq) into the alignment.
    // The first record sets the number of sites; every other one must have the same.
    auto finish_record = [&]() {
        if (!have_record)
            return;
        if (aln.n_taxa == 0) {
            if (seq.empty())
                throw std::runtime_error(path + ": first sequence '" + name + "' is empty");
            aln.n_sites = (int32_t)seq.size();
            aln.n_words = (aln.n_sites + 63) / 64;
        } else if ((int32_t)seq.size() != aln.n_sites) {
            throw std::runtime_error(path + ": sequence '" + name + "' has " + std::to_string(seq.size()) +
                                     " sites, the first has " + std::to_string(aln.n_sites) +
                                     " (the alignment must be aligned)");
        }
        if (!seen.emplace(name, aln.n_taxa).second)
            throw std::runtime_error(path + ": duplicate name '" + name + "'");
        // Appends the sequence's 5 bitplanes (A, C, G, T, observed), n_words 64-bit words each, zeroed.
        // pack_sequence then sets one bit per site.
        const size_t row = aln.packed.size();
        aln.packed.resize(row + (size_t)5 * aln.n_words, 0);
        pack_sequence(seq, aln.n_words, aln.packed.data() + row);
        aln.names.push_back(name);
        ++aln.n_taxa;
        seq.clear();
    };

    std::string line;
    long line_no = 0;
    // std::getline drops the '\n'; the file closes itself, also when an error is thrown.
    while (std::getline(in, line)) {
        ++line_no;
        // Strips the '\r' of Windows line endings (\r\n).
        while (!line.empty() && line.back() == '\r')
            line.pop_back();
        // A header: the name is the first word after '>'; the rest of the line (a description) is ignored.
        if (!line.empty() && line[0] == '>') {
            finish_record();
            size_t p = 1;
            while (p < line.size() && (line[p] == ' ' || line[p] == '\t'))
                ++p;
            size_t q = p;
            while (q < line.size() && line[q] != ' ' && line[q] != '\t')
                ++q;
            name.assign(line, p, q - p);
            if (name.empty())
                throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty sequence name");
            have_record = true;
            continue;
        }
        // A sequence line: every character but spaces and tabs is a site.
        // Gaps and ambiguity codes count as sites too; pack_sequence sets no bit for them.
        for (const char ch : line) {
            if (ch == ' ' || ch == '\t')
                continue;
            if (!have_record)
                throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                         ": sequence data before the first '>' header");
            seq.push_back(ch);
        }
    }
    if (in.bad())
        throw std::runtime_error("cannot read " + path);
    finish_record(); // the last record has no '>' after it
    if (aln.n_taxa == 0)
        throw std::runtime_error(path + ": no sequences");
    return aln;
}

} // namespace p2vbme
