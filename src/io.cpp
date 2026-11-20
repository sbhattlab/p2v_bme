/*
 * io.cpp
 *
 * Alignment input by file type.
 */
#include <ctype.h>
#include <string>

#include "fasta.hpp"
#include "io.hpp"
#include "maple.hpp"

namespace p2vbme {

Alignment read_alignment(const std::string &path) {
    std::string ext = path.size() >= 6 ? path.substr(path.size() - 6) : "";
    for (char &c : ext)
        c = (char)tolower((unsigned char)c);
    return ext == ".maple" ? read_maple(path) : read_fasta(path);
}

} // namespace p2vbme
