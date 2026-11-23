/*
 * io.cpp
 *
 * Alignment input by file type.
 */
#include <ctype.h>
#include <string>
#include <string_view>

#include "fasta.hpp"
#include "io.hpp"
#include "maple.hpp"

namespace p2vbme {

Alignment read_alignment(const std::string &path) {
    constexpr std::string_view MAPLE_EXTENSION = ".maple";
    std::string ext = path.size() >= MAPLE_EXTENSION.size() ? path.substr(path.size() - MAPLE_EXTENSION.size()) : "";
    for (char &c : ext)
        c = (char)tolower((unsigned char)c);
    return ext == MAPLE_EXTENSION ? read_maple(path) : read_fasta(path);
}

} // namespace p2vbme
