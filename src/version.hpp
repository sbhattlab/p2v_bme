/*
 * version.hpp
 *
 * The package version, set once in CMakeLists.txt (project()), and the git revision of the build.
 */
#ifndef P2VBME_VERSION_HPP
#define P2VBME_VERSION_HPP

#include <string>

namespace p2vbme {

// The package version, e.g. 0.1.0.
std::string package_version();

// The git revision of a development build ("" if unknown).
std::string package_git_revision();

// The version followed by the git revision in brackets, if known: "0.1.0 (5084b53)".
std::string version_string();

} // namespace p2vbme

#endif // P2VBME_VERSION_HPP
