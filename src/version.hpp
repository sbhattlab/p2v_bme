/*
 * version.hpp
 *
 * The package version, set once in CMakeLists.txt (project()), and the git revision of the build.
 */
#ifndef VECME_VERSION_HPP
#define VECME_VERSION_HPP

#include <string>

namespace vecme {

// The package version, e.g. 0.1.0.
std::string package_version();

// The git revision of a development build ("" if unknown).
std::string package_git_revision();

// The version followed by the git revision in brackets, if known: "0.1.0 (5084b53)".
std::string version_string();

} // namespace vecme

#endif // VECME_VERSION_HPP
