/*
 * version.cpp
 *
 * CMakeLists.txt defines P2VBME_VERSION and P2VBME_GIT for this file only, so a new revision recompiles only it.
 */
#include "version.hpp"

#ifndef P2VBME_VERSION
#define P2VBME_VERSION "unknown"
#endif
#ifndef P2VBME_GIT
#define P2VBME_GIT ""
#endif

namespace p2vbme {

std::string package_version() { return P2VBME_VERSION; }

std::string package_git_revision() { return P2VBME_GIT; }

std::string version_string() {
    const std::string git = package_git_revision();
    return package_version() + (git.empty() ? "" : " (" + git + ")");
}

} // namespace p2vbme
