/*
 * version.cpp
 *
 * CMakeLists.txt defines VECME_VERSION and VECME_GIT for this file only, so a new revision recompiles only it.
 */
#include "version.hpp"

#ifndef VECME_VERSION
#define VECME_VERSION "unknown"
#endif
#ifndef VECME_GIT
#define VECME_GIT ""
#endif

namespace vecme {

std::string package_version() { return VECME_VERSION; }

std::string package_git_revision() { return VECME_GIT; }

std::string version_string() {
    const std::string git = package_git_revision();
    return package_version() + (git.empty() ? "" : " (" + git + ")");
}

} // namespace vecme
