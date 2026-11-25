# Installs the paper's comparison programs that pixi does not provide, into extern/
# (FastTree, VeryFastTree, FastME and CMAPLE come with the paper environment: pixi install -e paper).
#
#   maple   MAPLE 0.7.6 (the script at commit 3c4c91b) and PyPy 3.11 (v8.0.1), as MAPLE's README recommends:
#             extern/maple/pypy/bin/pypy3 extern/maple/MAPLEv0.7.6.py --input ... --output ...
#   dipper  DIPPER 0.1.5 for NVIDIA GPUs (bioconda's dipper is built without CUDA), built from its source release
#           with the compilers and libraries of the pixi environment dipper (pixi.toml):
#             extern/dipper/bin/dipper
#           Only the CUDA program is built, for the GPU architectures in DIPPER_CUDA_ARCHITECTURES (default native:
#           those of the machine that builds it; e.g. 86 or 90 to build without a GPU): the release hard-codes
#           7.0, 7.5, 8.0 and 8.6 in its CMakeLists.txt, and the script replaces that line.
#           Its version.txt still reads v0.1.4, so the program reports version 0.1.4.
#
# Every download is checked against its SHA-256. An installed program is kept; OVERWRITE=1 installs it again.
#
#   bash scripts/install_baselines.sh          # every program
#   bash scripts/install_baselines.sh maple    # one
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
OVERWRITE=${OVERWRITE:-0}
JOBS=${JOBS:-16} # parallel compile jobs for DIPPER
DIPPER_CUDA_ARCHITECTURES=${DIPPER_CUDA_ARCHITECTURES:-native}
EXTERN=extern

MAPLE_COMMIT=3c4c91b023cabfceb78b70a7967bc23f5cc613de
MAPLE_URL=https://raw.githubusercontent.com/NicolaDM/MAPLE/$MAPLE_COMMIT/MAPLEv0.7.6.py
MAPLE_SHA256=04398a469df032efa7bb265b81d0ab74b386c80b2ea95cff0b3c6d23379a4e62
PYPY_URL=https://downloads.python.org/pypy/pypy3.11-v8.0.1-linux64.tar.gz
PYPY_SHA256=706965bcf372752ba69dafd59d860e51bc2fadbb0f8a47cec3426125d7058ceb # as on pypy.org/checksums.html
DIPPER_URL=https://github.com/TurakhiaLab/DIPPER/archive/refs/tags/v0.1.5.tar.gz
DIPPER_SHA256=7e60648d372677a5107d112f7316808d3a14c56521a7d5748bd873d81e034c62 # as in bioconda's recipe
DIPPER_HARDCODED_ARCHITECTURES="set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)"

# Downloads a file and checks its SHA-256: (url, checksum, path) -> None
fetch() {
  local url=$1 sum=$2 path=$3
  curl -sSfL -o "$path.part" "$url"
  if ! echo "$sum  $path.part" | sha256sum -c --quiet -; then
    rm -f "$path.part"
    echo "checksum mismatch: $url" >&2
    exit 1
  fi
  mv "$path.part" "$path"
}

# Installs MAPLE and PyPy into extern/maple
install_maple() {
  local dir=$EXTERN/maple
  if [ -x "$dir/pypy/bin/pypy3" ] && [ -f "$dir/MAPLEv0.7.6.py" ] && [ "$OVERWRITE" != 1 ]; then
    echo "kept MAPLE ($dir; OVERWRITE=1 installs it again)"
    return
  fi
  rm -rf "$dir"
  mkdir -p "$dir/pypy"
  fetch "$MAPLE_URL" "$MAPLE_SHA256" "$dir/MAPLEv0.7.6.py"
  fetch "$PYPY_URL" "$PYPY_SHA256" "$dir/pypy.tar.gz"
  tar xzf "$dir/pypy.tar.gz" -C "$dir/pypy" --strip-components 1
  rm "$dir/pypy.tar.gz"
  echo "installed MAPLE: $dir/pypy/bin/pypy3 $dir/MAPLEv0.7.6.py ($("$dir/pypy/bin/pypy3" --version 2>&1 | tail -1))"
}

# Builds DIPPER for NVIDIA GPUs into extern/dipper (source in src/, build files in build/)
install_dipper() {
  local dir=$EXTERN/dipper prefix
  if [ -x "$dir/bin/dipper" ] && [ "$OVERWRITE" != 1 ]; then
    echo "kept DIPPER ($dir; OVERWRITE=1 installs it again)"
    return
  fi
  rm -rf "$dir"
  mkdir -p "$dir/src" "$dir/bin"
  fetch "$DIPPER_URL" "$DIPPER_SHA256" "$dir/src.tar.gz"
  tar xzf "$dir/src.tar.gz" -C "$dir/src" --strip-components 1
  rm "$dir/src.tar.gz"
  if [ "$(grep -cF "$DIPPER_HARDCODED_ARCHITECTURES" "$dir/src/CMakeLists.txt")" != 1 ]; then
    echo "DIPPER's CMakeLists.txt does not hard-code its GPU architectures as expected: $DIPPER_HARDCODED_ARCHITECTURES" >&2
    exit 1
  fi
  sed -i "s/$DIPPER_HARDCODED_ARCHITECTURES/set(CMAKE_CUDA_ARCHITECTURES $DIPPER_CUDA_ARCHITECTURES)/" \
    "$dir/src/CMakeLists.txt"
  # the environment's CUDA 12 compiler, not one found elsewhere on the system
  prefix=$(pixi run -e dipper printenv CONDA_PREFIX)
  pixi run -e dipper cmake -S "$dir/src" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON -DUSE_HIP=OFF -DUSE_CPU=OFF -DCMAKE_CUDA_COMPILER="$prefix/bin/nvcc" -DCMAKE_PREFIX_PATH="$prefix"
  pixi run -e dipper cmake --build "$dir/build" --parallel "$JOBS"
  cp "$dir/src/bin/dipper" "$dir/bin/dipper"
  if ldd "$dir/bin/dipper" | grep 'not found'; then
    echo "DIPPER does not find the libraries above" >&2
    exit 1
  fi
  echo "installed DIPPER: $dir/bin/dipper"
}

programs=("$@")
[ ${#programs[@]} -gt 0 ] || programs=(maple dipper)
for program in "${programs[@]}"; do
  case $program in
    maple) install_maple ;;
    dipper) install_dipper ;;
    *)
      echo "unknown program: $program (maple, dipper)" >&2
      exit 1
      ;;
  esac
done
