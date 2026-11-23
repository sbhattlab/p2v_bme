# Phylo2Vec BME

In progress

## Build

With [pixi](https://pixi.sh) and a C++17 compiler, CMake and (optionally) the CUDA toolkit on the system:

```bash
pixi run build   # build/p2v-bme, with the GPU code if CUDA is found
pixi run test
```

To build the reference library (`src/reference`, used by the tests and the Python helpers) for this machine's CPU, turn
on `P2VBME_NATIVE` once; later builds keep it:

```bash
pixi run cmake --preset default -DP2VBME_NATIVE=ON
pixi run build
```
