// Link-only fault injection. Production sources are not modified.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
extern "C" cudaError_t __real_cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
static int injected = 0;
extern "C" int audit_fault_count() { return injected; }
extern "C" cudaError_t __wrap_cudaMemcpy(void *dst, const void *src, size_t n, cudaMemcpyKind kind) {
    const char *mode = std::getenv("AUDIT_COPY_FAULT");
    bool fail = mode && kind == cudaMemcpyDeviceToHost && ((mode[0] == 'g' && n == 112) || (mode[0] == 'o' && n == 8));
    if (fail && injected == 0) {
        ++injected;
        std::fprintf(stderr, "audit injected DtoH failure: bytes=%zu\n", n);
        return cudaErrorUnknown;
    }
    return __real_cudaMemcpy(dst, src, n, kind);
}
