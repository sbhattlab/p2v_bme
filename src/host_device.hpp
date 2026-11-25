/*
 * host_device.hpp
 *
 * P2V_HD marks a function for both the CPU and the GPU under nvcc (__host__ __device__), and is empty elsewhere, so the
 * per-pair distance formulas compile for the GPU kernels and for the CPU-only build alike.
 */
#ifndef VECME_HOST_DEVICE_HPP
#define VECME_HOST_DEVICE_HPP

#ifdef __CUDACC__
#define P2V_HD __host__ __device__
#else
#define P2V_HD
#endif

#endif // VECME_HOST_DEVICE_HPP
