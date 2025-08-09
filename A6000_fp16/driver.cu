#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <sstream>
#include <mma.h>
#include <cuda/barrier>
#include <cmath> // Added for validation utilities
#include <algorithm>
using namespace nvcuda;

#pragma nv_diag_suppress static_var_with_dynamic_init

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#include "kernel.h"

#define CUDA_CALL(call)                                                     \
    {                                                                      \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                  \
            std::cerr << "CUDA error in " << __FILE__ << " at line "        \
                      << __LINE__ << ": " << cudaGetErrorString(err)        \
                      << std::endl;                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                  \
    }

// Host-side random initializers (restored)
static inline void init_random_float16(__half* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
}
static inline void init_random_float32(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Compute one element of C = A x B in row-major order using float accumulation
static inline float ref_matmul_elem(const __half* A, const __half* B, int n, int row, int col) {
    float acc = 0.0f;
    const int aBase = row * n;
    for (int k = 0; k < n; ++k) {
        acc += __half2float(A[aBase + k]) * __half2float(B[k * n + col]);
    }
    return acc;
}

// Validate device result C_host against a CPU reference computed on a sample of indices
static void validate_sampled(const __half* A_host, const __half* B_host, const __half* C_host,
                             int n, int samples = 16, float tol = 1e-1f) {
    if (samples > n) samples = n;
    int mismatches = 0;
    double maxAbsErr = 0.0;
    double meanAbsErr = 0.0;
    int checked = 0;

    // deterministic sampling of (row, col) pairs
    // cover corners and a grid across the matrix
    auto idx = [&](int t) { return (long long)t * (n - 1) / (samples - 1); };
    for (int si = 0; si < samples; ++si) {
        int r = (samples > 1) ? (int)idx(si) : 0;
        for (int sj = 0; sj < samples; ++sj) {
            int c = (samples > 1) ? (int)idx(sj) : 0;

            float ref = ref_matmul_elem(A_host, B_host, n, r, c);
            float got = __half2float(C_host[r * n + c]);
            double err = std::fabs((double)got - (double)ref);
            maxAbsErr = std::max(maxAbsErr, err);
            meanAbsErr += err;
            ++checked;
            if (err > tol) {
                if (mismatches < 10) {
                    std::cerr << "Mismatch at (" << r << ", " << c << ")"
                              << ": got=" << got << ", ref=" << ref
                              << ", absErr=" << err << std::endl;
                }
                ++mismatches;
            }
        }
    }

    meanAbsErr /= std::max(1, checked);
    if (mismatches == 0) {
        std::cout << "Validation PASSED on " << checked << " sampled elements."
                  << " maxAbsErr=" << maxAbsErr << " meanAbsErr=" << meanAbsErr
                  << " tol=" << tol << std::endl;
    } else {
        std::cout << "Validation FAILED: " << mismatches << "/" << checked
                  << " sampled elements exceed tol=" << tol
                  << ", maxAbsErr=" << maxAbsErr
                  << ", meanAbsErr=" << meanAbsErr << std::endl;
    }
}

int main(int argc, char *argv[]) {

  bool write_output = false;
  char *fileName;
  if (argc == 2) {
     write_output = true;
     fileName = argv[1];
     // avoid unused warning if not writing to file
     (void)fileName;
   }
  // We no longer write outputs to a file; silence unused flag
  (void)write_output;

  // Declare variables
    __half* A_host;
  __half* A_device;
  __half* B_host;
  __half* B_device;
  __half* C_host;
  __half* C_device;
  const int n = n_VALUE;

  // Initialize arrays and copy input arrays
    A_host = new __half[A_LEN];
  CUDA_CALL(cudaMalloc((void**)&A_device, A_LEN * sizeof(__half)));
  init_random_float16(A_host, A_LEN);
  CUDA_CALL(cudaMemcpy(A_device, A_host, A_LEN * sizeof(__half), cudaMemcpyHostToDevice));
  B_host = new __half[B_LEN];
  CUDA_CALL(cudaMalloc((void**)&B_device, B_LEN * sizeof(__half)));
  init_random_float16(B_host, B_LEN);
  CUDA_CALL(cudaMemcpy(B_device, B_host, B_LEN * sizeof(__half), cudaMemcpyHostToDevice));
  C_host = new __half[C_LEN];
  CUDA_CALL(cudaMalloc((void**)&C_device, C_LEN * sizeof(__half)));

  #if CHECK_AND_SET_SHARED == 1
  if (!check_and_set_shared_memory()) {
    std::cerr << "Error: too many resources requested for launch. (dynamic shared memory)" << std::endl;
    return 1;
  }
  #endif 

  // Warm up kernel run
  if (REPEAT > 1) {
    for (int i = 0; i < REPEAT/10; ++i) {
    foo_host(A_device, B_device, C_device, n);
    }
  }

  // Create events
    CUDA_CALL(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start, 0));

  // Call kernel for real
  for (int i = 0; i < REPEAT; ++i) {
    foo_host(A_device, B_device, C_device, n);
 }
    
  // Stop the events and report the time
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));

  CUDA_CALL(cudaGetLastError());

    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds/REPEAT << " ms" << std::endl;

    // Copy values back to the host
      CUDA_CALL(cudaMemcpy(C_host, C_device, C_LEN * sizeof(__half), cudaMemcpyDeviceToHost));

    // Validate against a CPU reference GEMM (sampled to keep runtime reasonable for large n)
    // Adjust samples/tolerance via macros if desired
#if VALIDATE == 1
    {
        int samples = n;      // number of row/col samples per dimension
        if (samples > n) samples = n;
        float tol = 10;     // tolerance appropriate for FP16 accumulation to float
        validate_sampled(A_host, B_host, C_host, n, samples, tol);
    }
#endif

}
