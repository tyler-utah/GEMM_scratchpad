
/*MACROS START*/
/* the following macros will be defined by the runtime. Do *not* define them in the code
 * START runtime macro definitions
 * THREADS_X_VALUE
 * THREADS_Y_VALUE
 * TUNE_SPLIT_VALUE
 * THREAD_TILE_X_VALUE
 * THREAD_TILE_Y_VALUE
 * WARP_TILE_X_VALUE
 * WARP_TILE_Y_VALUE
 * ASYNC_LOADS_VALUE
 * SWAP_XY_IJ_BLOCK_MAPPING_VALUE
 * SWIZZLEN_VALUE
 * A_OFFSET_VALUE
 * B_OFFSET_VALUE
 * END runtime macro definitions
*/


/* Starting macros for runtime theme */
#define CHECK_AND_SET_SHARED (1)
#define TOTAL_SHARED_SIZE ((0) + (A_SHARED_SIZE * 2) + (B_SHARED_SIZE * 2) + (NUM_BARRIERS * 8))
/* Ending macros for runtime theme */


/* Starting macros for new base ids theme */
#define get_local_id_x() ((get_warp_id_x() * WARP_SHAPE_X_VALUE) + get_lane_id_x())
#define get_local_id_y() ((get_warp_id_y() * WARP_SHAPE_Y_VALUE) + get_lane_id_y())
#define get_global_id_x() (get_block_id_x() * THREADS_X_VALUE + get_local_id_x())
#define get_global_id_y() (get_block_id_y() * THREADS_Y_VALUE + get_local_id_y())
#define get_flattened_dim() (THREADS_X_VALUE * THREADS_Y_VALUE)
#define get_flattened_id() threadIdx.x
/* Ending macros for new base ids theme */


/* Starting macros for specifying the dimensions of how many elements are computed per thread theme */
#define THREADS_X_DIM (THREADS_X_VALUE * THREAD_TILE_X_VALUE * WARP_TILE_X_VALUE)
#define THREADS_Y_DIM (THREADS_Y_VALUE * THREAD_TILE_Y_VALUE * WARP_TILE_Y_VALUE)
/* Ending macros for specifying the dimensions of how many elements are computed per thread theme */


/* Starting macros for indexing macros theme */
#define index2D(i, j, n) ((i) * (n) + (j))
/* Ending macros for indexing macros theme */


/* Starting macros for block tiling theme */
#define get_global_block_tile_y() (get_block_id_y() * THREADS_Y_DIM)
#define get_global_block_tile_x() (get_block_id_x() * THREADS_X_DIM)
/* Ending macros for block tiling theme */


/* Starting macros for shared memory offset theme */
#define A_TYPE_SIZE (2)
#define B_TYPE_SIZE (2)
#define A_OFFSET_SIZE ((16/A_TYPE_SIZE) * A_OFFSET_VALUE)
#define B_OFFSET_SIZE ((16/B_TYPE_SIZE) * B_OFFSET_VALUE)
/* Ending macros for shared memory offset theme */


/* Starting macros for dynamic shared memory theme */
#define A_SHARED_SIZE (THREADS_Y_DIM * (TUNE_SPLIT_VALUE + A_OFFSET_SIZE))
#define B_SHARED_SIZE (TUNE_SPLIT_VALUE * (THREADS_X_DIM + B_OFFSET_SIZE))
#define NUM_BARRIERS (ASYNC_LOADS_VALUE)
/* Ending macros for dynamic shared memory theme */


/* Starting macros for warp reshaping theme */
#define WARP_SHAPE_Y_VALUE (32 / WARP_SHAPE_X_VALUE)
#define get_lane_id() (threadIdx.x % 32)
#define get_lane_id_x() (get_lane_id() % WARP_SHAPE_X_VALUE)
#define get_lane_id_y() (get_lane_id() / WARP_SHAPE_X_VALUE)
#define get_warp_id() (threadIdx.x / 32)
#define get_warp_id_x() (get_warp_id() % (THREADS_X_VALUE / WARP_SHAPE_X_VALUE))
#define get_warp_id_y() (get_warp_id() / (THREADS_X_VALUE / WARP_SHAPE_X_VALUE))
#define get_local_warp_tile_x(warp_tile_x) (get_warp_id_x() * WARP_SHAPE_X_VALUE * THREAD_TILE_X_VALUE * WARP_TILE_X_VALUE + warp_tile_x * THREAD_TILE_X_VALUE * WARP_SHAPE_X_VALUE)
#define get_local_warp_tile_y(warp_tile_y) (get_warp_id_y() * WARP_SHAPE_Y_VALUE * THREAD_TILE_Y_VALUE * WARP_TILE_Y_VALUE + warp_tile_y * THREAD_TILE_Y_VALUE*WARP_SHAPE_Y_VALUE)
#define get_global_warp_tile_x(warp_tile_x) (get_global_block_tile_x() + get_local_warp_tile_x(warp_tile_x))
#define get_global_warp_tile_y(warp_tile_y) (get_global_block_tile_y() + get_local_warp_tile_y(warp_tile_y))
/* Ending macros for warp reshaping theme */


/* Starting macros for thread tiling theme */
#define get_local_id_x_tiled(thread_tile_x) (get_local_id_x() * THREAD_TILE_X_VALUE + thread_tile_x)
#define get_global_id_x_tiled(thread_tile_x) (get_global_block_tile_x() + get_local_id_x_tiled(thread_tile_x))
#define get_local_id_y_tiled(thread_tile_y) (get_local_id_y() * THREAD_TILE_Y_VALUE + thread_tile_y)
#define get_global_id_y_tiled(thread_tile_y) (get_global_block_tile_y() + get_local_id_y_tiled(thread_tile_y))
/* Ending macros for thread tiling theme */


/* START extra macro logic */
#if SWAP_XY_IJ_BLOCK_MAPPING_VALUE == 0
#define get_block_id_x() (blockIdx.x / SWIZZLEN_VALUE)
#define get_block_id_y() (blockIdx.y * SWIZZLEN_VALUE + (blockIdx.x % SWIZZLEN_VALUE))
#define mk_gridDim(variable_name) dim3 variable_name((n / THREADS_X_DIM) * SWIZZLEN_VALUE, (n / THREADS_Y_DIM)/ SWIZZLEN_VALUE)
#else
#define get_block_id_x() (blockIdx.y * SWIZZLEN_VALUE + (blockIdx.x % SWIZZLEN_VALUE))
#define get_block_id_y() (blockIdx.x / SWIZZLEN_VALUE)
#define mk_gridDim(variable_name) dim3 variable_name((n / THREADS_Y_DIM) * SWIZZLEN_VALUE, (n / THREADS_X_DIM)/ SWIZZLEN_VALUE)
#endif

#if WARP_SHAPE_Y_VALUE > THREADS_Y_VALUE
    #error "WARP_RESHAPE_ERROR: WARP_SHAPE_Y_VALUE cannot be greater than THREADS_Y_VALUE"
#endif
#if WARP_SHAPE_X_VALUE > THREADS_X_VALUE
    #error "WARP_RESHAPE_ERROR: WARP_SHAPE_X_VALUE cannot be greater than THREADS_X_VALUE"
#endif

#define TENSOR_CORE_CHECK(M, N, K) (THREADS_X_DIM % N == 0) && (THREADS_Y_DIM % M == 0) && (TUNE_SPLIT_VALUE >= K) && (WARP_SHAPE_X_VALUE * THREAD_TILE_X_VALUE == N) && (WARP_SHAPE_Y_VALUE * THREAD_TILE_Y_VALUE == M)

#if (TENSOR_CORE_CHECK(16, 16, 16))
#define TENSOR_M 16
#define TENSOR_N 16
#define TENSOR_K 16

#elif (TENSOR_CORE_CHECK(32, 8, 16))
#define TENSOR_M 32
#define TENSOR_N 8
#define TENSOR_K 16

#elif (TENSOR_CORE_CHECK(8, 32, 16))
#define TENSOR_M 8
#define TENSOR_N 32
#define TENSOR_K 16
#else
    #error "TENSOR_CORE_SHAPE_ERROR: Tuning parameters could not be shaped into a valid tensor core configuration"
#endif
/* END extra macro logic */

/*MACROS END*/

#include "cache_shared_memory.h"
/*DEVICE START*/

#if !defined(SWIZZLEN_VALUE)
#define SWIZZLEN_VALUE 1
#endif

#include <cuda/barrier>  // Ensure the CUDA barrier header is included if not already

__device__ __forceinline__ void load_shared_memory_tile(
    __half* A_shared,
    __half* B_shared,
    const __half* A,
    const __half* B,
    int k_outer,
    int n,
    cuda::barrier<cuda::thread_scope_block>* block_barrier)
{
    // Load a TILE of A into shared memory
    cache_shared_memory<__half, THREADS_Y_DIM, TUNE_SPLIT_VALUE, get_flattened_dim(), false>(
        A_shared,
        A,
        get_global_block_tile_y(),
        k_outer,
        n,
        block_barrier,
        A_OFFSET_SIZE
    );
    // Load a TILE of B into shared memory
    cache_shared_memory<__half, TUNE_SPLIT_VALUE, THREADS_X_DIM, get_flattened_dim(), false>(
        B_shared,
        B,
        k_outer,
        get_global_block_tile_x(),
        n,
        block_barrier,
        B_OFFSET_SIZE
    );
}

__global__ __launch_bounds__(get_flattened_dim())
void foo(const __half* A, const __half* B, __half* C, int n) {
    // Declare block barrier for async loads
    cuda::barrier<cuda::thread_scope_block>* block_barrier = NULL;

    // Dynamic shared memory declaration
    extern __shared__ __half all_shared_data[];
    // Shared memory buffer for caching tiles of A
    __half* A_shared = all_shared_data;
    // Shared memory buffer for caching tiles of B
    __half* B_shared = all_shared_data + A_SHARED_SIZE;

#if ASYNC_LOADS_VALUE == 1
    // Initialize the block-wide barrier just past B_shared
    block_barrier = (cuda::barrier<cuda::thread_scope_block>*)(B_shared + B_SHARED_SIZE);
    if (threadIdx.x == 0) {
        int total = THREADS_X_VALUE * THREADS_Y_VALUE;
        init(block_barrier, total);
    }
    __syncthreads();
#endif

    const uint i       = get_global_id_y();
    const uint j       = get_global_id_x();
    const uint local_i = get_local_id_y();
    const uint local_j = get_local_id_x();

    // Tensor Core fragment declarations (tiled across warps)
    wmma::fragment<wmma::matrix_a,    TENSOR_M, TENSOR_N, TENSOR_K, half, wmma::row_major>    a_frag[WARP_TILE_Y_VALUE];
    wmma::fragment<wmma::matrix_b,    TENSOR_M, TENSOR_N, TENSOR_K, half, wmma::row_major>    b_frag[WARP_TILE_X_VALUE];
    // Expand accumulator into an array of warp-tile accumulators
    wmma::fragment<wmma::accumulator, TENSOR_M, TENSOR_N, TENSOR_K, half>                   acc_frag[WARP_TILE_Y_VALUE * WARP_TILE_X_VALUE];

    // Initialize all accumulator fragments to zero
    for (int warp_tile_y = 0; warp_tile_y < WARP_TILE_Y_VALUE; warp_tile_y++) {
        for (int warp_tile_x = 0; warp_tile_x < WARP_TILE_X_VALUE; warp_tile_x++) {
            wmma::fill_fragment(
                acc_frag[index2D(warp_tile_y, warp_tile_x, WARP_TILE_X_VALUE)],
                0.0f
            );
        }
    }

    // check if thread is in bounds
    // but keep this check (and comment) because it provides the compiler range information 
    // that it can use to further optimize later code 
    if (i >= n || j >= n) return;

    // Original scalar accumulator (now unused, kept for compatibility)
    __half value = 0;

    for (int k_outer = 0; k_outer < n; k_outer += TUNE_SPLIT_VALUE) {
        // Load shared memory tile phase
        load_shared_memory_tile(
            A_shared,
            B_shared,
            A,
            B,
            k_outer,
            n,
            block_barrier
        );
        // Use the async-aware barrier
        ASYNC_BARRIER();

        // Start of process shared memory tile phase (Tensor Core version)
        for (int k_inner = 0; k_inner < TUNE_SPLIT_VALUE; k_inner += TENSOR_K) {
            // Tile in the Y dimension: load A fragments for each warp_tile_y
            for (int warp_tile_y = 0; warp_tile_y < WARP_TILE_Y_VALUE; warp_tile_y++) {
                __half* A_local_warp_tile = &(
                    A_shared[index2D(
                        get_local_warp_tile_y(warp_tile_y),    // row offset of this warp-tile in A_shared
                        k_inner,                                // column offset in A_shared
                        TUNE_SPLIT_VALUE + A_OFFSET_SIZE        // padded stride of A_shared
                    )]
                );
                wmma::load_matrix_sync(
                    a_frag[warp_tile_y],
                    A_local_warp_tile,
                    TUNE_SPLIT_VALUE + A_OFFSET_SIZE         // add A_OFFSET_SIZE to stride
                );
            }

            // Tile in the X dimension: load B fragments for each warp_tile_x
            for (int warp_tile_x = 0; warp_tile_x < WARP_TILE_X_VALUE; warp_tile_x++) {
                __half* B_local_warp_tile = &(
                    B_shared[index2D(
                        k_inner,                                // row offset in B_shared
                        get_local_warp_tile_x(warp_tile_x),     // column offset of this warp-tile in B_shared
                        THREADS_X_DIM + B_OFFSET_SIZE           // padded stride of B_shared
                    )]
                );
                wmma::load_matrix_sync(
                    b_frag[warp_tile_x],
                    B_local_warp_tile,
                    THREADS_X_DIM + B_OFFSET_SIZE            // add B_OFFSET_SIZE to stride
                );
            }

            // Perform the matrix multiply-accumulate on each warp-tile fragment
            for (int warp_tile_y = 0; warp_tile_y < WARP_TILE_Y_VALUE; warp_tile_y++) {
                for (int warp_tile_x = 0; warp_tile_x < WARP_TILE_X_VALUE; warp_tile_x++) {
                    int idx = index2D(warp_tile_y, warp_tile_x, WARP_TILE_X_VALUE);
                    wmma::mma_sync(
                        acc_frag[idx],
                        a_frag[warp_tile_y],
                        b_frag[warp_tile_x],
                        acc_frag[idx]
                    );
                }
            }
        }
        // End of process shared memory tile phase

        // Replace the sync with the async-aware barrier
        ASYNC_BARRIER();
    }

    // Write the result from the accumulator fragments back to global memory
    for (int warp_tile_y = 0; warp_tile_y < WARP_TILE_Y_VALUE; warp_tile_y++) {
        for (int warp_tile_x = 0; warp_tile_x < WARP_TILE_X_VALUE; warp_tile_x++) {
            int idx = index2D(warp_tile_y, warp_tile_x, WARP_TILE_X_VALUE);
            __half* C_global_warp_tile = &(
                C[index2D(
                    get_global_warp_tile_y(warp_tile_y), // row offset of this warp-tile in C
                    get_global_warp_tile_x(warp_tile_x), // column offset of this warp-tile in C
                    n                                     // global leading dimension of C
                )]
            );
            wmma::store_matrix_sync(
                C_global_warp_tile,
                acc_frag[idx],
                n,
                wmma::mem_row_major
            );
        }
    }
}
/*DEVICE END*/
/*HOST BEGIN*/

#ifdef INCLUDE_HOST

#ifndef SWIZZLEN_VALUE
#define SWIZZLEN_VALUE 1
#endif

// This function will be called by the driver to configure dynamic shared memory
bool check_and_set_shared_memory() {
    int maxConfigurableSharedMem;
    cudaDeviceGetAttribute(&maxConfigurableSharedMem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           0);
    if (maxConfigurableSharedMem < TOTAL_SHARED_SIZE) {
        return false;
    }

    // change kernel name to the name of the kernel
    cudaFuncSetAttribute(foo,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         TOTAL_SHARED_SIZE);
    return true;
}

void foo_host(const __half* A, const __half* B, __half* C, int n) {
    // Change block dimensions to a 1D launch
    dim3 blockDim(THREADS_X_VALUE * THREADS_Y_VALUE);

    /* Please preserve this comment:
     * This macro creates a dim3 object called gridDim initialized to the right sizes.
     * Do not redeclare gridDim in this function, as that will cause a compiler error.
     */
    mk_gridDim(gridDim);

    // Launch with dynamic shared memory size
    foo<<<gridDim, blockDim, TOTAL_SHARED_SIZE>>>(A, B, C, n);
}

#endif
/*HOST END*/