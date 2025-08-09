// For GEMMs that are sufficiently optimized, these should be provided in the main kernel file.
// However, to make sure this header can be used earlier than warp tiling, we provide trivial definitions here

#if !defined THREAD_TILE_X_VALUE
#define THREAD_TILE_X_VALUE 1
#endif

#if !defined THREAD_TILE_Y_VALUE
#define THREAD_TILE_Y_VALUE 1
#endif

#if !defined THREAD_TILE_FLATTENED_DIM
#define THREAD_TILE_FLATTENED_DIM (THREAD_TILE_X_VALUE * THREAD_TILE_Y_VALUE)
#endif

#if !defined X_LOCAL_CHUNK_VALUE
#define X_LOCAL_CHUNK_VALUE (256)
#endif

template <typename T, int local_x_chunk = X_LOCAL_CHUNK_VALUE> 
__device__ __forceinline__ void write_output(T *target, const T* registers, int n) {
  
  constexpr uint f4_multiple  = std::is_same_v<T, __half> ? 8 : 4;
  constexpr uint f2_multiple = std::is_same_v<T, __half> ? 4 : 2;
  constexpr uint f1_multiple  = std::is_same_v<T, __half> ? 2 : 1;

  if constexpr (THREAD_TILE_X_VALUE >= f4_multiple && local_x_chunk >= f4_multiple) { 
    for (int thread_tile_y = 0; thread_tile_y < THREAD_TILE_Y_VALUE; thread_tile_y++) {
      const uint tiled_i = get_global_id_y_tiled(thread_tile_y);
      for (int thread_tile_x = 0; thread_tile_x < THREAD_TILE_X_VALUE; thread_tile_x+=f4_multiple) {
        const uint tiled_j = get_global_id_x_tiled(thread_tile_x);
        if constexpr (std::is_same_v<T, float>) {
          float4 tmp;
          
          tmp.x = registers[index2D(thread_tile_y, thread_tile_x + 0, THREAD_TILE_X_VALUE)];
          tmp.y = registers[index2D(thread_tile_y, thread_tile_x + 1, THREAD_TILE_X_VALUE)];
          tmp.z = registers[index2D(thread_tile_y, thread_tile_x + 2, THREAD_TILE_X_VALUE)];
          tmp.w = registers[index2D(thread_tile_y, thread_tile_x + 3, THREAD_TILE_X_VALUE)];
          reinterpret_cast<float4 *>(
            &target[index2D(tiled_i, tiled_j, n)])[0] = tmp;
          /*if (threadIdx.x == 464 && blockIdx.x == 59 && blockIdx.y == 0) {
              printf("target in writeouput: %d!\n", index2D(tiled_i, tiled_j, n));
          }*/
          //if (target[index2D(tiled_i, tiled_j, n)] != registers[index2D(thread_tile_y, thread_tile_x + 0, THREAD_TILE_X_VALUE)]) {
          //  printf("here %d %d %d!\n", threadIdx.x, blockIdx.x, blockIdx.y);
          //}
        }
        else if constexpr (std::is_same_v<T, __half>) {
          Float4AsHalfs data;
          for (uint qq = 0; qq < 8; qq++) {
            data.h[qq] = registers[index2D(thread_tile_y, thread_tile_x + qq, THREAD_TILE_X_VALUE)];
          }
          reinterpret_cast<float4 *>(
            &target[index2D(tiled_i, tiled_j, n)])[0] = data.f4;
        }
        else {
          assert(0);
        }
        
      }
    }
  }
  else if constexpr (THREAD_TILE_X_VALUE == f2_multiple && local_x_chunk >= f2_multiple) {      
    for (int thread_tile_y = 0; thread_tile_y < THREAD_TILE_Y_VALUE; thread_tile_y++) {
      const uint tiled_i = get_global_id_y_tiled(thread_tile_y);
      for (int thread_tile_x = 0; thread_tile_x < THREAD_TILE_X_VALUE; thread_tile_x+=f2_multiple) {
        const uint tiled_j = get_global_id_x_tiled(thread_tile_x);
        if constexpr (std::is_same_v<T, float>) {
          float2 tmp;
          tmp.x = registers[index2D(thread_tile_y, thread_tile_x + 0, THREAD_TILE_X_VALUE)];
          tmp.y = registers[index2D(thread_tile_y, thread_tile_x + 1, THREAD_TILE_X_VALUE)];
          reinterpret_cast<float2 *>(
            &target[index2D(tiled_i, tiled_j, n)])[0] = tmp;
        }
        else if constexpr (std::is_same_v<T, __half>) {
          Float2AsHalfs data;
          for (uint qq = 0; qq < 4; qq++) {
            data.h[qq] = registers[index2D(thread_tile_y, thread_tile_x + qq, THREAD_TILE_X_VALUE)];
          }
          reinterpret_cast<float2 *>(
            &target[index2D(tiled_i, tiled_j, n)])[0] = data.f2;
        }
        else {
          assert(0);
        }
      }
    }
  }
  else if constexpr (THREAD_TILE_X_VALUE == f1_multiple && std::is_same_v<T, __half> && local_x_chunk >= f1_multiple) {      
    for (int thread_tile_y = 0; thread_tile_y < THREAD_TILE_Y_VALUE; thread_tile_y++) {
      const uint tiled_i = get_global_id_y_tiled(thread_tile_y);
      for (int thread_tile_x = 0; thread_tile_x < THREAD_TILE_X_VALUE; thread_tile_x+=f1_multiple) {
        const uint tiled_j = get_global_id_x_tiled(thread_tile_x);
        if constexpr (std::is_same_v<T, __half>) {
          Float1AsHalfs data;
          for (uint qq = 0; qq < 2; qq++) {
            data.h[qq] = registers[index2D(thread_tile_y, thread_tile_x + qq, THREAD_TILE_X_VALUE)];
          }
          reinterpret_cast<float *>(
            &target[index2D(tiled_i, tiled_j, n)])[0] = data.f1;
        }
        else {
          assert(0);
        }
      }
    }
  }
  else {
    for (int thread_tile_y = 0; thread_tile_y < THREAD_TILE_Y_VALUE; thread_tile_y++) {
      const uint tiled_i = get_global_id_y_tiled(thread_tile_y);
      for (int thread_tile_x = 0; thread_tile_x < THREAD_TILE_X_VALUE; thread_tile_x++) {
        const uint tiled_j = get_global_id_x_tiled(thread_tile_x);
        target[index2D(tiled_i,tiled_j,n)] = registers[index2D(thread_tile_y, thread_tile_x, THREAD_TILE_X_VALUE)];
      }
    }
  }
}