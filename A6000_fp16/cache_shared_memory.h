#include <assert.h>

#if !defined ASYNC_LOADS_VALUE
#define ASYNC_LOADS_VALUE 0
#endif

#if !defined get_local_id_y_tiled
#define get_local_id_y_tiled(thread_tile_y) (get_local_id_y() + thread_tile_y)
#endif

#if !defined get_local_id_x_tiled
#define get_local_id_x_tiled(thread_tile_x) (get_local_id_x() + thread_tile_x)
#endif

union Float4AsHalfs {
    float4 f4;
    __half h[8];
};

union Float2AsHalfs {
    float2 f2;
    __half h[4];
};

union Float1AsHalfs {
    float f1;
    __half h[1];
};

enum DimTiling {
    X_DIM_TILING,
    Y_DIM_TILING
};

template <typename T, int async_loads, bool registers>
void __device__ __forceinline__ cach_shared_memory_load_value(const T *src, T *target, cuda::barrier<cuda::thread_scope_block> *bar) {
    if constexpr (registers) {
        target[0] = src[0];
        return;
    }
    if constexpr (async_loads) {
        // Use async memcpy if async loads are enabled
        cuda::memcpy_async(target, src, sizeof(T), *bar);
        return;
    }
    target[0] = src[0];
    return;
}

#if ASYNC_LOADS_VALUE == 1
#define ASYNC_BARRIER() block_barrier->arrive_and_wait();
#define ASYNC_BARRIER_NUMBERED(stage) block_barrier[stage].arrive_and_wait();
#else 
#define ASYNC_BARRIER() __syncthreads();
#define ASYNC_BARRIER_NUMBERED(stage) __syncthreads();
#endif

// The __half implementation isn't quite right. CUDA doesn't support a __half4. Instead, we need to use a float4 and then cast to 8 __half4's.
template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_vec4(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, cuda::barrier<cuda::thread_scope_block> *bar, const uint offset = 0)
{
    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 8 : 4;

    const uint flattened_id = get_flattened_id();
    const uint new_j = (flattened_id*first_multiple) % (dim1);
    const uint new_i = (flattened_id*first_multiple) / (dim1);
    const uint stride_i = (flattened_dim*first_multiple) / (dim1);
    const T *src2 = src + src_dim0 * src_stride;
    int reg_index = 0;

    if constexpr (flattened_dim * first_multiple == (dim0 * dim1))
    {
        const uint i = new_i;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
            float4 * new_target = NULL;
            const float4 * new_source = NULL;
            if constexpr (!to_registers) {
                new_target = reinterpret_cast<float4 *>(&target[index2D(i, new_j, dim1 + offset)]);
            }
            else {
                new_target = reinterpret_cast<float4 *>(&target[0]);
            }

            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const float4 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const float4 *>(&src[0]);
            }

            cach_shared_memory_load_value<float4, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                  new_source,
                                                  new_target, 
                                                  bar);
            
        }
        else {
            assert(0);
            // implement this for quarter
        }
    }
    else if constexpr (flattened_dim * first_multiple >= dim1)
    {
        for (uint outer = 0; outer < dim0 / (stride_i); outer++)
        {
            const uint i = outer * stride_i + new_i;
            float4 * new_target = NULL;
            const float4 * new_source = NULL;

            if constexpr (!to_registers) {
                new_target = reinterpret_cast<float4 *>(&target[index2D(i, new_j, dim1 + offset)]);
            }
            else {
                new_target = reinterpret_cast<float4 *>(&target[outer*first_multiple]);
            }
           

            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const float4 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const float4 *>(&src[outer*first_multiple]);
            }
            
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
                
                cach_shared_memory_load_value<float4, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                    new_source,
                                                    new_target, 
                                                    bar);
            }
            else {
                assert(0);
                // implement this for quarter
            }
        }
    }
    else
    {
        for (int outer = 0; outer < dim0; outer++)
        {
            for (int inner = 0; inner < dim1 / flattened_dim; inner+=first_multiple)
            {
                float4 * new_target = NULL;
                const float4 * new_source = NULL;
                const uint j = (flattened_id*first_multiple + inner * flattened_dim);

                if constexpr (!to_registers) {
                    new_target = reinterpret_cast<float4 *>((&target[index2D(outer, j, dim1 + offset)]));
                }
                else {
                    new_target = reinterpret_cast<float4 *>(&target[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const float4 *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const float4 *>(&src[reg_index*first_multiple]);
                    reg_index += 1;
                }
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
                    //reinterpret_cast<float4 *>(&target[index2D(outer, j, dim1)])[0] = 
                    //    reinterpret_cast<const float4 *>(&src2[index2D(outer, src_dim1 + j, src_stride)])[0];
                    cach_shared_memory_load_value<float4, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                        new_source,
                                                        new_target, 
                                                        bar);
                }
                
                else {
                    assert(0);
                    // implement this for quarter
                }         
            }
        }
    }
}

// The __half implementation isn't quite right. CUDA doesn't support a __half4. Instead, we need to use a float4 and then cast to 8 __half4's.
template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_vec2(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, cuda::barrier<cuda::thread_scope_block> *bar, const uint offset = 0)
{

    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 4 : 2;

    const uint flattened_id = get_flattened_id();
    const uint new_j = (flattened_id*first_multiple) % (dim1);
    const uint new_i = (flattened_id*first_multiple) / (dim1);
    const uint stride_i = (flattened_dim*first_multiple) / (dim1);
    const T *src2 = src + src_dim0 * src_stride;
    int reg_index = 0;

    if constexpr (flattened_dim * first_multiple == (dim0 * dim1))
    {
        const uint i = new_i;
        float2 * new_target = NULL;
        const float2 * new_source = NULL;
        if constexpr (!to_registers) {
            new_target = reinterpret_cast<float2 *>(&target[index2D(i, new_j, dim1 + offset)]);
        }
        else {
            new_target = reinterpret_cast<float2 *>(&target[0]);
        }

        if constexpr (!from_registers) {
            new_source = reinterpret_cast<const float2 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
        }
        else {
            new_source = reinterpret_cast<const float2 *>(&src[0]);
        }

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {

            cach_shared_memory_load_value<float2, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                new_source,
                                                new_target, 
                                                bar);
        }
        else {
            assert(0);
            // implement this for quarter
        }
    }
    else if constexpr (flattened_dim * first_multiple >= dim1)
    {
        for (uint outer = 0; outer < dim0 / (stride_i); outer++)
        {
            const uint i = outer * stride_i + new_i;
            float2 * new_target = NULL;
            const float2 * new_source = NULL;

            if constexpr (!to_registers) {
                new_target = reinterpret_cast<float2 *>(&target[index2D(i, new_j, dim1 + offset)]);
            }
            else {
                new_target = reinterpret_cast<float2 *>(&target[outer*first_multiple]);
            }

            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const float2 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const float2 *>(&src[outer*first_multiple]);
            }

            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
                cach_shared_memory_load_value<float2, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                    new_source,
                                                    new_target, 
                                                    bar);
            }
            else {
                assert(0);
                // implement this for quarter
            }
        }
    }
    else
    {
        for (int outer = 0; outer < dim0; outer++)
        {
            for (int inner = 0; inner < dim1 / flattened_dim; inner+=first_multiple)
            {
                const uint j = (flattened_id*first_multiple + inner * flattened_dim);
                float2 * new_target = NULL;
                const float2 * new_source = NULL;
                if constexpr (!to_registers) {
                    new_target = reinterpret_cast<float2 *>((&target[index2D(outer, j, dim1 + offset)]));
                }
                else {
                    new_target = reinterpret_cast<float2 *>(&target[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const float2 *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const float2 *>(&src[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {

                    cach_shared_memory_load_value<float2, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                    new_source,
                                                    new_target, 
                                                    bar);

                }
                else {
                    assert(0);
                    // implement this for quarter
                }         
            }
        }
    }
}

template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_vec1(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, cuda::barrier<cuda::thread_scope_block> *bar, const uint offset = 0)
{

    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 2 : 1;

    const uint flattened_id = get_flattened_id();
    const uint new_j = (flattened_id*first_multiple) % (dim1);
    const uint new_i = (flattened_id*first_multiple) / (dim1);
    const uint stride_i = (flattened_dim*first_multiple) / (dim1);
    const T *src2 = src + src_dim0 * src_stride;
    int reg_index = 0;

    if constexpr (flattened_dim * first_multiple == (dim0 * dim1))
    {
        const uint i = new_i;
        float * new_target = NULL;
        const float * new_source = NULL;
        if constexpr (!to_registers) {
            new_target = reinterpret_cast<float *>(&target[index2D(i, new_j, dim1 + offset)]);
        }
        else {
            new_target = reinterpret_cast<float *>(&target[0]);
        }

        if constexpr (!from_registers) {
            new_source = reinterpret_cast<const float *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
        }
        else {
            new_source = reinterpret_cast<const float *>(&src[0]);
        }

        if constexpr (std::is_same_v<T, __half>) {
            cach_shared_memory_load_value<float, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                  new_source,
                                                  new_target, 
                                                  bar);
        }
        else {
            assert(0);
            // implement this for quarter
        }
    }
    else if constexpr (flattened_dim * first_multiple >= dim1)
    {
        for (uint outer = 0; outer < dim0 / (stride_i); outer++)
        {
            const uint i = outer * stride_i + new_i;

            float * new_target = NULL;
            const float * new_source = NULL;
            if constexpr (!to_registers) {
                new_target = reinterpret_cast<float *>(&target[index2D(i, new_j, dim1 + offset)]);
            }
            else {
                new_target = reinterpret_cast<float *>(&target[outer*first_multiple]);
            }

            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const float *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const float *>(&src[outer*first_multiple]);
            }

            if constexpr (std::is_same_v<T, __half>) {
                cach_shared_memory_load_value<float, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                    new_source,
                    new_target, 
                    bar);
            }
            else {
                assert(0);
                // implement this for quarter
            }
        }
    }
    else
    {
        for (int outer = 0; outer < dim0; outer++)
        {
            for (int inner = 0; inner < dim1 / flattened_dim; inner+=first_multiple)
            {
                const uint j = (flattened_id*first_multiple + inner * flattened_dim);
                float * new_target = NULL;
                const float * new_source = NULL;
                if constexpr (!to_registers) {
                    new_target = reinterpret_cast<float *>(&target[index2D(outer, j, dim1 + offset)]);
                }
                else {
                    new_target = reinterpret_cast<float *>(&target[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const float *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const float *>(&src[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (std::is_same_v<T, __half>) {
                    cach_shared_memory_load_value<float, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                                                                            new_source,
                                                                            new_target, 
                                                                            bar);
                }
                else {
                    assert(0);
                    // implement this for quarter
                }         
            }
        }
    }
}


template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_normal(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, cuda::barrier<cuda::thread_scope_block> *bar, const uint offset = 0)
{
    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 8 : 4;
    constexpr uint second_multiple = std::is_same_v<T, __half> ? 4 : 2;
    constexpr uint third_multiple  = std::is_same_v<T, __half> ? 2 : 1;

    if constexpr (dim1 >= first_multiple && flattened_dim * first_multiple <= dim0 * dim1)
    {
        cache_shared_memory_vec4<T, dim0, dim1, flattened_dim, to_registers, from_registers>(target, src, src_dim0, src_dim1, src_stride, bar, offset);
    }
    else if constexpr (dim1 >= second_multiple && flattened_dim * second_multiple <= dim0 * dim1) {
        cache_shared_memory_vec2<T,dim0, dim1, flattened_dim, to_registers, from_registers>(target, src, src_dim0, src_dim1, src_stride, bar, offset);
    }
    else if constexpr (std::is_same_v<T, __half> && dim1 >= third_multiple && flattened_dim * third_multiple <=dim0 * dim1) {
        cache_shared_memory_vec1<T,dim0, dim1, flattened_dim, to_registers, from_registers>(target, src, src_dim0, src_dim1, src_stride, bar, offset);
    }
    else
    {
        const uint flattened_id = get_flattened_id();
        const uint new_j = flattened_id % dim1;
        const uint new_i = flattened_id / dim1;
        const uint stride_i = flattened_dim / dim1;
        const T *src2 = src + src_dim0 * src_stride;
        int reg_index = 0;

        if constexpr (flattened_dim >dim0 * dim1)
        {
            const uint i = new_i;
            T * new_target = NULL;
            const T * new_source = NULL;
            if constexpr (!to_registers) {
                new_target = reinterpret_cast<T *>(&target[index2D(i, new_j, dim1 + offset)]);
            }
            else {
                new_target = reinterpret_cast<T *>(&target[0]);
            }

            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const T *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const T *>(&src[0]);
            }

            
            if (index2D(i, new_j, dim1) <dim0 * dim1)
            {
                cach_shared_memory_load_value<T, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                    new_source,
                    new_target, 
                    bar);
                
            }
        }
        else if constexpr (flattened_dim ==dim0 * dim1)
        {
            const uint i = new_i;
            T * new_target = NULL;
            const T * new_source = NULL;
            if constexpr (!to_registers) {
                new_target = reinterpret_cast<T *>(&target[index2D(i, new_j, dim1 + offset)]);
            }
            else {
                new_target = reinterpret_cast<T *>(&target[0]);
            }

            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const T *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const T *>(&src[0]);
            }

            cach_shared_memory_load_value<T, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                new_source,
                new_target,
                bar);
        }
        else if constexpr (flattened_dim >= dim1)
        {
            for (uint outer = 0; outer <dim0 / (stride_i); outer++)
            {
                const uint i = outer * stride_i + new_i;
                T * new_target = NULL;
                const T * new_source = NULL;
                if constexpr (!to_registers) {
                    new_target = reinterpret_cast<T *>(&target[index2D(i, new_j, dim1 + offset)]);
                }
                else {
                    new_target = reinterpret_cast<T *>(&target[outer]);
                }
                
                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const T *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const T *>(&src[outer]);
                }

                cach_shared_memory_load_value<T, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                    new_source,
                    new_target,
                    bar);
            }
        }
        else
        {
            for (int outer = 0; outer <dim0; outer++)
            {
                for (int inner = 0; inner < dim1 / flattened_dim; inner++)
                {
                    const uint j = flattened_id + inner * flattened_dim;
                    T * new_target = NULL;
                    const T * new_source = NULL;
                    if constexpr (!to_registers) {
                        new_target = reinterpret_cast<T *>(&target[index2D(outer, j, dim1 + offset)]);
                    }
                    else {
                        new_target = reinterpret_cast<T *>(&target[reg_index]);
                        reg_index += 1;
                    }

                    if constexpr (!from_registers) {
                        new_source = reinterpret_cast<const T *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                    }
                    else {
                        new_source = reinterpret_cast<const T *>(&src[reg_index]);
                        reg_index += 1;
                    }

                    cach_shared_memory_load_value<T, ASYNC_LOADS_VALUE, to_registers || from_registers>(
                        new_source,
                        new_target, 
                        bar);
                }
            }
        }
    }
}

template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_vec4_transpose(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, const uint offset = 0)
{

    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 8 : 4;


    const uint flattened_id = get_flattened_id();
    const uint new_j = (flattened_id*first_multiple) % (dim1);
    const uint new_i = (flattened_id*first_multiple) / (dim1);
    const uint stride_i = (flattened_dim*first_multiple) / (dim1);
    const T *src2 = src + src_dim0 * src_stride;
    int reg_index = 0;

    if constexpr (flattened_dim * first_multiple == (dim0 * dim1))
    {
        const uint i = new_i;
        const float4 * new_source = NULL;
        if constexpr (!from_registers) {
            new_source = reinterpret_cast<const float4 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
        }
        else {
            new_source = reinterpret_cast<const float4 *>(&src[0]);
        }

        if constexpr (std::is_same_v<T, float>) {
            float4 tmp = reinterpret_cast<const float4 *>(new_source)[0];
            target[index2D(new_j + 0, i,dim0 + offset)] = tmp.x;
            target[index2D(new_j + 1, i,dim0 + offset)] = tmp.y;
            target[index2D(new_j + 2, i,dim0 + offset)] = tmp.z;
            target[index2D(new_j + 3, i,dim0 + offset)] = tmp.w;        
        }
        else if constexpr (std::is_same_v<T, __half>) {
            Float4AsHalfs data;
            data.f4 = reinterpret_cast<const float4 *>(new_source)[0];
            for (uint qq = 0; qq < 8; qq++) {
                target[index2D(new_j + qq, i,dim0 + offset)] = data.h[qq];
            }
           
        }
        else {
            assert(0);
            // implement this for quarter

        }
    }
    else if constexpr (flattened_dim * first_multiple >= dim1)
    {
        for (uint outer = 0; outer <dim0 / (stride_i); outer++)
        {
            const uint i = outer * stride_i + new_i;
            const float4 * new_source = NULL;
            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const float4 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const float4 *>(&src[outer*first_multiple]);
            }

            if constexpr (std::is_same_v<T, float>) {
                float4 tmp = reinterpret_cast<const float4 *>(new_source)[0];
                target[index2D(new_j + 0, i,dim0 + offset)] = tmp.x;
                target[index2D(new_j + 1, i,dim0 + offset)] = tmp.y;
                target[index2D(new_j + 2, i,dim0 + offset)] = tmp.z;
                target[index2D(new_j + 3, i,dim0 + offset)] = tmp.w; 
            }
            else if constexpr (std::is_same_v<T, __half>) {
                Float4AsHalfs data;
                data.f4 = reinterpret_cast<const float4 *>(new_source)[0];
                for (uint qq = 0; qq < 8; qq++) {
                    target[index2D(new_j + qq, i,dim0 + offset)] = data.h[qq];
                }
            }
            else {
                assert(0);
                // implement this for quarter
            }
        }
    }
    else
    {
        for (int outer = 0; outer <dim0; outer++)
        {
            for (int inner = 0; inner < dim1 / flattened_dim; inner+=first_multiple)
            {
                const uint j = (flattened_id*first_multiple + inner * flattened_dim);
                const float4 * new_source = NULL;
                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const float4 *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const float4 *>(&src[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (std::is_same_v<T, float>) {
                    float4 tmp = reinterpret_cast<const float4 *>(new_source)[0];
                    target[index2D(j + 0, outer,dim0 + offset)] = tmp.x;
                    target[index2D(j + 1, outer,dim0 + offset)] = tmp.y;
                    target[index2D(j + 2, outer,dim0 + offset)] = tmp.z;
                    target[index2D(j + 3, outer,dim0 + offset)] = tmp.w;                         
                }
                else if constexpr (std::is_same_v<T, __half>) {
                    Float4AsHalfs data;
                    data.f4 = reinterpret_cast<const float4 *>(new_source)[0];
                    for (uint qq = 0; qq < 8; qq++) {
                        target[index2D(j + qq, outer,dim0 + offset)] = data.h[qq];
                    }
                    
                }
                else {
                    assert(0);
                    // implement this for quarter
                }         
            }
        }
    }
}

template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_vec2_transpose(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, const uint offset = 0)
{

    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 4 : 2;


    const uint flattened_id = get_flattened_id();
    const uint new_j = (flattened_id*first_multiple) % (dim1);
    const uint new_i = (flattened_id*first_multiple) / (dim1);
    const uint stride_i = (flattened_dim*first_multiple) / (dim1);
    const T *src2 = src + src_dim0 * src_stride;
    int reg_index = 0;


    if constexpr (flattened_dim * first_multiple == (dim0 * dim1))
    {
        const uint i = new_i;
        const float2 * new_source = NULL;
        if constexpr (!from_registers) {
            new_source = reinterpret_cast<const float2 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
        }
        else {
            new_source = reinterpret_cast<const float2 *>(&src[0]);
        }
        if constexpr (std::is_same_v<T, float>) {
            float2 tmp = reinterpret_cast<const float2 *>(new_source)[0];
            target[index2D(new_j + 0, i,dim0 + offset)] = tmp.x;
            target[index2D(new_j + 1, i,dim0 + offset)] = tmp.y;       
        }
        else if constexpr (std::is_same_v<T, __half>) {
            Float2AsHalfs data;
            data.f2 = reinterpret_cast<const float2 *>(new_source)[0];
            for (uint qq = 0; qq < 4; qq++) {
                target[index2D(new_j + qq, i,dim0)] = data.h[qq];
            }
           
        }
        else {
            assert(0);
            // implement this for quarter

        }
    }
    else if constexpr (flattened_dim * first_multiple >= dim1)
    {
        for (uint outer = 0; outer <dim0 / (stride_i); outer++)
        {
            const uint i = outer * stride_i + new_i;
            const float2 * new_source = NULL;
            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const float2 *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const float2 *>(&src[outer*first_multiple]);
            }
            if constexpr (std::is_same_v<T, float>) {
                float2 tmp = reinterpret_cast<const float2 *>(new_source)[0];
                target[index2D(new_j + 0, i,dim0)] = tmp.x;
                target[index2D(new_j + 1, i,dim0)] = tmp.y;
            }
            else if constexpr (std::is_same_v<T, __half>) {
                Float2AsHalfs data;
                data.f2 = reinterpret_cast<const float2 *>(new_source)[0];
                for (uint qq = 0; qq < 4; qq++) {
                    target[index2D(new_j + qq, i,dim0)] = data.h[qq];
                }
                
            }
            else {
                assert(0);
                // implement this for quarter
            }
        }
    }
    else
    {
        for (int outer = 0; outer <dim0; outer++)
        {
            for (int inner = 0; inner < dim1 / flattened_dim; inner+=first_multiple)
            {
                const uint j = (flattened_id*first_multiple + inner * flattened_dim);
                const float2 * new_source = NULL;
                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const float2 *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const float2 *>(&src[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (std::is_same_v<T, float>) {
                    float2 tmp = reinterpret_cast<const float2 *>(new_source)[0];
                    target[index2D(j + 0, outer,dim0)] = tmp.x;
                    target[index2D(j + 1, outer,dim0)] = tmp.y;                       
                }
                else if constexpr (std::is_same_v<T, __half>) {
                    Float2AsHalfs data;
                    data.f2 = reinterpret_cast<const float2 *>(new_source)[0];
                    for (uint qq = 0; qq < 4; qq++) {
                        target[index2D(j + qq, outer,dim0)] = data.h[qq];
                    }
                    
                }
                else {
                    assert(0);
                    // implement this for quarter
                }         
            }
        }
    }
}

template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_vec1_transpose(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, const uint offset = 0) {

    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 2 : 1;


    const uint flattened_id = get_flattened_id();
    const uint new_j = (flattened_id*first_multiple) % (dim1);
    const uint new_i = (flattened_id*first_multiple) / (dim1);
    const uint stride_i = (flattened_dim*first_multiple) / (dim1);
    const T *src2 = src + src_dim0 * src_stride;
    int reg_index = 0;

    if constexpr (flattened_dim * first_multiple == (dim0 * dim1))
    {
        const uint i = new_i;
        const float * new_source = NULL;
        if constexpr (!from_registers) {
            new_source = reinterpret_cast<const float *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
        }
        else {
            new_source = reinterpret_cast<const float *>(&src[0]);
        }
        if constexpr (std::is_same_v<T, __half>) {
            Float1AsHalfs data;
            data.f1 = reinterpret_cast<const float *>(new_source)[0];
            for (uint qq = 0; qq < 2; qq++) {
                target[index2D(new_j + qq, i, dim0)] = data.h[qq];
            }          
        }
        else {
            assert(0);
            // implement this for quarter

        }
    }
    else if constexpr (flattened_dim * first_multiple >= dim1)
    {
        for (uint outer = 0; outer <dim0 / (stride_i); outer++)
        {
            const float * new_source = NULL;
            const uint i = outer * stride_i + new_i;

            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const float *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const float *>(&src[outer*first_multiple]);
            }

            if constexpr (std::is_same_v<T, __half>) {
                Float1AsHalfs data;
                data.f1 = reinterpret_cast<const float *>(new_source)[0];
                for (uint qq = 0; qq < 2; qq++) {
                    target[index2D(new_j + qq, i,dim0)] = data.h[qq];
                }
                
            }
            else {
                assert(0);
                // implement this for quarter
            }
        }
    }
    else
    {
        for (int outer = 0; outer <dim0; outer++)
        {
            for (int inner = 0; inner < dim1 / flattened_dim; inner+=first_multiple)
            {
                const uint j = (flattened_id*first_multiple + inner * flattened_dim);
                const float * new_source = NULL;
                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const float *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const float *>(&src[reg_index*first_multiple]);
                    reg_index += 1;
                }

                if constexpr (std::is_same_v<T, __half>) {
                    Float1AsHalfs data;
                    data.f1 = reinterpret_cast<const float *>(new_source)[0];
                    for (uint qq = 0; qq < 2; qq++) {
                        target[index2D(j + qq, outer,dim0)] = data.h[qq];
                    }
                    
                }
                else {
                    assert(0);
                    // implement this for quarter
                }         
            }
        }
    }
}


template <typename T, int dim0, int dim1, int flattened_dim, bool to_registers = false, bool from_registers = false>
__device__ __forceinline__ void cache_shared_memory_transpose(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, cuda::barrier<cuda::thread_scope_block> *bar = NULL, const uint offset = 0)
{

    assert(!to_registers);
    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 8 : 4;
    constexpr uint second_multiple = std::is_same_v<T, __half> ? 4 : 2;
    constexpr uint third_multiple  = std::is_same_v<T, __half> ? 2 : 1;

    if constexpr (dim1 >= first_multiple && flattened_dim * first_multiple <=dim0 * dim1)
    {
        cache_shared_memory_vec4_transpose<T,dim0, dim1, flattened_dim, to_registers, from_registers>(target, src, src_dim0, src_dim1, src_stride, offset);
    }
    else if constexpr (dim1 >= second_multiple && flattened_dim * second_multiple <=dim0 * dim1)
    {
        cache_shared_memory_vec2_transpose<T,dim0, dim1, flattened_dim, to_registers, from_registers>(target, src, src_dim0, src_dim1, src_stride, offset);
    }
    else if constexpr (std::is_same_v<T, __half> && dim1 >= third_multiple && flattened_dim * third_multiple <=dim0 * dim1) {
        cache_shared_memory_vec1_transpose<T,dim0, dim1, flattened_dim, to_registers, from_registers>(target, src, src_dim0, src_dim1, src_stride, offset);
    }
    else
    {
        const uint flattened_id = get_flattened_id();
        const uint new_j = flattened_id % dim1;
        const uint new_i = flattened_id / dim1;
        const uint stride_i = flattened_dim / dim1;
        const T *src2 = src + src_dim0 * src_stride;
        int reg_index = 0;
        
        if constexpr (flattened_dim >dim0 * dim1)
        {
            const T * new_source = NULL;
            const uint i = new_i;
            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const T *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const T *>(&src[0]);
            }

            if (index2D(i, new_j, dim1) <dim0 * dim1)
            {
                //target[index2D(new_j, i,dim0)] = src2[index2D(i, src_dim1 + new_j, src_stride)];
                cach_shared_memory_load_value<T, to_registers || from_registers>(
                    new_source,
                    &target[index2D(new_j, i,dim0)], 
                    bar);
            }

        }
        else if constexpr (flattened_dim ==dim0 * dim1)
        {
            const uint i = new_i;
            const T * new_source = NULL;
            if constexpr (!from_registers) {
                new_source = reinterpret_cast<const T *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
            }
            else {
                new_source = reinterpret_cast<const T *>(&src[0]);
            }

            cach_shared_memory_load_value<T, to_registers || from_registers>(
                    new_source,
                    &target[index2D(new_j, i,dim0)], 
                    bar);
        }
        else if constexpr (flattened_dim >= dim1)
        {
            for (uint outer = 0; outer <dim0 / (stride_i); outer++)
            {
                const T * new_source = NULL;
                const uint i = outer * stride_i + new_i;
                if constexpr (!from_registers) {
                    new_source = reinterpret_cast<const T *>(&src2[index2D(i, src_dim1 + new_j, src_stride)]);
                }
                else {
                    new_source = reinterpret_cast<const T *>(&src[outer]);
                }
                cach_shared_memory_load_value<T, to_registers || from_registers>(
                    new_source,
                    &target[index2D(new_j, i,dim0)], 
                    bar);
            }
        }
        else
        {
            for (int outer = 0; outer <dim0; outer++)
            {
                for (int inner = 0; inner < dim1 / flattened_dim; inner++)
                {
                    const T * new_source = NULL;
                    const uint j = flattened_id + inner * flattened_dim;

                    if constexpr (!from_registers) {
                        new_source = reinterpret_cast<const T *>(&src2[index2D(outer, src_dim1 + j, src_stride)]);
                    }
                    else {
                        new_source = reinterpret_cast<const T *>(&src[reg_index]);
                        reg_index += 1;
                    }
                    cach_shared_memory_load_value<T, to_registers || from_registers>(
                        new_source,
                        &target[index2D(j, outer,dim0)], 
                        bar);
                }
            }
        }
    }
}

template <typename T, int dim0, int dim1, int flattened_dim, bool transpose>
__device__ __forceinline__ void cache_shared_memory(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride, cuda::barrier<cuda::thread_scope_block> *bar = NULL, const uint offset = 0) {
    if constexpr (transpose) {
        cache_shared_memory_transpose<T,dim1, dim0, flattened_dim>(target, src, src_dim1, src_dim0, src_stride, bar, offset);
    } else {
        cache_shared_memory_normal<T, dim0, dim1, flattened_dim, false>(target, src, src_dim0, src_dim1, src_stride, bar, offset);
    }
}

template <typename T, int dim0, int dim1, int flattened_dim, bool transpose>
__device__ __forceinline__ void cache_to_registers(T *target, const T *src, const int src_dim0, const int src_dim1, const int src_stride) {
    if constexpr (transpose) {
        assert(0);
        cache_shared_memory_normal<T, dim1, dim0, flattened_dim, true>(target, src, src_dim1, src_dim0, src_stride, nullptr, 0);
    }
    else {
        cache_shared_memory_normal<T, dim0, dim1, flattened_dim, true>(target, src, src_dim0, src_dim1, src_stride, nullptr, 0);
    }
}

template <typename T, int dim0, int dim1, int flattened_dim, bool transpose>
__device__ __forceinline__ void registers_to_shmem(T *target, const T *src, const uint offset = 0) {
    if constexpr (transpose) {
        cache_shared_memory_transpose<T, dim1, dim0, flattened_dim, false, true>(target, src, -256, -256, -256, nullptr, offset);
    }
    else {
        cache_shared_memory_normal<T, dim0, dim1, flattened_dim, false, true>(target, src, -256, -256, -256, nullptr, offset);
    }
}

template <typename T, int number, int shmem_stride, DimTiling dt>
__device__ __forceinline__ void copy_from_shared_column_to_registers(T* target, const T* src, const int row, const uint offset = 0) {
    int col = 0;
    for (int i = 0; i < number; i++) {        
        if constexpr(dt == Y_DIM_TILING) {
            col = get_local_id_y_tiled(i);
        }
        else {
            col = get_local_id_x_tiled(i);
        }
        target[i] = src[index2D(col, row, shmem_stride + offset)];
    }
}

template <typename T, int number, int shmem_stride, DimTiling dt, int local_chunk_size = 256>
__device__ __forceinline__ void copy_from_shared_row_to_registers(T* target, const T* src, const int col, const uint offset = 0) {
    constexpr uint first_multiple  = std::is_same_v<T, __half> ? 8 : 4;
    constexpr uint second_multiple = std::is_same_v<T, __half> ? 4 : 2;
    constexpr uint third_multiple  = std::is_same_v<T, __half> ? 2 : 1;
    int row = -1;

    if constexpr (number >= first_multiple && local_chunk_size >= first_multiple) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
            for (int i = 0; i < number; i += first_multiple) {
                if constexpr(dt == Y_DIM_TILING) {
                    row = get_local_id_y_tiled(i);
                }
                else {
                    row = get_local_id_x_tiled(i);
                }
                reinterpret_cast<float4 *>(target + i)[0] = reinterpret_cast<const float4 *>(src + index2D(col, row,  shmem_stride + offset))[0];
            }
        }
        else {
            assert(0);
            // implement this for quarter
        }
    }
    else if constexpr (number >= second_multiple && local_chunk_size >= second_multiple) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
            for (int i = 0; i < number; i += second_multiple) {
                if constexpr(dt == Y_DIM_TILING) {
                    row = get_local_id_y_tiled(i);
                }
                else {
                    row = get_local_id_x_tiled(i);
                }
                reinterpret_cast<float2 *>(target + i)[0] = reinterpret_cast<const float2 *>(src + index2D(col, row,  shmem_stride + offset))[0];
            }
        }
        else {
            assert(0);
            // implement this for quarter
        }
    }
    else if constexpr (number >= third_multiple && local_chunk_size >= third_multiple) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __half>) {
            for (int i = 0; i < number; i += third_multiple) {
                if constexpr(dt == Y_DIM_TILING) {
                    row = get_local_id_y_tiled(i);
                }
                else {
                    row = get_local_id_x_tiled(i);
                }
                reinterpret_cast<float *>(target + i)[0] = reinterpret_cast<const float *>(src + index2D(col, row,  shmem_stride + offset))[0];
            }
        }
        else {
            assert(0);
            // implement this for quarter
        }
    }
    else {
        // for cases that aren't implemented yet
        assert(0);
    }
}