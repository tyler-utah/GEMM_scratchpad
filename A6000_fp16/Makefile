#input values. This is just square matrixes. Just change the top number to change the size of the input.
n_VALUE           := 2048
A_LEN             := -DA_LEN=$(shell echo $$(($(n_VALUE)*$(n_VALUE))))
B_LEN             := -DB_LEN=$(shell echo $$(($(n_VALUE)*$(n_VALUE))))
C_LEN             := -DC_LEN=$(shell echo $$(($(n_VALUE)*$(n_VALUE))))
n_VALUE_DEFINE    := -Dn_VALUE=$(n_VALUE)

#tuning parameters
THREADS_X_VALUE   := -DTHREADS_X_VALUE=16
THREADS_Y_VALUE   := -DTHREADS_Y_VALUE=8
TUNE_SPLIT_VALUE  := -DTUNE_SPLIT_VALUE=32 # K split
WARP_SHAPE_X_VALUE:= -DWARP_SHAPE_X_VALUE=16
THREAD_TILE_X_VALUE:= -DTHREAD_TILE_X_VALUE=1
THREAD_TILE_Y_VALUE:= -DTHREAD_TILE_Y_VALUE=8
WARP_TILE_X_VALUE := -DWARP_TILE_X_VALUE=8
WARP_TILE_Y_VALUE := -DWARP_TILE_Y_VALUE=4
ASYNC_LOADS_VALUE := -DASYNC_LOADS_VALUE=0
SWAP_XY_IJ_BLOCK_MAPPING_VALUE := -DSWAP_XY_IJ_BLOCK_MAPPING_VALUE=1
A_OFFSET_VALUE    := -DA_OFFSET_VALUE=0
B_OFFSET_VALUE    := -DB_OFFSET_VALUE=3

# how many times to repeat the kernel execution
REPEAT            := -DREPEAT=500
#VALIDATE		  := -DVALIDATE=1
VALIDATE		  := -DVALIDATE=0

all: SharedMemoryOffset

SharedMemoryOffset: driver.cu kernel.h cache_shared_memory.h write_output.h
	nvcc -DINCLUDE_HOST -gencode=arch=compute_86,code=sm_86 $(REPEAT) $(VALIDATE) -O3 -I./ $(KERNEL) $(A_LEN) $(B_LEN) $(C_LEN) $(n_VALUE_DEFINE) $(THREADS_X_VALUE) $(THREADS_Y_VALUE) $(TUNE_SPLIT_VALUE) $(WARP_SHAPE_X_VALUE) $(THREAD_TILE_X_VALUE) $(THREAD_TILE_Y_VALUE) $(WARP_TILE_X_VALUE) $(WARP_TILE_Y_VALUE) $(ASYNC_LOADS_VALUE) $(SWAP_XY_IJ_BLOCK_MAPPING_VALUE) $(A_OFFSET_VALUE) $(B_OFFSET_VALUE) $(GENCODE) driver.cu -o to_run

clean:
	rm to_run