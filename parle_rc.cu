#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <random>

#define WARP_SIZE 32

__inline__ __device__ int warpScanInclusive(int val, unsigned mask = 0xffffffff)
{
    // offset = 1, 2, 4, 8, 16 ...
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
         
        int temp = __shfl_up_sync(mask, val, offset);
       
        if (threadIdx.x >= offset) {
            val += temp;
        }
    }
    return val;
}

__inline__ __device__ void blockPrefixSum(int* sdata)
{
    int tid = threadIdx.x;

    int warpId = tid / WARP_SIZE;

    int lane   = tid % WARP_SIZE;

    int val = sdata[tid];
    val = warpScanInclusive(val);  
    sdata[tid] = val;              

    if (lane == (WARP_SIZE - 1)) {
        sdata[warpId] = val;  
    }

    __syncthreads(); 

    if (warpId == 0) {
        int numWarp = (blockDim.x + (WARP_SIZE - 1)) / WARP_SIZE;
        int myWarpLane = lane; 

        int warpTotal = 0;
        if (myWarpLane < numWarp) {
            warpTotal = sdata[myWarpLane];
        }

        warpTotal = warpScanInclusive(warpTotal);

        if (myWarpLane < numWarp) {
            sdata[myWarpLane] = warpTotal;
        }
    }

    __syncthreads(); 

    int blockSumBeforeMyWarp = 0;
    if (warpId > 0) {

        blockSumBeforeMyWarp = sdata[warpId - 1];
    }
    sdata[tid] += blockSumBeforeMyWarp;
}


/**
 * Single kernel implementation of Run Length Encoding (RLE)
 * Combines boundary marking, prefix sum, and run extraction into one kernel
 * Uses shared memory for block-level operations and atomic operations for global coordination
 */
__global__ void rle_encode_kernel(const int* input, int n,
                                int* values, int* counts,
                                int* num_runs) {
    // Shared memory layout: [boundary_flags][prefix_sums]
    extern __shared__ int sdata[];
    int* s_flags = sdata;                     // Boundary flags array
    int* s_prefix_sum = &sdata[blockDim.x];   // Prefix sum array
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Mark boundaries between runs
    int cur  = (gid < n) ? input[gid] : 0;
    int prev = (gid > 0) ? input[gid - 1] : cur;

    int is_boundary = (gid < n) && ((gid == 0) || (cur != prev));

    s_flags[tid] = is_boundary;
    __syncthreads();
    
    s_prefix_sum[tid] = s_flags[tid];
    __syncthreads();
        
    blockPrefixSum(s_prefix_sum);
    __syncthreads();
    
    // Calculate number of runs in this block
    int block_runs = 0;
    if (tid == blockDim.x - 1) {
        block_runs = s_prefix_sum[tid];
    } else if (gid + 1 == n) {
        block_runs = s_prefix_sum[tid];
    }
    __syncthreads();
    
    // Use atomic operation to get global offset
    int global_offset = 0;
    if (tid == 0 && block_runs > 0) {
        global_offset = atomicAdd(num_runs, block_runs);
    }
    __syncthreads();
    
    const unsigned int full_mask = 0xffffffff;
    bool my_boundary = (gid < n && is_boundary);
    // Check if any thread in the warp has a boundary.
    bool any_boundary = __any_sync(full_mask, my_boundary);

    if (any_boundary) {
        // Collect a bitmask of threads that are boundaries in the warp.
        unsigned int boundary_mask = __ballot_sync(full_mask, my_boundary);
        int num_boundaries = __popc(boundary_mask);

        if (num_boundaries == 1) {
            // Broadcast the run start index and run value from the leader (first boundary in the warp).
            int my_run_start = my_boundary ? gid : 0;
            int my_run_val   = my_boundary ? cur : 0;

            // Identify the leader lane: the first boundary in the warp.
            int leader_lane = __ffs(boundary_mask) - 1;  // __ffs returns 1-indexed, so subtract 1.
            int warp_run_start = __shfl_sync(full_mask, my_run_start, leader_lane);
            int warp_run_val   = __shfl_sync(full_mask, my_run_val,   leader_lane);

            // Cooperative scanning: All threads in the warp collaborate.
            // Each thread examines a candidate index in batches of 32.
            int offset = 0;
            int run_end_candidate = warp_run_start;  // Initialize the candidate.

            while (true) {
                // Each thread computes a candidate index based on its lane.
                int lane = threadIdx.x & 31;
                int candidate_idx = warp_run_start + offset + lane;
                // Check if the candidate index is within bounds and has the same value as the run.
                bool valid = (candidate_idx < n) && (input[candidate_idx] == warp_run_val);
                // Collect the validity mask from the entire warp.
                unsigned int mask = __ballot_sync(full_mask, valid);

                // If not all lanes are valid, then the first false indicates the end of the run.
                if (mask != full_mask) {
                    int first_false = __ffs(~mask) - 1; // 0-indexed lane where false occurred.
                    run_end_candidate = warp_run_start + offset + first_false - 1;
                    break;
                }

                // Otherwise, move to the next batch of 32 elements.
                offset += 32;
                if (warp_run_start + offset >= n) {
                    run_end_candidate = n - 1;
                    break;
                }
            }

            // Broadcast the final run end candidate from the leader to all threads in the warp.
            int final_run_end = __shfl_sync(full_mask, run_end_candidate, leader_lane);

            // The boundary thread writes the output.
            if (my_boundary) {
                int local_idx = s_prefix_sum[tid] - 1;  // Local run index within the block.
                int global_idx = global_offset + local_idx;
                values[global_idx] = warp_run_val;
                counts[global_idx] = final_run_end - warp_run_start + 1;
            }

        } else {
            // ---- Fallback: Multiple boundaries in the same warp; use serial scanning ----
            if (my_boundary) {
                int run_end_serial = gid;
                while (run_end_serial + 1 < n && input[run_end_serial + 1] == cur)
                    run_end_serial++;
                int local_idx = s_prefix_sum[tid] - 1;
                int global_idx = global_offset + local_idx;
                values[global_idx] = cur;
                counts[global_idx] = run_end_serial - gid + 1;
            }
        }
    }
}


/**
 * Main RLE encoding function
 */
cudaError_t run_length_encode(const int* d_input, int n,
                            int* d_values, int* d_counts,
                            int* num_runs) {
    const int BLOCK_SIZE = 512;
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int sdata_size = 2 * BLOCK_SIZE * sizeof(int);
    
    cudaMemset(num_runs, 0, sizeof(int));
    
    rle_encode_kernel<<<num_blocks, BLOCK_SIZE, sdata_size>>>(
        d_input, n, d_values, d_counts, num_runs);
    
    return cudaGetLastError();
}

/**
 * Generate random test data
 */
int* generateRandomData(int n) {
    int* data = new int[n];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9);
    
    for (int i = 0; i < n; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

/**
 * Generate compressible test data
 */
int* generateCompressibleData(int n, float change_prob = 0.2) {
    int* data = new int[n];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> val_dis(0, 9);
    
    int current_val = val_dis(gen);
    for (int i = 0; i < n; ++i) {
        if (dis(gen) < change_prob) {
            current_val = val_dis(gen);
        }
        data[i] = current_val;
    }
    return data;
}

/**
 * Run benchmark for a specific input size
 */
 void runBenchmarkWithDisplay(int size, bool use_compressible_data = false) {
    // Generate test data
    int* h_input = use_compressible_data ? 
                   generateCompressibleData(size) : 
                   generateRandomData(size);
    
    // Allocate device memory
    int *d_input, *d_values, *d_counts, *d_num_runs;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_values, size * sizeof(int));
    cudaMalloc(&d_counts, size * sizeof(int));
    cudaMalloc(&d_num_runs, sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Run encoding
    int num_runs;
    cudaError_t err = run_length_encode(d_input, size, d_values, d_counts, d_num_runs);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Make sure kernel execution is complete
    cudaDeviceSynchronize();
    
    // Copy number of runs back to host
    cudaMemcpy(&num_runs, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Check if we got valid number of runs
    if (num_runs <= 0) {
        printf("Error: Invalid number of runs (%d)\n", num_runs);
        return;
    }
    
    // Allocate host memory for results
    int* h_values = new int[num_runs];
    int* h_counts = new int[num_runs];
    
    // Copy results back to host
    cudaMemcpy(h_values, d_values, num_runs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_counts, num_runs * sizeof(int), cudaMemcpyDeviceToHost);
    
    
    // Run actual benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int NUM_ITERATIONS = 100;
    float total_time = 0;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cudaEventRecord(start);
        run_length_encode(d_input, size, d_values, d_counts, d_num_runs);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    
    // Calculate and print benchmark results
    float avg_time = total_time / NUM_ITERATIONS;
    float bytes_processed = static_cast<float>(size) * sizeof(int);
    float gb_per_s = (bytes_processed / (avg_time * 1e-3)) / 1e9;
    
    if (size >= 1000000) {
        printf("| %dM      | %-15.3f | %-15.2f |\n", 
               size/1000000, avg_time, gb_per_s);
    } else if (size >= 1000) {
        printf("| %dK      | %-15.3f | %-15.2f |\n", 
               size/1000, avg_time, gb_per_s);
    } else {
        printf("| %d       | %-15.3f | %-15.2f |\n", 
               size, avg_time, gb_per_s);
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_values;
    delete[] h_counts;
    cudaFree(d_input);
    cudaFree(d_values);
    cudaFree(d_counts);
    cudaFree(d_num_runs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int test_sizes[] = {
        10000,     // 10K
        50000,     // 50K
        100000,    // 100K
        200000,    // 200K
        500000,    // 500K
        1000000,   // 1M
        2000000,   // 2M
        5000000,   // 5M
        10000000,  // 10M
        20000000,  // 20M
        40000000   // 40M
    };
    
    // Random data benchmark
    printf("\nRandom Data Benchmark:\n");
    printf("| %-8s | %-15s | %-15s |\n", "Size", "Average Time (ms)", "Throughput (GB/s)");
    printf("|----------|-----------------|------------------|\n");
    
    for (int size : test_sizes) {
        runBenchmarkWithDisplay(size, false);
    }
    
    // Compressible data benchmark
    printf("\nCompressible Data Benchmark:\n");
    printf("| %-8s | %-15s | %-15s |\n", "Size", "Average Time (ms)", "Throughput (GB/s)");
    printf("|----------|-----------------|------------------|\n");
    
    for (int size : test_sizes) {
        runBenchmarkWithDisplay(size, true);
    }
    
    return 0;
}