#include <iostream>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <chrono>
#include <curand_kernel.h>

#define CUDA_SAFE_CALL(call) \
do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

const int WIDTH = 512;
const int HEIGHT = 512;
const int DEPTH = 512;
const int SIZE = WIDTH * HEIGHT * DEPTH;

const int ITERATIONS = 110;

__device__
int indexAt(int x, int y, int z) {
    return WIDTH * HEIGHT * z + WIDTH * y + x;
}

__device__
int indexAtWithPadding(int x, int y, int z) {
    return (WIDTH + 2) * (HEIGHT + 2) * z + (WIDTH + 2) * y + x;
}

__device__
bool doubleEquals(double a, double b) {
    return fabs(a - b) < 1e-8;
}

__global__
void setupRandom(curandState *random, int seed) {
    curand_init(seed, 0, 0, random);
}

__global__
void initSlice(double *slice, curandState *random, dim3 max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < max.z && y < max.y && x < max.x) {

        // adjust for halo padding
        z += 1;
        y += 1;
        x += 1;

        slice[indexAtWithPadding(x, y, z)] = curand_uniform(random);
    }
}

__global__
void removePadding(const double *slice, double *unpaddedSlice, dim3 max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < max.z && y < max.y && x < max.x) {
        unpaddedSlice[indexAt(x, y, z)] = slice[indexAtWithPadding(x + 1, y + 1, z + 1)];
    }
}

__global__
void getInitialHalo(double *input, dim3 max, int worldRank, int worldSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;


    if (z < max.z && y < max.y && x < max.x) {

        if (z == 0) {
            if (worldRank != 0) {
                nvshmemx_double_get_block(&input[indexAtWithPadding(1, y + 1, 0)], &input[indexAtWithPadding(1, y + 1, max.z)], max.x, worldRank - 1);
            }
        }

        if (z == max.z - 1) {
            if (worldRank != worldSize - 1) {
                nvshmemx_double_get_block(&input[indexAtWithPadding(1, y + 1, max.z + 1)], &input[indexAtWithPadding(1, y + 1, 1)], max.x, worldRank + 1);
            }
        }
    }
}

__global__
void laplace(const double *slice, double *buffer, dim3 max, int worldRank, int worldSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // compute
    double sum = -6 * slice[indexAtWithPadding(x + 1, y + 1, z + 1)];
    sum += slice[indexAtWithPadding(x, y + 1, z + 1)];
    sum += slice[indexAtWithPadding(x + 2, y + 1, z + 1)];
    sum += slice[indexAtWithPadding(x + 1, y, z + 1)];
    sum += slice[indexAtWithPadding(x + 1, y + 2, z + 1)];
    sum += slice[indexAtWithPadding(x + 1, y + 1, z)];
    sum += slice[indexAtWithPadding(x + 1, y + 1, z + 2)];

    buffer[indexAtWithPadding(x + 1, y + 1, z + 1)] += 0.01 * sum;

    // communicate
    if (z == max.z - 1 && worldRank != worldSize - 1) {
        nvshmemx_double_put_nbi_block(&buffer[indexAtWithPadding(1, y + 1, 0)], &buffer[indexAtWithPadding(1, y + 1, max.z)], max.x, worldRank + 1);
    }

    if (z == 0 && worldRank != 0) {
        nvshmemx_double_put_nbi_block(&buffer[indexAtWithPadding(1, y + 1, max.z + 1)], &buffer[indexAtWithPadding(1, y + 1, 1)], max.x, worldRank - 1);
    }
}

__global__
void sync_neighbours(int worldRank, int worldSize, uint64_t *sync, int counter) {
    nvshmem_quiet();

    // notify neighboring gpus
    if (worldRank != 0) {
        nvshmemx_signal_op(sync, counter, NVSHMEM_SIGNAL_SET, worldRank - 1);
    }

    if (worldRank != worldSize - 1) {
        nvshmemx_signal_op(sync + 1, counter, NVSHMEM_SIGNAL_SET, worldRank + 1);
    }



    // wait for own signal
    if (worldRank != 0) {
        nvshmem_uint64_wait_until(sync + 1, NVSHMEM_CMP_GE, counter);
    }

    if (worldRank != worldSize - 1) {
        nvshmem_uint64_wait_until(sync, NVSHMEM_CMP_GE, counter);
    }
}


__global__
void laplace_singleGPU(double *input, const double *buffer, dim3 max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < max.z && y < max.y && x < max.x) {

        double sum = -6 * buffer[indexAt(x, y, z)];

        if (x - 1 >= 0) {
            sum += buffer[indexAt(x - 1, y, z)];
        }

        if (x + 1 < WIDTH) {
            sum += buffer[indexAt(x + 1, y, z)];
        }

        if (y - 1 >= 0) {
            sum += buffer[indexAt(x, y - 1, z)];
        }

        if (y + 1 < HEIGHT) {
            sum += buffer[indexAt(x, y + 1, z)];
        }

        if (z - 1 >= 0) {
            sum += buffer[indexAt(x, y, z - 1)];
        }

        if (z + 1 < DEPTH) {
            sum += buffer[indexAt(x, y, z + 1)];
        }


        input[indexAt(x, y, z)] += 0.01 * sum;
    }
}

__global__
void verify(const double *a, const double *b, dim3 max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < max.z && y < max.y && x < max.x) {
        if (!doubleEquals(a[indexAt(x, y, z)], b[indexAt(x, y, z)])) {
            printf("computation mismatch at index x=%d,y=%d,z=%d: %f vs %f!\n", x, y, z, a[indexAt(x, y, z)], b[indexAt(x, y, z)]);
        }
    }
}

int main(int argc, char *argv[]) {

    bool verifyResults = false;
    if (argc > 1) {
        if (strcmp(argv[1], "1") == 0) {
            verifyResults = true;
        }
    }

    nvshmem_init();

    int worldSize = nvshmem_n_pes();
    int worldRank = nvshmem_my_pe();

    int nodeRank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_SAFE_CALL(cudaSetDevice(nodeRank));

    cudaStream_t stream;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));

    dim3 sliceSize, paddedSliceSize, grid, threads;
    sliceSize.x = WIDTH;
    sliceSize.y = HEIGHT;
    sliceSize.z = DEPTH / worldSize;

    paddedSliceSize.x = sliceSize.x + 2;
    paddedSliceSize.y = sliceSize.y + 2;
    paddedSliceSize.z = sliceSize.z + 2;

    threads.x = WIDTH;
    threads.y = 1;
    threads.z = 1;
    grid.x = std::ceil((float) sliceSize.x / threads.x);
    grid.y = std::ceil((float) sliceSize.y / threads.y);
    grid.z = std::ceil((float) sliceSize.z / threads.z);

    if (worldRank == 0) {
        printf("computation start: \n gridDim: %d, %d, %d\n threadDim: %d, %d, %d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);
    }

    size_t sliceSizeMemWithPadding = paddedSliceSize.x * paddedSliceSize.y * paddedSliceSize.z * sizeof(double);

    // device memory
    auto *slice = (double *) nvshmem_malloc(sliceSizeMemWithPadding);
    CUDA_SAFE_CALL(cudaMemset(slice, 0, sliceSizeMemWithPadding));
    auto *buffer = (double *) nvshmem_malloc(sliceSizeMemWithPadding);
    CUDA_SAFE_CALL(cudaMemset(buffer, 0, sliceSizeMemWithPadding));

    auto sync = (uint64_t *) nvshmem_malloc(2 * sizeof(uint64_t));
    CUDA_SAFE_CALL(cudaMemset(sync, 0, 2 * sizeof(uint64_t)));

    // memory needed for verification
    double *unpaddedInputSlice;
    double *unpaddedOutputSlice;
    double *aggregatedInput;
    double *aggregatedResult;
    double *computeBuffer;

    if (verifyResults) {
        size_t sliceSizeMem = sliceSize.x * sliceSize.y * sliceSize.z * sizeof(double);
        size_t totalMem = SIZE * sizeof(double);

        unpaddedInputSlice = (double *) nvshmem_malloc(sliceSizeMem);
        unpaddedOutputSlice = (double *) nvshmem_malloc(sliceSizeMem);

        aggregatedInput = (double *) nvshmem_malloc(totalMem);
        aggregatedResult = (double *) nvshmem_malloc(totalMem);
        computeBuffer = (double *) nvshmem_malloc(totalMem);
        CUDA_SAFE_CALL(cudaMemset(computeBuffer, 0, totalMem));
    }

    // create pseudo-random number generator
    curandState *random;
    CUDA_SAFE_CALL(cudaMalloc(&random, sizeof(curandState)));
    setupRandom<<<1, 1, 0, stream>>>(random, worldRank);

    // init the data on every gpu
    initSlice<<<grid, threads, 0, stream>>>(slice, random, sliceSize);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    // send all slices to one GPU for verification
    if (verifyResults) {
        // remove padding
        removePadding<<<grid, threads, 0, stream>>>(slice, unpaddedInputSlice, sliceSize);
        cudaStreamSynchronize(stream);

        // gather to GPU 0
        nvshmem_double_fcollect(NVSHMEM_TEAM_WORLD, aggregatedInput, unpaddedInputSlice, sliceSize.x * sliceSize.y * sliceSize.z);
    }

    // start measure execution time
    cudaEvent_t start, end;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));

    nvshmem_barrier_all();

    getInitialHalo<<<grid, threads, 0, stream>>>(slice, sliceSize, worldRank, worldSize);

    // computation loop
    for (int i = 0; i < ITERATIONS; i++) {

        if (i == 10) {
            CUDA_SAFE_CALL(cudaEventRecord(start, stream));
        }

        laplace<<<grid, threads, 0, stream>>>(slice, buffer, sliceSize, worldRank, worldSize);
        if (worldSize > 1) {
            sync_neighbours<<<1, 1, 0, stream>>>(worldRank, worldSize, sync, i + 1);
        }

        std::swap(slice, buffer);
    }

    // end measure execution time
    CUDA_SAFE_CALL(cudaEventRecord(end, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));

    nvshmem_barrier_all();

    float milliseconds;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, end));
    printf("execution time on rank %d: %f ms\n", worldRank, milliseconds);

    // verify results on one GPU
    if (verifyResults) {
        // remove padding
        removePadding<<<grid, threads, 0, stream>>>(slice, unpaddedOutputSlice, sliceSize);
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        // gather to GPU 0
        nvshmem_double_fcollect(NVSHMEM_TEAM_WORLD, aggregatedResult, unpaddedOutputSlice, sliceSize.x * sliceSize.y * sliceSize.z);

        // compute on single GPU and compare
        if (worldRank == 0) {
            // need new grid dimensions for total grid
            dim3 total, totalGrid;
            total.x = WIDTH;
            total.y = HEIGHT;
            total.z = DEPTH;

            totalGrid.x = std::ceil((float) total.x / threads.x);
            totalGrid.y = std::ceil((float) total.y / threads.y);
            totalGrid.z = std::ceil((float) total.z / threads.z);

            for (int i = 0; i < ITERATIONS; i++) {
                // swap buffers
                std::swap(aggregatedInput, computeBuffer);

                // laplace step
                laplace_singleGPU<<<totalGrid, threads, 0, stream>>>(aggregatedInput, computeBuffer, total);
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
            }

            // compare
            verify<<<totalGrid, threads, 0, stream>>>(aggregatedInput, aggregatedResult, total);
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

            // verification done
            printf("result correctness verified.\n");
        }
    }

    // cleanup
    nvshmem_free(slice);
    nvshmem_free(buffer);

    nvshmem_free(sync);

    if (verifyResults) {
        nvshmem_free(unpaddedInputSlice);
        nvshmem_free(unpaddedOutputSlice);
        nvshmem_free(aggregatedInput);
        nvshmem_free(aggregatedResult);
        nvshmem_free(computeBuffer);
    }

    nvshmem_finalize();
    return 0;
}