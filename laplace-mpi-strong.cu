#include <cmath>
#include <iostream>
#include <mpi.h>
#include <curand_kernel.h>
#include <chrono>

#define CUDA_SAFE_CALL(call) \
do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        fprintf (stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

const long WIDTH = 512;
const long HEIGHT = 512;
const long DEPTH = 512;
const long SIZE = WIDTH * HEIGHT * DEPTH;

const int ITERATIONS = 110;

__device__
long indexAt(int x, int y, int z) {
    return WIDTH * HEIGHT * z + WIDTH * y + x;
}

__device__
long indexAtWithPadding(int x, int y, int z) {
    return (WIDTH + 2) * (HEIGHT + 2) * z + (WIDTH + 2) * y + x;
}

__device__
bool doubleEquals(double a, double b) {
    return fabs(a - b) < 1e-8;
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
void laplaceInnerKernel(const double *slice, double *resultSlice, dim3 max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // adjust for halo padding and inner computation
    z += 2;
    y += 1;
    x += 1;

    double sum = -6 * slice[indexAtWithPadding(x, y, z)];
    sum += slice[indexAtWithPadding(x - 1, y, z)];
    sum += slice[indexAtWithPadding(x + 1, y, z)];
    sum += slice[indexAtWithPadding(x, y - 1, z)];
    sum += slice[indexAtWithPadding(x, y + 1, z)];
    sum += slice[indexAtWithPadding(x, y, z - 1)];
    sum += slice[indexAtWithPadding(x, y, z + 1)];

    resultSlice[indexAtWithPadding(x, y, z)] += 0.01 * sum;
}

__global__
void laplaceOuterKernel(const double *slice, double *resultSlice, dim3 max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // move to z=0 and z=max.z-1
    if (z == 1) {
        z = max.z - 1;
    }

    // adjust for halo padding
    z += 1;
    y += 1;
    x += 1;

    double sum = -6 * slice[indexAtWithPadding(x, y, z)];
    sum += slice[indexAtWithPadding(x - 1, y, z)];
    sum += slice[indexAtWithPadding(x + 1, y, z)];
    sum += slice[indexAtWithPadding(x, y - 1, z)];
    sum += slice[indexAtWithPadding(x, y + 1, z)];
    sum += slice[indexAtWithPadding(x, y, z - 1)];
    sum += slice[indexAtWithPadding(x, y, z + 1)];

    resultSlice[indexAtWithPadding(x, y, z)] += 0.01 * sum;
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

    MPI_Init(nullptr, nullptr);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    MPI_Comm nodeCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeCommunicator);
    int nodeRank;
    MPI_Comm_rank(nodeCommunicator, &nodeRank);

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

    // inner and outer kernel grids
    dim3 innerGrid, outerGrid;
    innerGrid.x = std::ceil((float) sliceSize.x / threads.x);
    innerGrid.y = std::ceil((float) sliceSize.y / threads.y);
    innerGrid.z = std::ceil((float) (sliceSize.z - 2) / threads.z);

    outerGrid.x = std::ceil((float) sliceSize.x / threads.x);
    outerGrid.y = std::ceil((float) sliceSize.y / threads.y);
    outerGrid.z = std::ceil((float) 2 / threads.z);

    if (worldRank == 0) {
        printf("computation start: \n gridDim: %d, %d, %d\n threadDim: %d, %d, %d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);
    }

    // memory needed every time
    double *slice;
    double *resultSlice;

    size_t sliceSizeMemWithPadding = paddedSliceSize.x * paddedSliceSize.y * paddedSliceSize.z * sizeof(double);

    CUDA_SAFE_CALL(cudaMalloc(&slice, sliceSizeMemWithPadding));
    CUDA_SAFE_CALL(cudaMemset(slice, 0, sliceSizeMemWithPadding));
    CUDA_SAFE_CALL(cudaMalloc(&resultSlice, sliceSizeMemWithPadding));
    CUDA_SAFE_CALL(cudaMemset(resultSlice, 0, sliceSizeMemWithPadding));

    // memory needed for verification
    double *unpaddedInputSlice;
    double *unpaddedOutputSlice;
    double *aggregatedInput;
    double *aggregatedResult;
    double *computeBuffer;

    if (verifyResults) {
        size_t sliceSizeMem = sliceSize.x * sliceSize.y * sliceSize.z * sizeof(double);

        CUDA_SAFE_CALL(cudaMalloc(&unpaddedInputSlice, sliceSizeMem));
        CUDA_SAFE_CALL(cudaMalloc(&unpaddedOutputSlice, sliceSizeMem));

        if (worldRank == 0) {
            size_t totalMem = SIZE * sizeof(double);

            CUDA_SAFE_CALL(cudaMalloc(&aggregatedInput, totalMem));
            CUDA_SAFE_CALL(cudaMalloc(&aggregatedResult, totalMem));
            CUDA_SAFE_CALL(cudaMalloc(&computeBuffer, totalMem));
            CUDA_SAFE_CALL(cudaMemset(computeBuffer, 0, totalMem));
        }
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
        MPI_Gather(unpaddedInputSlice, sliceSize.x * sliceSize.y * sliceSize.z, MPI_DOUBLE, aggregatedInput, sliceSize.x * sliceSize.y * sliceSize.z, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // start measure execution time
    cudaEvent_t start, end;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));

    MPI_Barrier(MPI_COMM_WORLD);
//    CUDA_SAFE_CALL(cudaEventRecord(start, stream));

    for (int i = 0; i < ITERATIONS; i++) {

        if (i == 10) {
            CUDA_SAFE_CALL(cudaEventRecord(start, stream));
        }

        // execute the kernel
        laplaceInnerKernel<<<innerGrid, threads, 0, stream>>>(slice, resultSlice, sliceSize);

        int count = 0;
        MPI_Request requests[4];

        if (worldRank != worldSize - 1) {
            MPI_Isend(&slice[paddedSliceSize.x * paddedSliceSize.y * (paddedSliceSize.z - 2)], paddedSliceSize.x * paddedSliceSize.y, MPI_DOUBLE, worldRank + 1, 0, MPI_COMM_WORLD, &requests[count]);
            count++;
        }

        if (worldRank != 0) {
            MPI_Irecv(slice, paddedSliceSize.x * paddedSliceSize.y, MPI_DOUBLE, worldRank - 1, 0, MPI_COMM_WORLD, &requests[count]);
            count++;
        }

        if (worldRank != 0) {
            MPI_Isend(&slice[paddedSliceSize.x * paddedSliceSize.y], paddedSliceSize.x * paddedSliceSize.y, MPI_DOUBLE, worldRank - 1, 1, MPI_COMM_WORLD, &requests[count]);
            count++;
        }

        if (worldRank != worldSize - 1) {
            MPI_Irecv(&slice[paddedSliceSize.x * paddedSliceSize.y * (paddedSliceSize.z - 1)], paddedSliceSize.x * paddedSliceSize.y, MPI_DOUBLE, worldRank + 1, 1, MPI_COMM_WORLD, &requests[count]);
            count++;
        }

        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

        // execute the kernel
        laplaceOuterKernel<<<outerGrid, threads, 0, stream>>>(slice, resultSlice, sliceSize);
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        // swap buffers
        std::swap(slice, resultSlice);
    }

    // end measure execution time
    CUDA_SAFE_CALL(cudaEventRecord(end, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));

    MPI_Barrier(MPI_COMM_WORLD);

    float milliseconds;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, end));

    float maxExecutionTime = 0;
    MPI_Reduce(&milliseconds, &maxExecutionTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    // print execution time
    if (worldRank == 0) {
        printf("computation done on rank %d.\n max execution time: %f ms\n", worldRank, maxExecutionTime);
    }

    // verify results on one GPU
    if (verifyResults) {
        // remove padding
        removePadding<<<grid, threads, 0, stream>>>(slice, unpaddedOutputSlice, sliceSize);
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        // gather to GPU 0
        MPI_Gather(unpaddedOutputSlice, sliceSize.x * sliceSize.y * sliceSize.z, MPI_DOUBLE, aggregatedResult, sliceSize.x * sliceSize.y * sliceSize.z, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    CUDA_SAFE_CALL(cudaFree(slice));
    CUDA_SAFE_CALL(cudaFree(resultSlice));

    if (verifyResults) {
        CUDA_SAFE_CALL(cudaFree(unpaddedInputSlice));
        CUDA_SAFE_CALL(cudaFree(unpaddedOutputSlice));

        if (worldRank == 0) {
            CUDA_SAFE_CALL(cudaFree(aggregatedInput));
            CUDA_SAFE_CALL(cudaFree(aggregatedResult));
            CUDA_SAFE_CALL(cudaFree(computeBuffer));
        }
    }

    MPI_Finalize();
    return 0;
}