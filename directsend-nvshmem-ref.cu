#include <iostream>
#include <nvshmem.h>
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

const int IMAGE_WIDTH = 960;
const int IMAGE_HEIGHT = 960;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

const int RAY_ARRAY_LENGTH = 2048;
const int ITERATIONS = 110;

__device__
int indexAt(int x, int y) {
    return x * IMAGE_HEIGHT + y;
}

__global__
void setupRandom(curandState *random, int seed) {
    curand_init(seed, 0, 0, random);
}

__global__
void initRayArray(float *rayArray, curandState* random) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < IMAGE_SIZE) {
        for (int i = 0; i < RAY_ARRAY_LENGTH; i++) {
            rayArray[index * RAY_ARRAY_LENGTH + i] = curand_uniform(random);
        }
    }
}

__global__
void computeImage(float *image, const float *rayArray, int worldRank, int sliceWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x / sliceWidth == worldRank) {
        float accumulated = 0;
        for (int i = 0; i < RAY_ARRAY_LENGTH; i++) {
            accumulated += rayArray[indexAt(x,y) * RAY_ARRAY_LENGTH + i];
        }

        image[indexAt(x,y)] = accumulated;
    } else {
        image[indexAt(x,y)] = 0;
    }
}

__global__
void composite(float *compositedSlice, const float *slices, int worldSize, int sliceWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for (int i = 0; i < worldSize; i++) {
        sum += slices[indexAt(i * sliceWidth + x, y)];
    }

    compositedSlice[indexAt(x,y)] = sum;
}

__global__
void printTimings(const float *maxExecutionTime) {
    printf("execution done on rank 0.\n max execution time: %f ms\n", *maxExecutionTime);
}

int main(int argc, char *argv[]) {

    nvshmem_init();

    int worldSize = nvshmem_n_pes();
    int worldRank = nvshmem_my_pe();

    int nodeRank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_SAFE_CALL(cudaSetDevice(nodeRank));

    cudaStream_t stream;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));

    int sliceSize = IMAGE_SIZE / worldSize;

    auto *result = (float *) nvshmem_malloc(IMAGE_SIZE * sizeof(float));

    auto *image1 = (float *) nvshmem_malloc(IMAGE_SIZE * sizeof(float));
    CUDA_SAFE_CALL(cudaMemset(image1, 0, IMAGE_SIZE * sizeof(float)));
    auto *image2 = (float *) nvshmem_malloc(IMAGE_SIZE * sizeof(float));
    CUDA_SAFE_CALL(cudaMemset(image2, 0, IMAGE_SIZE * sizeof(float)));

    auto *slices = (float *) nvshmem_malloc(IMAGE_SIZE * sizeof(float));
    auto *compositedSlice = (float *) nvshmem_malloc(sliceSize * sizeof(float));

    float *rayArray;
    CUDA_SAFE_CALL(cudaMalloc(&rayArray, IMAGE_SIZE * RAY_ARRAY_LENGTH * sizeof(float)));

    // create pseudo-random number generator
    curandState *random;
    CUDA_SAFE_CALL(cudaMalloc(&random, sizeof(curandState)));
    setupRandom<<<1, 1, 0, stream>>>(random, worldRank);

    initRayArray<<<std::ceil((float) IMAGE_SIZE / 1024), 1024, 0, stream>>>(rayArray, random);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    cudaEvent_t start, end;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));

    nvshmem_barrier_all();

    int sliceWidth = IMAGE_WIDTH / worldSize;

    dim3 threads(1, IMAGE_HEIGHT, 1);
    dim3 imageGrid(IMAGE_WIDTH, 1, 1);
    dim3 sliceGrid(sliceWidth, 1, 1);

    // compute image 1
    computeImage<<<imageGrid, threads, 0, stream>>>(image1, rayArray, worldRank, sliceWidth);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < ITERATIONS; i++) {

        if (i == 10) {
            CUDA_SAFE_CALL(cudaEventRecord(start, stream));
        }

        // compute image 2
        computeImage<<<imageGrid, threads, 0, stream>>>(image2, rayArray, worldRank, sliceWidth);

        // while image 2 is computing, distribute image 1 parts to the respective ranks
        nvshmem_float_alltoall(NVSHMEM_TEAM_WORLD, slices, image1, sliceSize);

        // composite the slices
        composite<<<sliceGrid, threads, 0, stream>>>(compositedSlice, slices, worldSize, sliceWidth);
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        // gather the composited slices of image 1 to rank 0
        nvshmem_float_put(&result[worldRank * sliceSize], compositedSlice, sliceSize, 0);

        // swap buffers
        std::swap(image1, image2);
    }

    // distribute the last image
    nvshmem_float_alltoall(NVSHMEM_TEAM_WORLD, slices, image1, sliceSize);

    composite<<<sliceGrid, threads, 0, stream>>>(compositedSlice, slices, worldSize, sliceWidth);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    nvshmem_float_put(&result[worldRank * sliceSize], compositedSlice, sliceSize, 0);

    // done.

    CUDA_SAFE_CALL(cudaEventRecord(end, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));

    nvshmem_barrier_all();

    float executionTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&executionTime, start, end));

    auto *maxExecutionTime = (float *) nvshmem_malloc(sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(maxExecutionTime, &executionTime, sizeof(float), cudaMemcpyHostToDevice));
    nvshmemx_float_max_reduce_on_stream(NVSHMEM_TEAM_WORLD, maxExecutionTime, maxExecutionTime, 1, stream);

    if (worldRank == 0) {
        printTimings<<<1, 1, 0, stream>>>(maxExecutionTime);
    }
    cudaStreamSynchronize(stream);

    nvshmem_free(result);
    nvshmem_free(image1);
    nvshmem_free(image2);
    nvshmem_free(slices);
    nvshmem_free(compositedSlice);

    CUDA_SAFE_CALL(cudaFree(rayArray));

    nvshmem_finalize();
    return 0;
}