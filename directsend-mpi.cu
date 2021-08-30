#include <iostream>
#include <mpi.h>
#include <curand_kernel.h>

#define CUDA_SAFE_CALL(call) \
do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
        exit(EXIT_FAILURE); \
    } \
} while (0)                  \

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
void initRayArray(float *rayArray, curandState *random, int sliceSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sliceSize) {
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

int main(int argc, char *argv[]) {

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

    int sliceSize = IMAGE_SIZE / worldSize;

    // only needed on rank 0
    float *result;

    // device memory
    float *image1;
    float *image2;
    float *slices;
    float *compositedSlice;

    float *rayArray;

    if (worldRank == 0) {
        CUDA_SAFE_CALL(cudaMalloc(&result, IMAGE_SIZE * sizeof(float)));
    }

    CUDA_SAFE_CALL(cudaMalloc(&image1, IMAGE_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(image1, 0, IMAGE_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&image2, IMAGE_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(image2, 0, IMAGE_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&slices, IMAGE_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&compositedSlice, sliceSize * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&rayArray, IMAGE_SIZE * RAY_ARRAY_LENGTH * sizeof(float)));

    // create pseudo-random number generator
    curandState *random;
    CUDA_SAFE_CALL(cudaMalloc(&random, sizeof(curandState)));
    setupRandom<<<1, 1, 0, stream>>>(random, worldRank);

    initRayArray<<<std::ceil((float) sliceSize / 1024), 1024, 0, stream>>>(rayArray, random, sliceSize);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    cudaEvent_t start, end;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));

    MPI_Barrier(MPI_COMM_WORLD);

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
        MPI_Alltoall(image1, sliceSize, MPI_FLOAT, slices, sliceSize, MPI_FLOAT, MPI_COMM_WORLD);

        // composite the slices of image 1
        composite<<<sliceGrid, threads, 0, stream>>>(compositedSlice, slices, worldSize, sliceWidth);
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        // gather the composited slices of image 1 to rank 0
        MPI_Gather(compositedSlice, sliceSize, MPI_FLOAT, result, sliceSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // swap buffers
        std::swap(image1, image2);
    }

    // distribute the last image
    MPI_Alltoall(image1, sliceSize, MPI_FLOAT, slices, sliceSize, MPI_FLOAT, MPI_COMM_WORLD);

    composite<<<sliceGrid, threads, 0, stream>>>(compositedSlice, slices, worldSize, sliceWidth);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    MPI_Gather(compositedSlice, sliceSize, MPI_FLOAT, result, sliceSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // done.

    CUDA_SAFE_CALL(cudaEventRecord(end, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));

    MPI_Barrier(MPI_COMM_WORLD);

    float executionTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&executionTime, start, end));

    float maxExecutionTime = 0;
    MPI_Reduce(&executionTime, &maxExecutionTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("execution done.\n max execution time: %f ms\n", maxExecutionTime);
    }

    if (worldRank == 0) {
        CUDA_SAFE_CALL(cudaFree(result));
    }

    CUDA_SAFE_CALL(cudaFree(image1));
    CUDA_SAFE_CALL(cudaFree(image2));
    CUDA_SAFE_CALL(cudaFree(slices));
    CUDA_SAFE_CALL(cudaFree(compositedSlice));
    CUDA_SAFE_CALL(cudaFree(rayArray));

    MPI_Finalize();
    return 0;
}