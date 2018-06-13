#include <cassert>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "kmeans.h"
}
#include "kmeans_config.h"

/* Helper Functions ***********************************************************/

#define cudaAssert(code, file, line) do {                  \
  if (code != cudaSuccess) {                               \
    fprintf(stderr, "A CUDA error occurred: %s (%s:%d)\n", \
            cudaGetErrorString(code), file, line);         \
    exit(code);                                            \
  }                                                        \
} while(0)

#define cudaCheck(code) do { cudaAssert(code, __FILE__, __LINE__); } while(0)

/* CUDA kernels ***************************************************************/

// reassign points to closest centroids
__global__
static void reassign(struct pixel *pixels, size_t n_pixels,
                     struct pixel *centroids, size_t n_centroids,
                     size_t *labels, struct pixel *sums, size_t *counts,
                     int *done)
{
    // index alias
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;

    // set up shared and global memory
    extern __shared__ char shared[];

    struct pixel *shared_pixels = (struct pixel *) shared;

    size_t shared_counts_offs = blockDim.x * sizeof(struct pixel);
    shared_counts_offs += sizeof(size_t) - sizeof(struct pixel) % sizeof(size_t);

    size_t *shared_counts = (size_t *) &shared[shared_counts_offs];

    if (tid < n_centroids) {
      shared_pixels[tid] = centroids[tid];

      struct pixel tmp = { 0.0, 0.0, 0.0 };
      sums[n_centroids * bid + tid] = tmp;

      counts[n_centroids * bid + tid] = 0u;
    }

    __syncthreads();

    // obtain pixel index and stride (less threads than total pixels available)
    size_t index = bid * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n_pixels; i += stride) {
        struct pixel *p = &pixels[i];

        // find centroid closest to pixel
        size_t closest_centroid = 0u;
        double min_dist = DBL_MAX;

        for (size_t j = 0u; j < n_centroids; ++j) {
            struct pixel *c = &shared_pixels[j];

            double dr = p->r - c->r;
            double dg = p->g - c->g;
            double db = p->b - c->b;

            double dist = sqrt(dr * dr + dg * dg + db * db);

            if (dist < min_dist) {
                closest_centroid = j;
                min_dist = dist;
            }
        }

        // if pixel has changed cluster...
        if (closest_centroid != labels[i]) {
            labels[i] = closest_centroid;

            *done = 0;
        }

        // perform cluster wise tree-reduction to obtain cluster sums / counts
        for (size_t j = 0u; j < n_centroids; ++j) {
            if (j == closest_centroid) {
                shared_pixels[tid] = *p;
                shared_counts[tid] = 1u;
            } else {
                struct pixel tmp = { 0.0, 0.0, 0.0 };
                shared_pixels[tid] = tmp;
                shared_counts[tid] = 0u;
            }

            __syncthreads();

            for (size_t dist = blockDim.x >> 1; dist > 0; dist >>= 1) {
                if (tid < dist) {
                    struct pixel *shared_sum1 = &shared_pixels[tid];
                    struct pixel *shared_sum2 = &shared_pixels[tid + dist];

                    shared_sum1->r += shared_sum2->r;
                    shared_sum1->g += shared_sum2->g;
                    shared_sum1->b += shared_sum2->b;

                    shared_counts[tid] += shared_counts[tid + dist];
                }

                __syncthreads();
            }

            if (tid == 0) {
                struct pixel *sum = &sums[n_centroids * bid + j];
                struct pixel *shared_sum = &shared_pixels[0];

                sum->r += shared_sum->r;
                sum->g += shared_sum->g;
                sum->b += shared_sum->b;

                counts[n_centroids * bid + j] += shared_counts[0];
            }
        }
    }
}

// reduce per-block cluster sums and counts
__global__
static void average(size_t n_blocks, struct pixel *centroids, size_t n_centroids,
                    struct pixel *sums, size_t *counts)
{
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;

    size_t index = bid * blockDim.x + tid;
    size_t stride = blockDim.x * gridDim.x;

    // reduce per-block clusters sums / counts
    for (size_t dist = (n_blocks * n_centroids) >> 1; dist >= n_centroids; dist >>= 1) {
        for (size_t i = index; i < dist; i += stride) {
            struct pixel *sum1 = &sums[i];
            struct pixel *sum2 = &sums[i + dist];

            sum1->r += sum2->r;
            sum1->g += sum2->g;
            sum1->b += sum2->b;

            counts[i] += counts[i + dist];
        }
        __syncthreads();
    }

    // compute new centroids
    for (size_t j = index; j < n_centroids; j += stride) {
        struct pixel *c = &centroids[j];
        struct pixel *sum = &sums[j];
        size_t count = counts[j];

        c->r = sum->r / count;
        c->g = sum->g / count;
        c->b = sum->b / count;
    }
}

/* Main Function **************************************************************/

extern "C" void kmeans_cuda(struct pixel *pixels, size_t n_pixels,
                            struct pixel *centroids, size_t n_centroids,
                            size_t *labels)
{
    // number of blocks to be used on device
    size_t n_blocks_reassign =
        (n_pixels + KMEANS_CUDA_BLOCKSIZE - 1) / KMEANS_CUDA_BLOCKSIZE;

    size_t n_blocks_average =
        (((n_centroids * n_blocks_reassign) >> 1) + KMEANS_CUDA_BLOCKSIZE - 1) /
        KMEANS_CUDA_BLOCKSIZE;

    // reassignment step shared memory size
    size_t shm_slots =
        n_centroids > KMEANS_CUDA_BLOCKSIZE ? n_centroids : KMEANS_CUDA_BLOCKSIZE;

    size_t shm_reassign = shm_slots * (sizeof(struct pixel) + sizeof(size_t));
    shm_reassign += sizeof(size_t) - sizeof(struct pixel) % sizeof(size_t);

    // initialize centroids with random pixels
    srand(time(NULL));

    for (size_t i = 0u; i < n_centroids; ++i)
        centroids[i] = pixels[rand() % n_pixels];

    // initialize device memory
    struct pixel *pixels_dev;
    struct pixel *centroids_dev;
    size_t *labels_dev;

    cudaCheck(cudaMalloc(&pixels_dev, n_pixels * sizeof(struct pixel)));
    cudaCheck(cudaMalloc(&centroids_dev, n_centroids * sizeof(struct pixel)));
    cudaCheck(cudaMalloc(&labels_dev, n_pixels * sizeof(size_t)));

    cudaCheck(cudaMemcpy(pixels_dev, pixels, n_pixels * sizeof(struct pixel),
                         cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(centroids_dev, centroids, n_centroids * sizeof(struct pixel),
                         cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(labels_dev, labels, n_pixels * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    // allocate and initialize auxiliary memory
    struct pixel *sums, *sums_dev;
    size_t *counts, *counts_dev;
    int done, *done_dev;

    sums = (struct pixel *) malloc(n_centroids * sizeof(struct pixel));
    counts = (size_t *) malloc(n_centroids *  sizeof(size_t));

    cudaCheck(cudaMalloc(&sums_dev, n_blocks_reassign * n_centroids * sizeof(struct pixel)));
    cudaCheck(cudaMalloc(&counts_dev, n_blocks_reassign * n_centroids *  sizeof(size_t)));
    cudaCheck(cudaMalloc(&done_dev, sizeof(int)));

    for (size_t i = 0u; i < n_centroids; ++i) {
        struct pixel tmp = { 0.0, 0.0, 0.0 };
        sums[i] = tmp;

        counts[i] = 0u;
    }

    // repeat for KMEANS_MAX_ITER or until solution is stationary
    for (int iter = 0; iter < KMEANS_MAX_ITER; ++iter) {
        done = 1;

        cudaCheck(cudaMemcpy(done_dev, &done, sizeof(int),
                             cudaMemcpyHostToDevice));

        // reassign points to closest centroids
        reassign<<<n_blocks_reassign, KMEANS_CUDA_BLOCKSIZE, shm_reassign>>>(
            pixels_dev, n_pixels, centroids_dev, n_centroids, labels_dev,
            sums_dev, counts_dev, done_dev
        );

        cudaCheck(cudaPeekAtLastError());
        cudaCheck(cudaDeviceSynchronize());

        average<<<n_blocks_average, KMEANS_CUDA_BLOCKSIZE>>>(
            n_blocks_reassign, centroids_dev, n_centroids, sums_dev, counts_dev
        );

        cudaCheck(cudaPeekAtLastError());
        cudaCheck(cudaMemcpy(&done, done_dev, sizeof(int),
                             cudaMemcpyDeviceToHost));

        // break if no pixel has changed cluster
        if (done)
            break;
    }

    // copy device memory back to host
    cudaCheck(cudaMemcpy(pixels, pixels_dev, n_pixels * sizeof(struct pixel),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(centroids, centroids_dev, n_centroids * sizeof(struct pixel),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(labels, labels_dev, n_pixels * sizeof(size_t),
                         cudaMemcpyDeviceToHost));

    // free host and device memory
    free(sums);
    free(counts);

    cudaCheck(cudaFree(pixels_dev));
    cudaCheck(cudaFree(centroids_dev));
    cudaCheck(cudaFree(labels_dev));
    cudaCheck(cudaFree(sums_dev));
    cudaCheck(cudaFree(counts_dev));
}
