#include <assert.h>
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

#define cudaAssert(code, file, line) do { \
  if (code != cudaSuccess) { \
    fprintf(stderr, "A CUDA error occurred: %s (%s:%d)\n", \
            cudaGetErrorString(code), file, line); \
    exit(code); \
  } \
} while(0)

#define cudaCheck(code) do { cudaAssert(code, __FILE__, __LINE__); } while(0)

/* CUDA kernels ***************************************************************/

// reassign points to closest centroids (#threads must be a power of two)
__global__
static void reassign(struct pixel *pixels, size_t n_pixels,
                     struct pixel *centroids, size_t n_centroids,
                     size_t *labels, struct pixel *sums, size_t *counts,
                     int *empty, int *done)
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

    // begin reassignment
    size_t index = bid * blockDim.x + tid;
    size_t closest_centroid = 0u;

    if (index >= n_pixels) {
       closest_centroid = n_centroids + 1u;
    } else {
        // find closest centroid
        double min_dist = DBL_MAX;

        struct pixel *p = &pixels[index];
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
        if (closest_centroid != labels[index]) {
            labels[index] = closest_centroid;

            *done = 0;
        }
    }

    // perform cluster wise tree-reduction to obtain cluster sums / counts
    for (size_t j = 0u; j < n_centroids; ++j) {
        if (j == closest_centroid) {
            shared_pixels[tid] = pixels[index];
            shared_counts[tid] = 1u;
            empty[j] = 0u;
        } else {
            struct pixel tmp = { 0.0, 0.0, 0.0 };
            shared_pixels[tid] = tmp;
            shared_counts[tid] = 0u;
        }

        __syncthreads();

        for (size_t dist = blockDim.x >> 1; dist > 0u; dist >>= 1) {
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

// reduce per-block cluster sums and counts
__device__
static void _reduce(size_t n_blocks, size_t n_centroids,
                    struct pixel *sums, size_t *counts)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // reduce per-block clusters sums / counts
    for (size_t dist = (n_blocks * n_centroids) >> 1;
         dist >= n_centroids; dist >>= 1) {

        for (size_t i = index; i < dist; i += stride) {
            struct pixel *sum1 = &sums[i];
            struct pixel *sum2 = &sums[i + dist];

            sum1->r += sum2->r;
            sum1->g += sum2->g;
            sum1->b += sum2->b;

            counts[index] += counts[i + dist];
       }

       __syncthreads();
    }
}

__global__
static void reduce(size_t n_blocks, size_t n_centroids,
                   struct pixel *sums, size_t *counts)
{
    _reduce(n_blocks, n_centroids, sums, counts);
}

// re-calculate centroids
__global__
static void average(size_t n_blocks, struct pixel *centroids, size_t n_centroids,
                    struct pixel *sums, size_t *counts, int reduce)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if (reduce)
        _reduce(n_blocks, n_centroids, sums, counts);

    // compute new centroids
    for (size_t i = index; i < n_centroids; i += stride) {
        struct pixel *c = &centroids[i];
        struct pixel *sum = &sums[i];
        size_t count = counts[i];

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
        (n_pixels + KMEANS_CUDA_BLOCKSIZE - 1u) / KMEANS_CUDA_BLOCKSIZE;

    // round upwards to power of two
    size_t n_blocks_reassign_log2 = 0u;
    while(n_blocks_reassign >>= 1)
        ++n_blocks_reassign_log2;

    n_blocks_reassign = 1 << (n_blocks_reassign_log2 + 1u);

    size_t n_block_reduce =
        (((n_centroids * n_blocks_reassign) >> 1) + KMEANS_CUDA_BLOCKSIZE - 1u) /
        KMEANS_CUDA_BLOCKSIZE;

    // reassignment step shared memory size
    size_t shm_slots;
    if (n_centroids > KMEANS_CUDA_BLOCKSIZE)
        shm_slots = n_centroids;
    else
        shm_slots = KMEANS_CUDA_BLOCKSIZE;

    size_t shm_reassign = shm_slots * (sizeof(struct pixel) + sizeof(size_t));
    shm_reassign += sizeof(size_t) - sizeof(struct pixel) % sizeof(size_t);

    // initialize centroids with random pixels
    srand(time(NULL));

    for (size_t i = 0u; i < n_centroids; ++i)
        centroids[i] = pixels[rand() % n_pixels];

    // initialize device memory
    size_t pixels_sz = n_pixels * sizeof(struct pixel);
    size_t centroids_sz = n_centroids * sizeof(struct pixel);
    size_t labels_sz = n_pixels * sizeof(size_t);

    struct pixel *pixels_dev;
    struct pixel *centroids_dev;
    size_t *labels_dev;

    cudaCheck(cudaMalloc(&pixels_dev, pixels_sz));
    cudaCheck(cudaMalloc(&centroids_dev, centroids_sz));
    cudaCheck(cudaMalloc(&labels_dev, labels_sz));

    cudaCheck(cudaMemcpy(pixels_dev, pixels, pixels_sz,
                         cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(centroids_dev, centroids, centroids_sz,
                         cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(labels_dev, labels, labels_sz,
                         cudaMemcpyHostToDevice));

    // allocate and initialize auxiliary memory
    size_t sums_sz = n_centroids * sizeof(struct pixel);
    size_t counts_sz = n_centroids * sizeof(size_t);
    size_t empty_sz = n_centroids * sizeof(int);
    size_t sums_dev_sz = n_blocks_reassign * n_centroids * sizeof(struct pixel);
    size_t counts_dev_sz = n_blocks_reassign * n_centroids * sizeof(size_t);

    struct pixel *sums, *sums_dev;
    size_t *counts, *counts_dev;
    int *empty, *empty_dev;
    int done, *done_dev;

    sums = (struct pixel *) malloc(sums_sz);
    counts = (size_t *) malloc(counts_sz);
    empty = (int *) malloc(empty_sz);

    cudaCheck(cudaMalloc(&sums_dev, sums_dev_sz));
    cudaCheck(cudaMalloc(&counts_dev, counts_dev_sz));
    cudaCheck(cudaMalloc(&empty_dev, empty_sz));
    cudaCheck(cudaMalloc(&done_dev, sizeof(int)));

    for (size_t i = 0u; i < n_centroids; ++i) {
        struct pixel tmp = { 0.0, 0.0, 0.0 };
        sums[i] = tmp;
        counts[i] = 0u;
    }

    // repeat for KMEANS_MAX_ITER or until solution is stationary
    for (int iter = 0; iter < KMEANS_MAX_ITER; ++iter) {
        for (size_t i = 0u; i < n_centroids; ++i)
            empty[i] = 1;

        done = 1;

        cudaCheck(cudaMemcpy(empty_dev, empty, empty_sz,
                             cudaMemcpyHostToDevice));

        cudaCheck(cudaMemcpy(done_dev, &done, sizeof(int),
                             cudaMemcpyHostToDevice));

        // reassign points to closest centroids
        reassign<<<n_blocks_reassign, KMEANS_CUDA_BLOCKSIZE, shm_reassign>>>(
            pixels_dev, n_pixels, centroids_dev, n_centroids, labels_dev,
            sums_dev, counts_dev, empty_dev, done_dev
        );

        cudaCheck(cudaPeekAtLastError());

        cudaCheck(cudaMemcpy(empty, empty_dev, empty_sz,
                             cudaMemcpyDeviceToHost));

        cudaCheck(cudaMemcpy(&done, done_dev, sizeof(int),
                             cudaMemcpyDeviceToHost));

        // check whether empty clusters need to be repaired
        int repair = 0;
        for (size_t i = 0u; i < n_centroids; ++i) {
            if (!empty[i])
                continue;

            // reduce in separate kernel
            reduce<<<n_block_reduce, KMEANS_CUDA_BLOCKSIZE>>>(
                n_blocks_reassign, n_centroids, sums_dev, counts_dev
            );

            cudaCheck(cudaMemcpy(sums, sums_dev, sums_sz,
                                 cudaMemcpyDeviceToHost));

            cudaCheck(cudaMemcpy(counts, counts_dev, counts_sz,
                                 cudaMemcpyDeviceToHost));

            done = 0;
            repair = 1;
            break;
        }

        // repair empty clusters (on host)
        if (repair) {
            for (size_t i = 0u; i < n_centroids; ++i) {
                if (!empty[i])
                    continue;

                done = 0;

                // determine largest cluster
                size_t largest_cluster = 0u;
                size_t largest_cluster_count = 0u;
                for (size_t j = 0u; j < n_centroids; ++j) {
                    if (j == i)
                        continue;

                    if (counts[j] > largest_cluster_count) {
                        largest_cluster = j;
                        largest_cluster_count = counts[j];
                    }
                }

                // determine pixel in this cluster furthest from its centroid
                struct pixel *largest_cluster_centroid =
                    &centroids[largest_cluster];

                size_t furthest_pixel = 0u;
                double max_dist = 0.0;
                for (size_t j = 0u; j < n_pixels; ++j) {
                    if (labels[j] != largest_cluster)
                        continue;

                    struct pixel *p = &pixels[j];

                    double dr = p->r - largest_cluster_centroid->r;
                    double dg = p->g - largest_cluster_centroid->g;
                    double db = p->b - largest_cluster_centroid->b;

                    double dist = sqrt(dr * dr + dg * dg + db * db);

                    if (dist > max_dist) {
                        furthest_pixel = j;
                        max_dist = dist;
                    }
                }

                // move that pixel to the empty cluster
                struct pixel replacement_pixel = pixels[furthest_pixel];
                centroids[i] = replacement_pixel;
                labels[furthest_pixel] = i;

                // correct cluster sums
                sums[i] = replacement_pixel;

                struct pixel *sum = &sums[largest_cluster];
                sum->r -= replacement_pixel.r;
                sum->g -= replacement_pixel.g;
                sum->b -= replacement_pixel.b;

                // correct cluster sizes
                counts[i] = 1u;
                counts[largest_cluster]--;
            }

            cudaCheck(cudaMemcpy(sums_dev, sums, sums_sz,
                                 cudaMemcpyHostToDevice));

            cudaCheck(cudaMemcpy(counts_dev, counts, counts_sz,
                                 cudaMemcpyHostToDevice));
        }

        // re-calculate centroids
        average<<<n_block_reduce, KMEANS_CUDA_BLOCKSIZE>>>(
            n_blocks_reassign, centroids_dev, n_centroids,
            sums_dev, counts_dev, !repair
        );

        cudaCheck(cudaPeekAtLastError());
        cudaCheck(cudaDeviceSynchronize());

        // break if no pixel has changed cluster
        if (done)
            break;
    }

    // copy device memory back to host
    cudaCheck(cudaMemcpy(pixels, pixels_dev, pixels_sz,
                         cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(centroids, centroids_dev, centroids_sz,
                         cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(labels, labels_dev, labels_sz,
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
