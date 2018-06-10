#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "kmeans.h"
}
#include "kmeans_config.h"

// reassign points to closest centroids
__global__
static void reassign_points(struct pixel *pixels, size_t n_pixels,
                            struct pixel *centroids, size_t n_centroids,
                            size_t *labels, struct pixel *sums, size_t *counts,
                            int *done)
{
  int index = blockIdx.x * blockDim.x * threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n_pixels; i += stride) {
      struct pixel pixel = pixels[i];

      // find centroid closest to pixel
      size_t closest_centroid = 0u;
      double min_dist = DBL_MAX;

      for (size_t j = 0; j < n_centroids; ++j) {
          struct pixel centroid = centroids[j];

          double dr = pixel.r - centroid.r;
          double dg = pixel.g - centroid.g;
          double db = pixel.b - centroid.b;

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

      // update cluster sum
      struct pixel *sum = &sums[closest_centroid];
      sum->r += pixel.r;
      sum->g += pixel.g;
      sum->b += pixel.b;

      // update cluster size
      counts[closest_centroid]++;
  }
}

// compute euclidean distance between two pixel values
static inline double pixel_dist(struct pixel p1, struct pixel p2)
{
    double dr = p1.r - p2.r;
    double dg = p1.g - p2.g;
    double db = p1.b - p2.b;

    return sqrt(dr * dr + dg * dg + db * db);
}

extern "C" void kmeans_cuda(struct pixel *pixels, size_t n_pixels,
                            struct pixel *centroids, size_t n_centroids,
                            size_t *labels)
{
    // number of blocks to be used on device
    int blocks = (n_pixels + KMEANS_CUDA_BLOCKSIZE - 1) / KMEANS_CUDA_BLOCKSIZE;

    // initialize device memory
    struct pixel *pixels_dev;
    struct pixel *centroids_dev;
    size_t *labels_dev;

    cudaMalloc(&pixels_dev, n_pixels * sizeof(struct pixel));
    cudaMalloc(&centroids_dev, n_centroids * sizeof(struct pixel));
    cudaMalloc(&labels_dev, n_pixels * sizeof(size_t));

    cudaMemcpy(pixels_dev, pixels, n_pixels * sizeof(struct pixel),
               cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_dev, centroids, n_centroids * sizeof(struct pixel),
               cudaMemcpyHostToDevice);
    cudaMemcpy(labels_dev, labels, n_pixels * sizeof(size_t),
               cudaMemcpyHostToDevice);

    // seed rand
    srand(time(NULL));

    // allocate auxiliary memory shared between host and device
    struct pixel *sums, *sums_dev;
    size_t *counts, *counts_dev;

    sums = malloc(n_centroids * sizeof(struct pixel));
    counts = malloc(n_centroids *  sizeof(size_t));

    cudaMalloc(&sums_dev, n_centroids * sizeof(struct pixel));
    cudaMalloc(&counts_dev, n_centroids *  sizeof(size_t));

    // randomly initialize centroids
    for (size_t i = 0u; i < n_centroids; ++i) {
        centroids[i] = pixels[rand() % n_pixels];

        struct pixel tmp = { 0.0, 0.0, 0.0 };
        sums[i] = tmp;

        counts[i] = 0u;
    }

    // repeat for KMEANS_MAX_ITER or until solution is stationary
    int iter;
    for (iter = 0; iter < KMEANS_MAX_ITER; ++iter) {
        int *done;
        cudaMallocManaged(&done, sizeof(int));
        *done = 1;

        // reassign points to closest centroids
        reassign_points<<<blocks, KMEANS_CUDA_BLOCKSIZE>>>(
            pixels_dev, n_pixels, centroids_dev, n_centroids, labels_dev,
            sums, counts, done);

        // repair empty clusters
        for (size_t i = 0u; i < n_centroids; ++i) {
            if (counts[i])
                continue;

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
            struct pixel largest_cluster_centroid = centroids[largest_cluster];

            size_t furthest_pixel = 0u;
            double max_dist = 0.0;
            for (size_t j = 0u; j < n_pixels; ++j) {
                if (labels[j] != largest_cluster)
                    continue;

                double dist = pixel_dist(pixels[j], largest_cluster_centroid);

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

        // average accumulated cluster sums
        for (int j = 0; j < n_centroids; ++j) {
            struct pixel *centroid = &centroids[j];
            struct pixel *sum = &sums[j];
            size_t count = counts[j];

            centroid->r = sum->r / count;
            centroid->g = sum->g / count;
            centroid->b = sum->b / count;

            sum->r = 0.0;
            sum->g = 0.0;
            sum->b = 0.0;

            counts[j] = 0u;
        }

        // break if no pixel has changed cluster
        if (*done)
            break;
    }

    // copy device memory back to host
    cudaMemcpy(pixels, pixels_dev, n_pixels * sizeof(struct pixel),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, centroids_dev, n_centroids * sizeof(struct pixel),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, labels_dev, n_pixels * sizeof(size_t),
               cudaMemcpyDeviceToHost);

    // free host and device memory
    cudaFree(pixels_dev);
    cudaFree(centroids_dev);
    cudaFree(labels_dev);
    cudaFree(sums);
    cudaFree(counts);
}
