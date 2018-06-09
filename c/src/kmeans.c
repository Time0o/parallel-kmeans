#include <float.h>
#include <math.h>
#include <omp.h>
#ifdef PROFILE
#include <stdio.h>
#endif
#include <stdlib.h>
#include <time.h>

#include "kmeans_config.h"
#include "kmeans.h"

// compute euclidean distance between two pixel values
static inline double pixel_dist(struct pixel p1, struct pixel p2)
{
    double dr = p1.r - p2.r;
    double dg = p1.g - p2.g;
    double db = p1.b - p2.b;

    return sqrt(dr * dr + dg * dg + db * db);
}

// find index of centroid with least distance to some pixel
static inline size_t find_closest_centroid(
    struct pixel pixel, struct pixel *centroids, size_t n_centroids)
{
    size_t closest_centroid = 0u;
    double min_dist = DBL_MAX;

    for (size_t i = 0; i < n_centroids; ++i) {
        double dist = pixel_dist(pixel, centroids[i]);

        if (dist < min_dist) {
            closest_centroid = i;
            min_dist = dist;
        }
    }

    return closest_centroid;
}

void kmeans_c(struct pixel *pixels, size_t n_pixels,
              struct pixel *centroids, size_t n_centroids,
              size_t *labels)
{
#ifdef PROFILE
    clock_t exec_begin;
    double exec_time_kernel1;
    double exec_time_kernel2;
    double exec_time_kernel3;
    double exec_time_kernel4;
#endif

    // seed rand
    srand(time(NULL));

    // allocate auxiliary heap memory
    struct pixel *sums = malloc(n_centroids * sizeof(struct pixel));
    size_t *counts = malloc(n_centroids *  sizeof(size_t));

    // randomly initialize centroids
#ifdef PROFILE
    exec_begin = clock();
#endif
    for (size_t i = 0u; i < n_centroids; ++i) {
        centroids[i] = pixels[rand() % n_pixels];

        struct pixel tmp = { 0.0, 0.0, 0.0 };
        sums[i] = tmp;

        counts[i] = 0u;
    }
#ifdef PROFILE
    exec_time_kernel1 = (double) (clock() - exec_begin) / CLOCKS_PER_SEC;
#endif

    // repeat for KMEANS_MAX_ITER or until solution is stationary
    int iter;
    for (iter = 0; iter < KMEANS_MAX_ITER; ++iter) {
        int done = 1;

        // reassign points to closest centroids
#ifdef PROFILE
        exec_begin = clock();
#endif
        for (size_t i = 0u; i < n_pixels; ++i) {
            struct pixel pixel = pixels[i];

            // find centroid closest to pixel
            size_t closest_centroid =
                find_closest_centroid(pixel, centroids, n_centroids);

            // if pixel has changed cluster...
            if (closest_centroid != labels[i]) {
                labels[i] = closest_centroid;

                done = 0;
            }

            // update cluster sum
            struct pixel *sum = &sums[closest_centroid];
            sum->r += pixel.r;
            sum->g += pixel.g;
            sum->b += pixel.b;

            // update cluster size
            counts[closest_centroid]++;
        }
#ifdef PROFILE
        exec_time_kernel2 = (double) (clock() - exec_begin) / CLOCKS_PER_SEC;
#endif

        // repair empty clusters
#ifdef PROFILE
        exec_begin = clock();
#endif
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
#ifdef PROFILE
        exec_time_kernel3 = (double) (clock() - exec_begin) / CLOCKS_PER_SEC;
#endif

        // average accumulated cluster sums
#ifdef PROFILE
        exec_begin = clock();
#endif
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
#ifdef PROFILE
        exec_time_kernel4 = (double) (clock() - exec_begin) / CLOCKS_PER_SEC;
#endif

        // break if no pixel has changed cluster
        if (done)
            break;
    }
#ifdef PROFILE
    printf("Total kernel execution times:\n");
    printf("Kernel 1 (random centroid initialization): %.3e\n", exec_time_kernel1);
    printf("Kernel 2 (reassigning points to closest centroids): %.3e\n", exec_time_kernel2);
    printf("Kernel 3 (repairing empty clusters): %.3e\n", exec_time_kernel3);
    printf("Kernel 4 (average accumulated centroids): %.3e\n", exec_time_kernel4);
#endif

    free(sums);
    free(counts);
}

void kmeans_omp(struct pixel *pixels, size_t n_pixels,
                struct pixel *centroids, size_t n_centroids,
                size_t *labels)
{
    // seed rand
    srand(time(NULL));

    // allocate auxiliary heap memory
    double *sums = malloc(3 * n_centroids * sizeof(double));
    size_t *counts = malloc(n_centroids * sizeof(size_t));

    // randomly initialize centroids
    for (size_t i = 0u; i < n_centroids; ++i) {
        centroids[i] = pixels[rand() % n_pixels];

        double *sum = &sums[3 * i];
        sum[0] = sum[1] = sum[2] = 0.0;

        counts[i] = 0u;
    }

    // repeat for KMEANS_MAX_ITER or until solution is stationary
    for (int iter = 0; iter < KMEANS_MAX_ITER; ++iter) {
        int done = 1;

        // reassign points to closest centroids
        #pragma omp parallel for \
            reduction(+ : sums[:(3 * n_centroids)], counts[:n_centroids])
        for (int i = 0; i < n_pixels; ++i) {
            struct pixel pixel = pixels[i];

            // find centroid closest to pixel
            int closest_centroid =
                find_closest_centroid(pixel, centroids, n_centroids);

            // if pixel has changed cluster...
            if (closest_centroid != labels[i]) {
                labels[i] = closest_centroid;

                #pragma omp atomic write
                done = 0;
            }

            // update cluster sum
            double *sum = &sums[3 * closest_centroid];
            sum[0] += pixel.r;
            sum[1] += pixel.g;
            sum[2] += pixel.b;

            // update cluster size
            counts[closest_centroid]++;
        }

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
            #pragma omp parallel for
            for (size_t j = 0u; j < n_pixels; ++j) {
                if (labels[j] != largest_cluster)
                    continue;

                double dist = pixel_dist(pixels[j], largest_cluster_centroid);

                #pragma omp critical
                {
                    if (dist > max_dist) {
                        furthest_pixel = j;
                        max_dist = dist;
                    }
                }
            }

            // move that pixel to the empty cluster
            struct pixel replacement_pixel = pixels[furthest_pixel];
            centroids[i] = replacement_pixel;
            labels[furthest_pixel] = i;

            // correct cluster sums
            double *sum = &sums[3 * i];
            sum[0] = replacement_pixel.r;
            sum[1] = replacement_pixel.g;
            sum[2] = replacement_pixel.b;

            sum = &sums[3 * largest_cluster];
            sum[0] -= replacement_pixel.r;
            sum[1] -= replacement_pixel.g;
            sum[2] -= replacement_pixel.b;

            // correct cluster sizes
            counts[i] = 1u;
            counts[largest_cluster]--;
        }

        // average accumulated cluster sums
        for (int j = 0; j < n_centroids; ++j) {
            struct pixel *centroid = &centroids[j];
            double *sum = &sums[3 * j];
            size_t count = counts[j];

            centroid->r = sum[0] / count;
            centroid->g = sum[1] / count;
            centroid->b = sum[2] / count;

            sum[0] = sum[1] = sum[2] = 0.0;
            counts[j] = 0u;
        }

        // break if no pixel has changed cluster
        if (done)
            break;
    }

    free(sums);
    free(counts);
}
