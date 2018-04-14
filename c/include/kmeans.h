#ifndef KMEANS_H
#define KMEANS_H

struct pixel
{
    double r, g, b;
};

void kmeans(struct pixel *pixels, size_t n_pixels,
            struct pixel *centroids, size_t n_centroids,
            size_t *labels);

void kmeans_omp(struct pixel *pixels, size_t n_pixels,
                struct pixel *centroids, size_t n_centroids,
                size_t *labels);

#endif
