#include <float.h> /* DBL_MAX */
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "kmeans_config.h"
#include "kmeans_c.h"

void kmeans_c(double *data_points, int n_data_points,
              double *centroids, int n_clusters, int *labels)
{
    srand(time(NULL));

    /* initialize centroids */
    for (int c = 0; c < n_clusters; ++c) {
        int c_idx = 3 * c;
        int d_idx = 3 * (rand() % n_data_points);

        for (int i = 0; i < 3; ++i)
            centroids[c_idx + i] = data_points[d_idx + i];
    }

    /* allocate auxilliary heap memory */
    int *points_per_cluster = calloc(n_clusters, sizeof(int));

    for (int iter = 0; iter < KMEANS_MAX_ITER; ++iter) {

        /* assign labels to data points */
        for (int d = 0; d < n_data_points; ++d) {
            int d_idx = 3 * d;

            double min_dist = DBL_MAX;
            for (int c = 0; c < n_clusters; ++c) {
                int c_idx = 3 * c;

                double acc = 0.0;
                for (int i = 0; i < 3; ++i) {
                    double delta = data_points[d_idx + i] - centroids[c_idx + i];
                    acc += delta * delta;
                }

                double dist = sqrt(acc);

                if (min_dist < 0 || dist < min_dist) {
                    labels[d] = c;
                    min_dist = dist;
                }
            }
        }

        /* null centroids */
        for (int c = 0; c < n_clusters; ++c) {
            int c_idx = 3 * c;
            for (int i = 0; i < 3; ++i)
                centroids[c_idx + i] = 0.0;
        }

        /* null points per cluster */
        for (int c = 0; c < n_clusters; ++c)
            points_per_cluster[c] = 0.0;

        /* sum data points for each centroid */
        for (int d = 0; d < n_data_points; ++d) {
            int label = labels[d];

            int d_idx = 3 * d;
            int c_idx = 3 * label;
            for (int i = 0; i < 3; ++i)
                centroids[c_idx + i] += data_points[d_idx + i];

            points_per_cluster[label]++;
        }

        for (int c = 0; c < n_clusters; ++c) {
            int c_idx = 3 * c;
            int n_points = points_per_cluster[c];

            if (n_points != 0) { /* normalize centroids */
                for (int i = 0; i < 3; ++i)
                    centroids[c_idx + i] /= n_points;
            } else { /* choose new random centroids */
                int rand_idx = 3 * (rand() % n_data_points);
                for (int i = 0; i < 3; ++i)
                    centroids[c_idx + i] = data_points[rand_idx + i];
            }
        }
    }

    free(points_per_cluster);
}

void kmeans_c2d(double **data_points, int n_data_points,
                double **centroids, int n_clusters, int *labels)
{
    srand(time(NULL));

    /* initialize centroids */
    for (int c = 0; c < n_clusters; ++c) {
        double *centr = centroids[c];

        int r = rand() % n_data_points;
        double *dp = data_points[r];

        for (int i = 0; i < 3; ++i)
            centr[i] = dp[i];
    }

    /* allocate auxilliary heap memory */
    int *points_per_cluster = calloc(n_clusters, sizeof(int));

    for (int iter = 0; iter < KMEANS_MAX_ITER; ++iter) {

        /* assign labels to data points */
        for (int d = 0; d < n_data_points; ++d) {

            double min_dist = DBL_MAX;
            for (int c = 0; c < n_clusters; ++c) {
                double *dp = data_points[d];
                double *centr = centroids[c];

                double acc = 0.0;
                for (int i = 0; i < 3; ++i) {
                    double delta = dp[i] - centr[i];
                    acc += delta * delta;
                }

                double dist = sqrt(acc);

                if (dist < min_dist) {
                    labels[d] = c;
                    min_dist = dist;
                }
            }
        }

        /* null centroids */
        for (int c = 0; c < n_clusters; ++c) {
            for (int i = 0; i < 3; ++i)
                centroids[c][i] = 0.0;
        }

        /* null points per cluster */
        for (int c = 0; c < n_clusters; ++c)
            points_per_cluster[c] = 0.0;

        /* sum data points for each centroid */
        for (int d = 0; d < n_data_points; ++d) {
            int label = labels[d];

            double *dp = data_points[d];
            double *centr = centroids[label];
            for (int i = 0; i < 3; ++i)
                centr[i] += dp[i];

            points_per_cluster[label]++;
        }

        for (int c = 0; c < n_clusters; ++c) {
            double *centr = centroids[c];
            int n_points = points_per_cluster[c];

            if (n_points != 0) { /* normalize centroids */
                for (int i = 0; i < 3; ++i)
                    centr[i] /= n_points;
            } else { /* choose new random centroids */
                int r = rand() % n_data_points;
                for (int i = 0; i < 3; ++i)
                    centr[i] = data_points[r][i];
            }
        }

    }

    free(points_per_cluster);
}
