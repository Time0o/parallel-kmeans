#ifndef KMEANS_C_H
#define KMEANS_C_H

void kmeans_c(double *data_points, int n_data_points,
              double *centroids, int n_clusters, int *labels);

void kmeans_c2d(double **data_points, int n_data_points,
                double **centroids, int n_clusters, int *labels);

#endif
