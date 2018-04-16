#include <omp.h>
#include <opencv2/core/core.hpp>

#include "kmeans_config.h"
#include "kmeans_wrapper.h"

void KmeansCWrapper::exec(cv::Mat const &image, size_t n_centroids) {

    size_t n_pixels = image.rows * image.cols;

    // allocate memory
    std::vector<pixel> pixels(n_pixels);
    for (int y = 0; y < image.rows; ++y) {
        for(int x = 0; x < image.cols; ++x) {
            int idx = y * image.cols + x;
            pixels[idx].r = image.at<cv::Vec3b>(y, x)[0];
            pixels[idx].g = image.at<cv::Vec3b>(y, x)[1];
            pixels[idx].b = image.at<cv::Vec3b>(y, x)[2];
        }
    }

    std::vector<pixel> centroids(n_centroids);
    for (size_t i = 0; i < n_centroids; ++i) {
        centroids[i].r = 0.0;
        centroids[i].g = 0.0;
        centroids[i].b = 0.0;
    }

    std::vector<size_t> labels(n_pixels);

    // perform calculations
    if (cores)
        omp_set_num_threads(cores);

    start_timer();
    impl(&pixels[0], n_pixels, &centroids[0], n_centroids, &labels[0]);
    stop_timer();

    // rebuild image from results
    result = cv::Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int label = labels[y * image.cols + x];
            result.at<cv::Vec3b>(y, x)[0] = centroids[label].r;
            result.at<cv::Vec3b>(y, x)[1] = centroids[label].g;
            result.at<cv::Vec3b>(y, x)[2] = centroids[label].b;
        }
    }
}

void KmeansOpenCVWrapper::exec(cv::Mat const &image, size_t n_centroids) {

    // construct input data points vector
    cv::Mat data_points(image.rows * image.cols, 3, CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel)
                data_points.at<float>(y *  image.cols + x, channel) =
                    image.at<cv::Vec3b>(y, x)[channel];
        }
    }

    // allocate space for labels
    cv::Mat labels;

    // transform cluster centers into **double format
    cv::Mat centroids;

    // specify termination criteria
    cv::TermCriteria term(CV_TERMCRIT_ITER, KMEANS_MAX_ITER, 0);

    // perform calculations
    start_timer();
    cv::kmeans(data_points, n_centroids, labels, term, 1,
               cv::KMEANS_RANDOM_CENTERS, centroids);
    stop_timer();

    // rebuild image from results
    result = cv::Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int idx = labels.at<int>(y * image.cols +  x, 0);
            for (int i = 0; i < 3; ++i)
                result.at<cv::Vec3b>(y, x)[i] = centroids.at<float>(idx, i);
        }
    }
}
