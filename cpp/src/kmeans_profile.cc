#include <opencv2/opencv.hpp>

#include "kmeans_wrapper.h"

int main(int argc, char **argv)
{
    cv::Mat image;
    int n_clusters;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " IMAGE CLUSTERS\n";
        return -1;
    }

    // load image
    image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Failed to load image file '" << argv[1] << "'\n";
        return -1;
    }

    // parse number of clusters
    try {
        size_t idx;
        n_clusters = std::stoi(argv[2], &idx);

        if (idx != strlen(argv[2]))
            throw std::invalid_argument("trailing garbage");

    } catch (std::exception const &e) {
        std::cerr << "Failed to parse number of clusters: " << e.what() << '\n';
        return -1;
    }

    KmeansPureCWrapper wrapper;
    wrapper.exec(image, n_clusters);

    return 0;
}
