#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

extern "C" {
  #include "kmeans.h"
}

#define MAX_CLUSTERS 10

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " IMAGE N_CLUSTERS\n"; 
    return 1;
  }
  
  // load image
  cv::Mat image = cv::imread(argv[1]);
  if (image.empty()) {
    std::cerr << "Failed to load image file '" << argv[1] << "'\n";
    return 2;
  }

  int n_clusters;
  try {
    size_t idx;
    n_clusters = std::stoi(argv[2], &idx);

    if (idx != strlen(argv[2]))
      throw std::invalid_argument("trailing garbage");
    else if (n_clusters <= 0 || n_clusters > MAX_CLUSTERS)
      throw std::invalid_argument("must be in range [1,  "
                                  + std::to_string(MAX_CLUSTERS) + "]");

  } catch (std::exception const &e) {
    std::cerr << "Could not evaluate number of clusters: " << e.what() << "\n";
    return 3;
  }

  // opencv
  // TODO

  // C reference implementation
  // TODO

  // C reference implementation using OpenMP
  // TODO

  // CUDA implementation
  // TODO

  // display result
  std::string disp_title(argv[1]);
  disp_title += " (clustered)";

  cv::namedWindow(disp_title, cv::WINDOW_AUTOSIZE);
  cv::imshow(disp_title, image);
  cv::waitKey(0);
  cv::destroyWindow(disp_title);

  return 0;
}
