#include <cstring>
#include <vector>

#include <opencv2/opencv.hpp>

#include "kmeans_wrapper.h"

static int parse_intarg(char const *arg)
{
    int res = 0;
    try {
        size_t idx;
        res = std::stoi(arg, &idx);

        if (idx != strlen(arg))
            throw std::invalid_argument("trailing garbage");

    } catch (std::exception const &e) {
        throw std::invalid_argument(
            std::string("malformed integer param: ") + e.what());
    }

    return res;
}

int main(int argc, char **argv)
{
    std::vector<std::pair<std::string, KmeansWrapper *>> wrappers;

    int const dim_min = parse_intarg(argv[1]);
    int const dim_max = parse_intarg(argv[2]);
    int const dim_step = parse_intarg(argv[3]);

    int const clusters_min = parse_intarg(argv[4]);
    int const clusters_max = parse_intarg(argv[5]);
    int const clusters_step = parse_intarg(argv[6]);

    int const n_exec = parse_intarg(argv[7]);

    std::string csvdir(argv[8]);
    if (csvdir.back() != '/')
        csvdir += '/';

    KmeansOpenCVWrapper opencv_wrapper;
    wrappers.push_back(std::make_pair("OpenCV", &opencv_wrapper));

    KmeansPureCWrapper pure_c_wrapper;
    wrappers.push_back(std::make_pair("C", &pure_c_wrapper));

    KmeansOMPWrapper omp_wrapper;
    wrappers.push_back(std::make_pair("OpenMP", &omp_wrapper));

    for (size_t i = 0u; i < wrappers.size(); ++i) {
        std::string &name = std::get<0>(wrappers[i]);
        KmeansWrapper *wrapper = std::get<1>(wrappers[i]);

        std::string outfile(name + ".csv");
        std::string csvfile(csvdir + outfile);

        std::ifstream is(csvfile);
        if (is.good())
            continue;

        std::cout << "creating: " << csvfile << '\n';

        std::ofstream os(csvfile);
        os << "dim,clusters,time\n";

        for (int dim = dim_min; dim <= dim_max; dim += dim_step) {
            std::cout << dim << "x" << dim << "...\n";

            cv::Mat image = cv::Mat::zeros(dim, dim, CV_8UC3);
            cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

            for (int c = clusters_min; c <= clusters_max; c += clusters_step) {
                for (int i = 0; i <= n_exec; ++i) {
                    wrapper->exec(image, c);
                    double t = wrapper->get_exec_time();
                    os << dim << ',' << c << ',' << t << '\n';
                }
            }
        }
    }
}
