This is a collection of implementations of the kmeans clustering algorithm
using pure C, C + OpenMP, CUDA C and OpenCV. All implementations can be applied
to a sample image and compared via a benchmark program and accompanying
visualization script.

Run `make demo` preview the different implementations (results should be
identical or at least very similar, you will need a C compiler supporting
OpenMP 4.5, an installation of OpenCV 3 and an Nvidia GPU with appropriate
compute capability).

Running `make benchmark` will re-generate the .csv files under `benchmarks` (
they must be removed beforehand).

A complete lab report (in German) can be found under `report/pdf/report.pdf`.
