# directories
C_SRC_DIR=c/src
C_OBJ_DIR=c/obj
C_INCLUDE_DIR=c/include

CPP_SRC_DIR=cpp/src
CPP_OBJ_DIR=cpp/obj
CPP_INCLUDE_DIR=cpp/include

CONFIG_DIR=config
BUILD_DIR=build
IMAGE_DIR=images
REPORT_DIR=report
REPORT_AUX_DIR=$(REPORT_DIR)/aux
REPORT_PDF_DIR=$(REPORT_DIR)/pdf
REPORT_TEX_DIR=$(REPORT_DIR)/tex

# files
_CONFIG_DEPS=kmeans_config.h
CONFIG_DEPS=$(patsubst %, $(CONFIG_DIR)/%, $(_CONFIG_DEPS))

_C_DEPS=kmeans.h
C_DEPS=$(patsubst %, $(C_INCLUDE_DIR)/%, $(_C_DEPS))

_CPP_DEPS=kmeans_wrapper.h
CPP_DEPS=$(patsubst %, $(CPP_INCLUDE_DIR)/%, $(_CPP_DEPS))

DEPS=$(CONFIG_DEPS) $(C_DEPS) $(CPP_DEPS)

_C_OBJ=kmeans.o
C_OBJ=$(patsubst %, $(C_OBJ_DIR)/%, $(_C_OBJ))

_CPP_OBJ=kmeans_wrapper.o
CPP_OBJ=$(patsubst %, $(CPP_OBJ_DIR)/%, $(_CPP_OBJ))

# compilation settings
CC_C=gcc
CC_CPP=g++

COMMON_CFLAGS=-Wall -g -O0 -I$(CONFIG_DIR) -I$(C_INCLUDE_DIR) -I$(CPP_INCLUDE_DIR) -fopenmp

C_CFLAGS=-std=c99 $(COMMON_CFLAGS)
CPP_CFLAGS=-std=c++11 $(COMMON_CFLAGS) `pkg-config --cflags opencv`

LFLAGS=`pkg-config --libs opencv` -fopenmp

#functions
define run_pdflatex
	pdflatex -halt-on-error -output-directory $(REPORT_AUX_DIR) $< > /dev/null
endef

# benchmark parameters
BENCHMARK_DIM_MIN=100
BENCHMARK_DIM_MAX=500
BENCHMARK_DIM_STEP=100

BENCHMARK_CLUSTER_MIN=2
BENCHMARK_CLUSTER_MAX=20
BENCHMARK_CLUSTER_STEP=18

BENCHMARK_OUTDIR=benchmarks

BENCHMARK_PLOT=tool/plot.py

# demo parameters
DEMO_CLUSTERS=5
DEMO_IMAGE=demo_image.jpg

# rules
demo: $(CPP_OBJ_DIR)/kmeans_demo.o $(C_OBJ) $(CPP_OBJ)
	$(CC_CPP) -o $(BUILD_DIR)/$@ $^ $(LFLAGS)
	@chmod +x $(BUILD_DIR)/$@
	./$(BUILD_DIR)/$@ $(IMAGE_DIR)/$(DEMO_IMAGE) $(DEMO_CLUSTERS)

benchmark: $(CPP_OBJ_DIR)/kmeans_benchmark.o $(C_OBJ) $(CPP_OBJ)
	$(CC_CPP) -o $(BUILD_DIR)/$@ $^ $(LFLAGS)
	@chmod +x $(BUILD_DIR)/$@
	./$(BUILD_DIR)/$@ \
	$(BENCHMARK_DIM_MIN) $(BENCHMARK_DIM_MAX) $(BENCHMARK_DIM_STEP) \
	$(BENCHMARK_CLUSTER_MIN) $(BENCHMARK_CLUSTER_MAX) $(BENCHMARK_CLUSTER_STEP) \
	$(BENCHMARK_OUTDIR)
	./$(BENCHMARK_PLOT) $(BENCHMARK_OUTDIR)

report: $(REPORT_TEX_DIR)/report.tex
	$(call run_pdflatex)
	$(call run_pdflatex)
	-@mv $(REPORT_AUX_DIR)/report.pdf $(REPORT_PDF_DIR)

$(C_OBJ_DIR)/%.o: $(C_SRC_DIR)/%.c $(C_DEPS) $(CONFIG_DEPS)
	$(CC_C) -c -o $@ $< $(C_CFLAGS)

$(CPP_OBJ_DIR)/%.o: $(CPP_SRC_DIR)/%.cc $(DEPS)
	$(CC_CPP) -c -o $@ $< $(CPP_CFLAGS)

.PHONY: clean
clean:
	rm $(C_OBJ_DIR)/*.o 2> /dev/null || true
	rm $(CPP_OBJ_DIR)/*.o 2> /dev/null || true
	rm $(BUILD_DIR)/* 2> /dev/null || true
	rm $(REPORT_AUX_DIR)/* 2> /dev/null || true
	rm $(REPORT_PDF_DIR)/* 2> /dev/null || true
