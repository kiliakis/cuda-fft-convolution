CC=nvcc
CFLAGS=-O4 --compiler-options '-fPIC' '-std=c++11'
NVFLAGS=-arch compute_35 -code sm_35
LDFLAGS=--shared -lcufft
INCLUDE=-I../include
LIB=libfftconvolve.so
LIB_SOURCE=fftconvolve.cu

all: $(LIB)
# %.o: %.cu
# 	$(CC) $(CFLAGS) $(INCLUDE) -L$(DIR)/lib -o $@ $^ -lpapiprof

$(LIB): $(LIB_SOURCE)
	$(CC) $(NVFLAGS) $(CFLAGS) $(INCLUDE) $(LDFLAGS) $^ -o $@

clean:
	rm $(LIB)

.PHONY: install
install: $(LIB)
	mkdir -p $(PREFIX)/lib
	cp $(LIB) $(PREFIX)/lib/
	cp -r ../include $(PREFIX)/include
