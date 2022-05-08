gemm: main.cpp
	g++ -fopenmp -msse4.1 -msse2 main.cpp -o gemm -O0 -Wall -std=c++11

.PHONY: clean
clean:
	- rm -f gemm

.PHONY: run
run: 
	./gemm

.PHONY: rmdata
rmdata:
	- rm -rf data*	 