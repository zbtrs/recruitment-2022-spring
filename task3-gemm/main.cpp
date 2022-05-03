#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>

#define PRINT_TIME(code) do { \
    auto start = system_clock::now(); \
    code \
    auto end   = system_clock::now(); \
    auto duration = duration_cast<microseconds>(end - start); \
    cout << "time spent: " << double(duration.count()) << "us" << endl; \
} while(0)

using namespace std;

using namespace chrono;

using vec = vector<int>; 

const int scale[] = {256, 512, 1024, 2048};
const string data_path("./data/");

void Gemm(const int &size, vec &a, vec &b, vec &c) {
    int N = 32;
    int T = size / N;
    int k = 0,i = 0,j = 0;
    for (int it = 0; it < N; ++it) {
        for (int kt = 0; kt < N; ++kt) {
            for (int jt = 0; jt < N; ++jt) {
                int ktt = kt * T,itt = it * T,jtt = jt * T;
                for (i = itt; i < itt + T; ++i) {
                    int num_i = i * size;
                    for (k = ktt; k < ktt + T; ++k) {
                        int num_k = k * size;
                        int r = a[num_i + k];
                        for (j = jtt; j < jtt + T; ++j)
                            c[num_i + j] += r * b[num_k + j];
                    }
                }
            }
        }
    }
//#pragma omp parallel for schedule(dynamic)
    for(;i < size; ++i) {
        int num_i = i * size;
        for(; k < size; ++k) {
            int num_k = k * size;
            int r = a[num_i + k];
            for(;j < size; ++j) {
                c[num_i + j] += r * b[num_k + j];
            }
        }
    }
}

void CheckResult(const vec &c, const string &result_path) {
    ifstream file_result(result_path);
    int nelems = c.size();
    float res_i;
    for(int i = 0; i < nelems; i++) {
        file_result >> res_i;
        assert(c[i] == res_i);
    }
    file_result.close();
}

// c = a * b
void Benchmark(const int &size) {
    const int nelems = size * size;
    const string a_path(data_path+to_string(size)+"/a");
    const string b_path(data_path+to_string(size)+"/b");
    const string result_path(data_path+to_string(size)+"/result");
    ifstream file_a(a_path);
    ifstream file_b(b_path);

    vec a(nelems, 0);
    vec b(nelems, 0);
    vec c(nelems, 0);

    for(int i = 0; i < nelems; i++) {
        file_a >> a[i];
    }
    for(int i = 0; i < nelems; i++) {
        file_b >> b[i];
    }

    PRINT_TIME(
       Gemm(size, a, b, c);
    );
    
    CheckResult(c, result_path);

    file_a.close();
    file_b.close();
}

int main() {
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("%d\n",id);
    }


    for(auto size: scale) {
        cout << "Running, dataset: size " << size << endl;
        Benchmark(size);
        cout << "Passed, dataset: size " << size << endl;
        cout << endl;
    }
    return 0;
}