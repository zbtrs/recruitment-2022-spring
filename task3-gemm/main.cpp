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

int A[2049 * 2049],B[2049 * 2049],C[2049 * 2049];

void print(const int &N,vec &c) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ",c[i * N + j]);
        }
        printf("\n");
    }
}

void Gemm(const int &size, vec &a, vec &b, vec &c) {
    int len = a.size();
    for (int i = 0; i < len; ++i) {
        A[i] = a[i];
        B[i] = b[i];
        C[i] = 0;
    }


    const int N = size;
    //printf("%d\n",N);
    for (int i = 0; i < N; i += 4)
        for (int j = 0; j < N; j += 4)
        {
            register int c00 = 0,c01 = 0,c02 = 0,c03 = 0,c10 = 0,c11 = 0,c12 = 0,c13 = 0,c20 = 0,c21 = 0,c22 = 0,c23 = 0,c30 = 0,c31 = 0,c32 = 0,c33 = 0;
            register int a0i,a1i,a2i,a3i;
            register int bi0,bi1,bi2,bi3;
            int *a0i_p,*a1i_p,*a2i_p,*a3i_p;
            a0i_p = A + i * N;
            a1i_p = A + (i + 1) * N;
            a2i_p = A + (i + 2) * N;
            a3i_p = A + (i + 3) * N;
            for (int k = 0; k < N; ++k) {
                bi0 = B[k * N + j];
                bi1 = B[k * N + j + 1];
                bi2 = B[k * N + j + 2];
                bi3 = B[k * N + j + 3];
                a0i = *a0i_p++;
                a1i = *a1i_p++;
                a2i = *a2i_p++;
                a3i = *a3i_p++;

                c00 += a0i * bi0;
                c01 += a0i * bi1;
                c02 += a0i * bi2;
                c03 += a0i * bi3;

                c10 += a1i * bi0;
                c11 += a1i * bi1;
                c12 += a1i * bi2;
                c13 += a1i * bi3;

                c20 += a2i * bi0;
                c21 += a2i * bi1;
                c22 += a2i * bi2;
                c23 += a2i * bi3;

                c30 += a3i * bi0;
                c31 += a3i * bi1;
                c32 += a3i * bi2;
                c33 += a3i * bi3;
            }
            C[i * N + j] += c00;C[i * N + j + 1] += c01;C[i * N + j + 2] += c02;C[i * N + j + 3] += c03;
            C[(i + 1) * N + j] += c10;C[(i + 1) * N + j + 1] += c11;C[(i + 1) * N + j + 2] += c12;C[(i + 1) * N + j + 3] += c13;
            C[(i + 2) * N + j] += c20;C[(i + 2) * N + j + 1] += c21;C[(i + 2) * N + j + 2] += c22;C[(i + 2) * N + j + 3] += c23;
            C[(i + 3) * N + j] += c30;C[(i + 3) * N + j + 1] += c31;C[(i + 3) * N + j + 2] += c32;C[(i + 3) * N + j + 3] += c33;
        }
    for (int i = 0; i < len; ++i)
        c[i] = C[i];

    //print(size,c);
    //printf("%d\n",c[0]);
    /*
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
     */
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


    for(auto size: scale) {
        cout << "Running, dataset: size " << size << endl;
        Benchmark(size);
        cout << "Passed, dataset: size " << size << endl;
        cout << endl;
    }
    return 0;
}