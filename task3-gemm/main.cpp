#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <xmmintrin.h>
#include <cmath>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <chrono>
#include <vector>
#include <immintrin.h>
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

int A[2080*2080],B[2080*2080],C[2080*2080],pA[2080*2080],pB[2080*2080],pC[2080*2080];

void print(const int &N,vec &c) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ",c[i * N + j]);
        }
        printf("\n");
    }
}

void print2(__m128i a,int *B) {
    int A[10];
    memset(A,0,sizeof(A));
    _mm_storeu_si128((__m128i*)A,a);
    cout << "sb!! " << endl;
    cout << A[0] << " " << A[1] << " " << A[2] << " " << A[3] << endl;
    cout << B[0] << " " << B[1] << " " << B[2] << " " << B[3] << endl;
}

void print3(__m128i a,__m128i b,__m128i c) {
    int A[10],B[10],C[10];
    memset(A,0,sizeof(A));
    memset(B,0,sizeof(B));
    memset(C,0,sizeof(C));
    _mm_storeu_si128((__m128i*)A,a);
    _mm_storeu_si128((__m128i*)B,b);
    _mm_storeu_si128((__m128i*)C,c);
    for (int i = 0; i <= 3; i++)
        cout << B[i] << " " << C[i] << " " << A[i] << endl;

}

void print4(__m128i a) {
    int A[10];
    memset(A,0,sizeof(A));
    _mm_storeu_si128((__m128i*)A,a);
    for (int i = 0; i <= 3; i++)
        cout << A[i] << " ";
    cout << endl;
}

void packB(int K,int ld,int *B,int *pB) {
    int *dst = pB;
    for (int k = 0; k < K; k += 4) {
        *(dst + 0) = *(B + k + 0 + 0 * ld);
        *(dst + 1) = *(B + k + 0 + 1 * ld);
        *(dst + 2) = *(B + k + 0 + 2 * ld);
        *(dst + 3) = *(B + k + 0 + 3 * ld);
        *(dst + 4) = *(B + k + 1 + 0 * ld);
        *(dst + 5) = *(B + k + 1 + 1 * ld);
        *(dst + 6) = *(B + k + 1 + 2 * ld);
        *(dst + 7) = *(B + k + 1 + 3 * ld);
        *(dst + 8) = *(B + k + 2 + 0 * ld);
        *(dst + 9) = *(B + k + 2 + 1 * ld);
        *(dst + 10) = *(B + k + 2 + 2 * ld);
        *(dst + 11) = *(B + k + 2 + 3 * ld);
        *(dst + 12) = *(B + k + 3 + 0 * ld);
        *(dst + 13) = *(B + k + 3 + 1 * ld);
        *(dst + 14) = *(B + k + 3 + 2 * ld);
        *(dst + 15) = *(B + k + 3 + 3 * ld);
        dst += 16;
    }
}

void packA(int K,int ld,int *A,int *pA) {
    int *dst = pA;
    for (int k = 0; k < K; k += 4) {
        int *a0_k0_p = A + k * ld;
        int *a0_k1_p = A + (k + 1) * ld;
        int *a0_k2_p = A + (k + 2) * ld;
        int *a0_k3_p = A + (k + 3) * ld;
        *(dst + 0) = *(a0_k0_p + 0);
        *(dst + 1) = *(a0_k0_p + 1);
        *(dst + 2) = *(a0_k0_p + 2);
        *(dst + 3) = *(a0_k0_p + 3);
        *(dst + 4) = *(a0_k1_p + 0);
        *(dst + 5) = *(a0_k1_p + 1);
        *(dst + 6) = *(a0_k1_p + 2);
        *(dst + 7) = *(a0_k1_p + 3);
        *(dst + 8) = *(a0_k2_p + 0);
        *(dst + 9) = *(a0_k2_p + 1);
        *(dst + 10) = *(a0_k2_p + 2);
        *(dst + 11) = *(a0_k2_p + 3);
        *(dst + 12) = *(a0_k3_p + 0);
        *(dst + 13) = *(a0_k3_p + 1);
        *(dst + 14) = *(a0_k3_p + 2);
        *(dst + 15) = *(a0_k3_p + 3);
        dst += 16;
    }
}

void cal(int K,int ld,int *A,int *B,int *C) {
    __m128i c_c0,c_c1,c_c2,c_c3,a_ri,b_vi0,b_vi1,b_vi2,b_vi3;
    register int b0;
    c_c0 = _mm_setzero_si128(),c_c1 = _mm_setzero_si128(),c_c2 = _mm_setzero_si128(),c_c3 = _mm_setzero_si128();
    int *bptr = B;
    for (int i = 0; i < K; i++) {
        a_ri = _mm_loadu_si128((const __m128i*)A + i);
        b0 = (*bptr);
        ++bptr;
        b_vi0 = _mm_set_epi32(b0,b0,b0,b0);
        b0 = (*bptr);
        ++bptr;
        b_vi1 = _mm_set_epi32(b0,b0,b0,b0);
        b0 = (*bptr);
        ++bptr;
        b_vi2 = _mm_set_epi32(b0,b0,b0,b0);
        b0 = (*bptr);
        ++bptr;
        b_vi3 = _mm_set_epi32(b0,b0,b0,b0);

        c_c0 = _mm_add_epi32(c_c0,_mm_mullo_epi32(a_ri,b_vi0));
        c_c1 = _mm_add_epi32(c_c1,_mm_mullo_epi32(a_ri,b_vi1));
        c_c2 = _mm_add_epi32(c_c2,_mm_mullo_epi32(a_ri,b_vi2));
        c_c3 = _mm_add_epi32(c_c3,_mm_mullo_epi32(a_ri,b_vi3));
        /*
        temp = bi0_p[0];
        b_vi0 = _mm_set_epi32(temp,temp,temp,temp);
        temp1 = bi1_p[0];
        b_vi1 = _mm_set_epi32(temp1,temp1,temp1,temp1);
        temp2 = bi2_p[0];
        b_vi2 = _mm_set_epi32(temp2,temp2,temp2,temp2);
        temp3 = bi3_p[0];
        b_vi3 = _mm_set_epi32(temp3,temp3,temp3,temp3);
         */
    }
    __m128i tempp = _mm_loadu_si128((const __m128i*)C);
    _mm_storeu_si128((__m128i*)C,_mm_add_epi32(c_c0,tempp));
    tempp = _mm_loadu_si128((const __m128i*)C + (ld >> 2));
    _mm_storeu_si128((__m128i*)C + (ld >> 2),_mm_add_epi32(c_c1,tempp));
    tempp = _mm_loadu_si128((const __m128i*)C + ((2 * ld) >> 2));
    _mm_storeu_si128((__m128i*)C + ((2 * ld) >> 2),_mm_add_epi32(c_c2,tempp));
    tempp = _mm_loadu_si128((const __m128i*)C + ((3 * ld) >> 2));
    _mm_storeu_si128((__m128i*)C + ((3 * ld) >> 2),_mm_add_epi32(c_c3,tempp));
}

void block_packing(int M,int N,int K,int ld,int *A,int *B,int *C,bool should_pack) {
    for (int j = 0; j < N; j += 4) {
        if (should_pack)
         packB(K,ld,B + j * ld,pB + j * K);
        for (int i = 0; i < M; i += 4) {
            if (j == 0)
                packA(K,ld,A + i,pA + i * K);
            cal(K,ld,pA + i * K,pB + j * K,C + i + j * ld);
        }
    }
}

void do_block(int M,int N,int K,int ld,int *A,int *B,int *C) {
    for (int i = 0; i < M; i += 4)
        for (int j = 0; j < N; j += 4) {
            __m128i c_c0,c_c1,c_c2,c_c3,a_ri,b_vi0,b_vi1,b_vi2,b_vi3;
            c_c0 = _mm_setzero_si128(),c_c1 = _mm_setzero_si128(),c_c2 = _mm_setzero_si128(),c_c3 = _mm_setzero_si128();
            int *bi0_p,*bi1_p,*bi2_p,*bi3_p;

            bi0_p = B + j * ld;
            bi1_p = B + (j + 1) * ld;
            bi2_p = B + (j + 2) * ld;
            bi3_p = B + (j + 3) * ld;
            register int temp,temp1,temp2,temp3;
            for (int k = 0; k < K; k++) {

                a_ri = _mm_loadu_si128((const __m128i*)A + ((i + k * ld) >> 2));
                temp = bi0_p[0];
                b_vi0 = _mm_set_epi32(temp,temp,temp,temp);
                temp1 = bi1_p[0];
                b_vi1 = _mm_set_epi32(temp1,temp1,temp1,temp1);
                temp2 = bi2_p[0];
                b_vi2 = _mm_set_epi32(temp2,temp2,temp2,temp2);
                temp3 = bi3_p[0];
                b_vi3 = _mm_set_epi32(temp3,temp3,temp3,temp3);

                bi0_p++;
                bi1_p++;
                bi2_p++;
                bi3_p++;

                c_c0 = _mm_add_epi32(c_c0,_mm_mullo_epi32(a_ri,b_vi0));
                c_c1 = _mm_add_epi32(c_c1,_mm_mullo_epi32(a_ri,b_vi1));
                c_c2 = _mm_add_epi32(c_c2,_mm_mullo_epi32(a_ri,b_vi2));
                c_c3 = _mm_add_epi32(c_c3,_mm_mullo_epi32(a_ri,b_vi3));
            }

            //一个块被多次计算，要累加结果
            __m128i tempp = _mm_loadu_si128((const __m128i*)C + ((i + j * ld) >> 2));
            _mm_storeu_si128((__m128i*)C + ((i + j * ld) >> 2),_mm_add_epi32(c_c0,tempp));
            tempp = _mm_loadu_si128((const __m128i*)C + ((i + (j + 1) * ld) >> 2));
            _mm_storeu_si128((__m128i*)C + ((i + (j + 1) * ld) >> 2),_mm_add_epi32(c_c1,tempp));
            tempp = _mm_loadu_si128((const __m128i*)C + ((i + (j + 2) * ld) >> 2));
            _mm_storeu_si128((__m128i*)C + ((i + (j + 2) * ld) >> 2),_mm_add_epi32(c_c2,tempp));
            tempp = _mm_loadu_si128((const __m128i*)C + ((i + (j + 3) * ld) >> 2));
            _mm_storeu_si128((__m128i*)C + ((i + (j + 3) * ld) >> 2),_mm_add_epi32(c_c3,tempp));

        }
}

void Gemm(const int &size, vec &a, vec &b, vec &c) {

    //先转置

    const int N = size;
    const int ld = N + 8;
    int len = a.size(),row = 0,coloum = 0;
    for (int i = 0; i < len; i++) {
        C[i] = 0;
        row = i / (int)N;
        coloum = i % (int)N;
        A[coloum * ld + row] = a[i];
        B[coloum * ld + row] = b[i];
    }

    const int KC = 128,MC = 64;
    for (int k = 0; k < N; k += KC) {
        int K = min(N - k,KC);
        for (int i = 0; i < N; i += MC) {
            int M = min(N - i,MC);
            //do_block(M,N,K,ld,A + i + k * ld,B + k,C + i);
            block_packing(M,N,K,ld,A + i + k * ld,B + k,C + i,i == 0);
        }
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c[i * N + j] = C[j * ld + i];
        }
    //print(size,c);
    /*
    for (int i = 0; i < N; i += 4)
        for (int j = 0; j < N; j += 4) {
            __m128i c_c0,c_c1,c_c2,c_c3,a_ri,b_vi0,b_vi1,b_vi2,b_vi3;
            c_c0 = _mm_setzero_si128(),c_c1 = _mm_setzero_si128(),c_c2 = _mm_setzero_si128(),c_c3 = _mm_setzero_si128();
            int *bi0_p,*bi1_p,*bi2_p,*bi3_p;

            bi0_p = B + j * ld;
            bi1_p = B + (j + 1) * ld;
            bi2_p = B + (j + 2) * ld;
            bi3_p = B + (j + 3) * ld;
            register int temp,temp1,temp2,temp3;
            for (int k = 0; k < N; k++) {

                a_ri = _mm_loadu_si128((const __m128i*)A + ((i + k * ld) >> 2));

                //temp = bi0_p[0];
                //b_vi0 = _mm_set1_epi32(temp);
                //temp1 = bi1_p[0];
                //b_vi1 = _mm_set1_epi32(temp1);
                //temp2 = bi2_p[0];
                //b_vi2 = _mm_set1_epi32(temp2);
                //temp3 = bi3_p[0];
                //b_vi3 = _mm_set1_epi32(temp3);


                temp = bi0_p[0];
                b_vi0 = _mm_set_epi32(temp,temp,temp,temp);
                temp1 = bi1_p[0];
                b_vi1 = _mm_set_epi32(temp1,temp1,temp1,temp1);
                temp2 = bi2_p[0];
                b_vi2 = _mm_set_epi32(temp2,temp2,temp2,temp2);
                temp3 = bi3_p[0];
                b_vi3 = _mm_set_epi32(temp3,temp3,temp3,temp3);

                //b_vi0 = _mm_loadu_si128((const __m128i*)bi0_p);
                //b_vi1 = _mm_loadu_si128((const __m128i*)bi1_p);
                //b_vi2 = _mm_loadu_si128((const __m128i*)bi2_p);
                //b_vi3 = _mm_loadu_si128((const __m128i*)bi3_p);
                bi0_p++;
                bi1_p++;
                bi2_p++;
                bi3_p++;

                c_c0 = _mm_add_epi32(c_c0,_mm_mullo_epi32(a_ri,b_vi0));
                c_c1 = _mm_add_epi32(c_c1,_mm_mullo_epi32(a_ri,b_vi1));
                c_c2 = _mm_add_epi32(c_c2,_mm_mullo_epi32(a_ri,b_vi2));
                c_c3 = _mm_add_epi32(c_c3,_mm_mullo_epi32(a_ri,b_vi3));
            }
            _mm_storeu_si128((__m128i*)C + ((i + j * ld) >> 2),c_c0);
            _mm_storeu_si128((__m128i*)C + ((i + (j + 1) * ld) >> 2),c_c1);
            _mm_storeu_si128((__m128i*)C + ((i + (j + 2) * ld) >> 2),c_c2);
            _mm_storeu_si128((__m128i*)C + ((i + (j + 3) * ld) >> 2),c_c3);
        }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            c[i * N + j] = C[j * ld + i];
        }
    //print(N,c);
     */

    /*
    const int N = size;
    int len = a.size();
    for (int i = 0; i < len; ++i) {
        A[i] = a[i];
        B[i] = b[i];
        C[i] = 0;
    }

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
    */
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
    //freopen("1.out","w",stdout);

    for(auto size: scale) {
        cout << "Running, dataset: size " << size << endl;
        Benchmark(size);
        cout << "Passed, dataset: size " << size << endl;
        cout << endl;
    }
    return 0;
}