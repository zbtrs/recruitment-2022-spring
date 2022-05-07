#include <random>
#include <fstream>
#include <string>

std::default_random_engine e;
std::uniform_int_distribution<int> u(-1000, 1000);
int a[2048][2048], b[2048][2048];

void generate(int size) {
    std::ofstream ofs;
    ofs.open(std::to_string(size) + "/a");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i][j] = u(e);
            ofs << a[i][j] << ' ';
        }
        ofs << std::endl;
    }
    ofs.close();
    ofs.open(std::to_string(size) + "/b");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            b[i][j] = u(e);
            ofs << b[i][j] << ' ';
        }
        ofs << std::endl;
    }
    ofs.close();
    ofs.open(std::to_string(size) + "/result");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int result = 0;
            for (int k = 0; k < size; ++k)
                result += a[i][k] * b[k][j];
            ofs << result << ' ';
        }
        ofs << std::endl;
    }
    ofs.close();
}

int main() {
    generate(256);
    generate(512);
    generate(1024);
    generate(2048);
}