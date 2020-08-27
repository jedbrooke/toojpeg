#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#if __has_include(<algorithm>)
#include <algorithm>
#endif

namespace utility{
    std::ostream* out = &std::cout; 
    void set_stream(std::ostream* s);
    void print_array(int n, float* a);
    void print_8x8(float A[8][8]);
    void print_array_2d(int m, int n, float** a);
    void print_array_2d(int m, int n, float* a);
    void print_array_3d(int x, int y, int z, float* a);
    void flatten_array(float** data, float* data_flat, int m, int n);
    float max_error(float* A, float* B, int n);
    int err_count(float* A, float* B, int n, float thresh);
    void load_data(std::string path, float** data, int numLines);
    float abs_max_element(float* err, int n);
    void identity_8x8(float* I);
}