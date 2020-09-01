#include "utility.h"

namespace utility{
    std::ostream* out; 
    void set_stream(std::ostream* s)
    {
        utility::out = s;
    }
    
    void print_array(int n, float* a)
    {
        for (int i = 0; i < n - 1; i++)
        {
            *utility::out << a[i] << ", ";
        }
        *utility::out << a[n-1];
    }

    void print_array(int n, int16_t* a)
    {
        for (int i = 0; i < n - 1; i++)
        {
            *utility::out << a[i] << ", ";
        }
        *utility::out << a[n-1];
    }

    void print_8x8(float A[8][8])
    {
        *utility::out << "{";
        for (int i = 0; i < 8; i++)
        { 
            std:: cout << "{";
            for (int j = 0; j < 7; j++)
            {
                *utility::out << A[i][j] << ", ";
            }
            *utility::out << A[i][7] << "}," << std::endl;
        }
    }


    void print_array_2d(int m, int n, float** a)
    {
        for (int i = 0; i < m; i++)
        {
            *utility::out << "[";
            print_array(n,a[i]);
            *utility::out << "]\n";
        }
    }

    void print_array_2d(int m, int n, float* a)
    {
        for (int i = 0; i < m; i++)
        {
            *utility::out << "[";
            print_array(n,&a[i*n]);
            *utility::out << "]\n";
        }
    }

    void print_array_3d(int x, int y, int z, float* a)
    {
        for (int i = 0; i < x; i++){
            *utility::out << "[";
            print_array_2d(y,z,&a[i*y*z]);
            *utility::out << "]\n";
        }
    }


    void flatten_array(float** data, float* data_flat, int m, int n)
    {
        for (int i = 0; i < m; i++)
        {
            memcpy(&data_flat[i*n],data[i],n*sizeof(float));
        }
    }

    float max_error(float* A, float* B, int n)
    {
        float* err = (float*) malloc(n*sizeof(float));
        for (int i = 0; i < n; i++)
        {
            err[i] = A[i] - B[i];
        }
        float ret = *std::max_element(err, err+n, [](float a, float b) {return abs(a) < abs(b);});
        free(err);
        return ret;
    }

    int err_count(float* A, float* B, int n, float thresh)
    {
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if(abs(A[i] - B[i]) > thresh) count++;
        }
        return count;
    }

    void load_data(std::string path, float** data, int numLines)
    {
        std::string val;
        std::string line;
        std::ifstream dataFile(path);

        if(dataFile.is_open()){
            for (int lineIdx = 0; lineIdx < numLines; lineIdx++)
            {
                std::getline(dataFile, line);
                std::stringstream lineStream(line);
                for(int colIdx = 0; colIdx < 8*8; colIdx++)
                {
                    std::getline(lineStream,val,',');
                    data[lineIdx][colIdx] = atof(val.c_str());
                }
            }
        } else {
            std::cerr << "Error! could not read input file! make sure '" << path << "' is the correct path" << std::endl;
            exit(1); 
        }   
    }

    float abs_max_element(float* err, int n)
    {
        return *std::max_element(err, err+n,[](float a, float b) {return abs(a) < abs(b);});
    }

    void identity_8x8(float* I)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                I[i*8 + j] = (i==j);
            }
        }
    }
}