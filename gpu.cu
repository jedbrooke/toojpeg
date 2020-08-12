#include "gpu.h"

namespace // anonymous namespace for helper functions
{ 
    __global__
    void elementwise_mult_8x8_multi(const float* K, float* A)
    {   
        if (KERNEL_LOOP_DEBUG)
            printf("performing elementwise_mult");
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int c = index; c < N_DATA * 8*8; c+= stride)
        {
            A[c] = K[c % 64] * A[c];
        }
    }

    /* 
        perform step 1 of multiplication on the big block of all mats
    */
    __device__
    void matmul_8x8_multi_step1_helper(const float* A, const float* B, float* C, int i, int j, int n)
    {
        for (int k = 0; k < 8; k++){
            C[n*64 + i*8 + j] += A[i*8 + k] * B[n*64 + k*8 + j];
            if (KERNEL_LOOP_DEBUG)
                printf("C[%u][%u][%u] += A[%u][%u] * B[%u][%u][%u]\n",n,i,j,i,k,n,i,k);
        }
    }

    __global__
    void matmul_8x8_multi_step1(const float* dct_mat, const float* data, float* temp, const uint32_t n)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int c = index; c < n * constants::block_size; c+= stride)
        {
            matmul_8x8_multi_step1_helper(dct_mat,data,temp,(c % 64) / 8, c % 8, c / 64);
        }    
    }

    __device__
    void matmul_8x8_multi_step2_helper(const float* A, const float* B, float* C, const int i, const int j, const int n)
    {
        for (int k = 0; k < 8; k++)
        {
            C[n*64 + i*8 + j] += A[n*64 + i*8 + k] * B[k*8 + j];
        }
    }

    __global__
    void matmul_8x8_multi_step2(const float* temp, const float* dct_mat, float* data, const uint32_t n)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int c = index; c < n * constants::block_size; c+= stride)
        {
            matmul_8x8_multi_step2_helper(temp,dct_mat,data,(c % 64) / 8, c % 8, c / 64);
        }   
    }
    
    /* 
        data is n 8x8 blocks of data to be transformed
        scale is a 8x8 block for the elementwise scale at the end
        output is returned in data
    */
    void DCT8x8_many(float* data, const uint32_t n, float* scale)
    {
        float* temp;
        cudaMallocManaged(&temp,n * constants::block_size_mem);
        cudaMemset(temp,0,array_size);

        cudaMallocManaged(scale,constants::block_size_mem);

        matmul_8x8_multi_step1<<<n,constants::block_size>>>(dct_matrix,data,temp,n);

        cudaDeviceSynchronize();
        cudaMemset(data,0,n * constants::block_size_mem);

        matmul_8x8_multi_step2<<<n,constants::block_size>>>(temp,dct_matrix_transpose,data);

        cudaDeviceSynchronize();

        elementwise_mult_8x8_multi<<<n,constants::block_size>>>(scale,data);

        cudaDeviceSynchronize();
        
        cudaFree(temp);
        cudaFree(scale);

    }

    /* 
        data is n 8x8 blocks of data to be quantized
        quantized is n 8x8 blocks of ints returned
    */
    void quantize_many(const float* data, const int n, int16_t* quantized);

    /* 
        quantized is n 8x8 blocks
        posNonZeros is the return value,
        the pos of the last non zero value for each of the 8x8 blocks
        it's a uint8 since the number can only be [0...63] (5bits)

    */
    void find_posNonZero_many(const int16_t* quantized, const uint32_t n, uint8_t* posNonZeros);

}

namespace gpu
{
    /* 
		data is (width * height) * 3
		data[i] = r, data[i + 1] = g, data[i + 2] = b
		Y is stored as (width/8 * height/8) * (8x8 Y block)
		etc for Cb, Cr
	*/
	int convertRGBtoYCbCr444(uint8_t* data, const int width, const int height, float* Y, float* Cb, float* Cr)
	{
		// Y = rgb2Y(data)
			// Y = Y - 128.f, probably in the same kernel so we dont need a deviceSynchronize
		// Cb = rgb2Cb(data)
		// Cr = rgb2Cr(data)
		// cudaDeviceSynchronize
		// n = number of 8x8 blocks in Y
	}

	/* 
		Y is stored as (width/8 * height/8) * (8x8 Y block)
		Cb/Cr is stored as (width/16 * height/16) * (1 Cb 8x8 block / 1 Cr 8x8 block)
	*/
	int convertRGBtoYCbCr420(uint8_t* data, const int width, const int height, float* Y, float* Cb, float* Cr)
	{
		// Y = rgb2Y(data)
			// Y = Y - 128.f, probably in the same kernel so we dont need a deviceSynchronize
		// downscale RGB to 1/4 size with averages
		// wait for the downscale kernel to finish
		// Cb = rgb2Cb(data)
		// Cr = rgb2Cr(data)
		// cudaDeviceSynchronize
		// n = number of 8*8 blocks in Y, length of Cb,Cr is 1/4 N
	}

	/* 
		data is (width * height) BW pixel values, 
		Y is returned in data as n 8x8 blocks
	*/
	int convertBWtoY(uint8_t* data, const int width, const int height)
	{
		// Y = pixel - 128.f but in CUDA
		// int n = number of 8x8 blocks 
    }
    
    /* 
		data is an aray of n 8*8 blocks
		scale is an array of 8*8
		posNonZero will store the position of the last non zero value for each N blocks
		n is the number of blocks
	*/
	void transformBlock_many(float* const data, const float* const scale, float* const posNonZero, const uint32_t n)
	{
        // Prepare scale matrix

        // DCT and Scale
        cudaMallocManaged(&constants::dct_matrix,           constants::block_size_mem);
        cudaMallocManaged(&constants::dct_matrix_transpose, constants::block_size_mem);
        // quantize (process many blocks at a time with paralell inside each block too)
        
		// find pos non zero (paralell many blocks but serial inside block)
			// start counting from back and stop at first non-zero value, can skip most of the block then
		 
	}
}