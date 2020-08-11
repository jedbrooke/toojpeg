#include "gpu.h"

namespace // anonymous namespace for helper functions
{ 
    /* 
        data is n 8x8 blocks of data to be transformed
        scale is a 8x8 block for the elementwise scale at the end
        output is returned in data
    */
    void DCT8x8_many(float* data, const int n, float* scale);

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
		data is an aray of n 8*8 blocks
		scale is an array of 8*8
		posNonZero will store the position of the last non zero value for each N blocks
		n is the number of blocks
	*/
	// might throw this whole function into gpu.cu so that we can use CUDA memory more efficiently by keeping everything in there insteading of passing it arounf between functions
	void transformBlock_many(float* const data, const float* const scale, float* const posNonZero, const uint32_t n)
	{
		// DCT
		// Scale (remove scale step from DCT and combine the scale matrix there with the one here so it's only 1 step instead of 2)
		// quantize (process many blocks at a time with paralell inside each block too)
		// find pos non zero (paralell many blocks but serial inside block)
			// start counting from back and stop at first non-zero value, can skip most of the block then
		 
	}

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
}