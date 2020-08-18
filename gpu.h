#include <thread>
#include <stdlib.h>
#include "constants.h"

namespace gpu
{
    /* 
		loads DCT tranform matricies and other constants in to device memory
	*/
	void initializeDevice();
	/* 
		frees device memory
	*/
	void retireDevice();
	/* 
		converts rgb to YCbCr444 and reshapes data from pixels into array of 8x8 blocks
        data is (width * height) * 3
		data[i] = r, data[i + 1] = g, data[i + 2] = b
		Y is stored as (width/8 * height/8) * (8x8 Y block)
		etc for Cb, Cr
	*/
    int convertRGBtoYCbCr444(uint8_t* data, const int width, const int height, float* Y, float* Cb, float* Cr);

    /* 
		Y is stored as (width/8 * height/8) * (8x8 Y block)
		Cb/Cr is stored as (width/16 * height/16) * (1 Cb 8x8 block / 1 Cr 8x8 block)
	*/
	int convertRGBtoYCbCr420(uint8_t* data, const int width, const int height, float* Y, float* Cb, float* Cr);

    /* 
		data is (width * height) BW pixel values, 
		Y is returned in data as n 8x8 blocks
	*/
	int convertBWtoY(uint8_t* data, const int width, const int height, float* Y);

    /* 
		data is an aray of n 8*8 blocks
		scale is an array of 8*8
		posNonZero will store the position of the last non zero value for each N blocks
		n is the number of blocks
		quantized is the quantized result of the transformation
	*/
	void transformBlock_many(float* const data, const float* const scale, const uint32_t n, uint8_t* posNonZero, int16_t* quantized);
}