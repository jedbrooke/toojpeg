using uint8_t  = unsigned char;
using uint16_t = unsigned short;
using  int16_t =          short;
using  int32_t =          int; // at least four bytes
using uint32_t = unsigned int;

namespace gpu
{

    /* 
		data is an aray of n 8*8 blocks
		scale is an array of 8*8
		posNonZero will store the position of the last non zero value for each N blocks
		n is the number of blocks
	*/
	void transformBlock_many(float* const data, const float* const scale, float* const posNonZero, const uint32_t n);
    
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
	int convertBWtoY(uint8_t* data, const int width, const int height);
}