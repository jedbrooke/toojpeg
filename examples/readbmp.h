
#include <string.h>
#include <fstream>
#include <iostream>


using uint8_t   = unsigned char;  // 1 byte, useful for 8bit color
using uint16_t  = unsigned short; // 2 bytes, for some 2 byte values in the header
using int32_t   =           int;  // at least 4 bytes, useful for storing full rgb pixel value
using uint32_t  = unsigned int;   // at least 4 bytes, but unsigned

class BMPImg
{
private:
    
public:
    uint8_t* pixels;
    uint32_t width;
    uint32_t height;
    uint16_t bpp;
    bool isRGB;

    BMPImg(const std::string path);
    ~BMPImg();
    /* 
        reads in the bmp file specified at the path
        returns the number of pixel values read 
        this is 3*num pixels for RGB and just num pixels for b/w
    */
    unsigned long get_pixels(const char* path, unsigned char* pixels);
    

};
    