#include "readbmp.h"
#include "../toojpeg.h"
#include <iostream>

// output file
std::ofstream myFile("earth.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput(unsigned char byte)
{
  myFile << byte;
}

int main(int argc, char const *argv[])
{
    BMPImg image("tulip.bmp");


    const auto quality    = 90;    // compression quality: 0 = worst, 100 = best, 80 to 90 are most often used
    const bool downsample = false; // false = save as YCbCr444 JPEG (better quality), true = YCbCr420 (smaller file)
    const char* comment = "TooJpeg example image"; // arbitrary JPEG comment
    auto ok = TooJpeg::writeJpeg(myOutput, image.pixels, image.width, image.height, image.isRGB, quality, downsample, comment);

    return 0;
}
