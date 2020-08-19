

#include "readbmp.h"


BMPImg::BMPImg(const std::string path)
{
    /* Header variables */
    uint8_t* header = new uint8_t[14]; // allocate 14 bytes to read in the header
    uint32_t offset;

    /* InfoHeader variables */
    uint8_t* infoHeader = new uint8_t[40]; // allocate 40 bytes to read in the infoHeader
    uint32_t compression;
    uint32_t imgSize;

    /* prepare file handle */
    std::ifstream bmp(path, std::ios::binary);    
    if(!bmp)
    {
        std::cerr << "Error: " << path << " could not be opened." << std::endl;
        throw "FileNotFound";  
    }

    /* 
        offsets and other values obtained from BMP file format guide.
        http://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm
    */

    /* read header */
    bmp.read((char*) header,14);
    memcpy(&offset,header + 0xA,4);

    /* read infoHeader */
    bmp.read((char*) infoHeader,40);
    memcpy(&width,      infoHeader + 4, 4);
    memcpy(&height,     infoHeader + 8, 4);
    memcpy(&bpp,        infoHeader + 14,2);
    memcpy(&compression,infoHeader + 16,4);

    if(compression) // for now we wont support compression
    {
        std::cerr << "Error: compressed bmp files are not supported at this time" << std::endl;
        throw "CompressionNotSupported";
    }

    /* 
        skip the color table
        we can come back to this later if we plan to deal with images that aren't 24bit RGB 
    */
    bmp.seekg(offset);

    if(bpp != 24)
    {
        std::cerr << "For now we are only dealing with 24bit RGB images" << std::endl;
        throw "UnsupportedBitDepth";
    }

    isRGB = true;

    imgSize = width * height * (isRGB ? 3 : 1);

    pixels = new uint8_t[imgSize];
    bmp.read((char*)pixels,imgSize);

    /* Convert pixels from B,G,R to R,G,B */
    for (int i = 0; i < imgSize - 2; i+=3)
    {
        std::swap(pixels[i + 0],pixels[i + 2]);
    }
    /* Convert scanlines from Bottom-to-top to Top-to-bottom */
    auto row_length = width * (isRGB ? 3 : 1);
    uint8_t* temp = new uint8_t[row_length];
    for (int low = 0, high = height - 1; low < high; low++, high--)
    {
        
        //copy low in to temp
        memcpy(temp,&pixels[low * row_length],row_length);
        //copy high in to low
        memcpy(&pixels[low * row_length],&pixels[high * row_length],row_length);
        //copy temp in to high
        memcpy(&pixels[high * row_length],temp,row_length);
    }
    // free(temp);
}


/* 
    Destructor!
*/
BMPImg::~BMPImg()
{
    delete[] pixels;
}