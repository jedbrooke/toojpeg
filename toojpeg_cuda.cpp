// //////////////////////////////////////////////////////////
// toojpeg.cpp
// written by Stephan Brumme, 2018-2019
// see https://create.stephan-brumme.com/toojpeg/
//

#include "toojpeg_cuda.h"

// - the "official" specifications: https://www.w3.org/Graphics/JPEG/itu-t81.pdf and https://www.w3.org/Graphics/JPEG/jfif3.pdf
// - Wikipedia has a short description of the JFIF/JPEG file format: https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format
// - the popular STB Image library includes Jon's JPEG encoder as well: https://github.com/nothings/stb/blob/master/stb_image_write.h
// - the most readable JPEG book (from a developer's perspective) is Miano's "Compressed Image File Formats" (1999, ISBN 0-201-60443-4),
//   used copies are really cheap nowadays and include a CD with C++ sources as well (plus great format descriptions of GIF & PNG)
// - much more detailled is Mitchell/Pennebaker's "JPEG: Still Image Data Compression Standard" (1993, ISBN 0-442-01272-1)
//   which contains the official JPEG standard, too - fun fact: I bought a signed copy in a second-hand store without noticing

namespace // anonymous namespace to hide local functions / constants / etc.
{
	// ////////////////////////////////////////
	// data types
	using uint8_t  = unsigned char;
	using uint16_t = unsigned short;
	using  int16_t =          short;
	using  int32_t =          int; // at least four bytes

	// ////////////////////////////////////////
	// constants

	// static Huffman code tables from JPEG standard Annex K
	// - CodesPerBitsize tables define how many Huffman codes will have a certain bitsize (plus 1 because there nothing with zero bits),
	//   e.g. DcLuminanceCodesPerBitsize[2] = 5 because there are 5 Huffman codes being 2+1=3 bits long
	// - Values tables are a list of values ordered by their Huffman code bitsize,
	//   e.g. AcLuminanceValues => Huffman(0x01,0x02 and 0x03) will have 2 bits, Huffman(0x00) will have 3 bits, Huffman(0x04,0x11 and 0x05) will have 4 bits, ...

	// Huffman definitions for first DC/AC tables (luminance / Y channel)
	const uint8_t DcLuminanceCodesPerBitsize[16]   = { 0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0 };   // sum = 12
	const uint8_t DcLuminanceValues         [12]   = { 0,1,2,3,4,5,6,7,8,9,10,11 };         // => 12 codes
	const uint8_t AcLuminanceCodesPerBitsize[16]   = { 0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125 }; // sum = 162
	const uint8_t AcLuminanceValues        [162]   =                                        // => 162 codes
		{ 0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08, // 16*10+2 symbols because
		  0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28, // upper 4 bits can be 0..F
		  0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59, // while lower 4 bits can be 1..A
		  0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89, // plus two special codes 0x00 and 0xF0
		  0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6, // order of these symbols was determined empirically by JPEG committee
		  0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
		  0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA };
	// Huffman definitions for second DC/AC tables (chrominance / Cb and Cr channels)
	const uint8_t DcChrominanceCodesPerBitsize[16] = { 0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0 };   // sum = 12
	const uint8_t DcChrominanceValues         [12] = { 0,1,2,3,4,5,6,7,8,9,10,11 };         // => 12 codes (identical to DcLuminanceValues)
	const uint8_t AcChrominanceCodesPerBitsize[16] = { 0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119 }; // sum = 162
	const uint8_t AcChrominanceValues        [162] =                                        // => 162 codes
		{ 0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91, // same number of symbol, just different order
		  0xA1,0xB1,0xC1,0x09,0x23,0x33,0x52,0xF0,0x15,0x62,0x72,0xD1,0x0A,0x16,0x24,0x34,0xE1,0x25,0xF1,0x17,0x18,0x19,0x1A,0x26, // (which is more efficient for AC coding)
		  0x27,0x28,0x29,0x2A,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,
		  0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x82,0x83,0x84,0x85,0x86,0x87,
		  0x88,0x89,0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,
		  0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,
		  0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA };
	const int16_t CodeWordLimit = 2048; // +/-2^11, maximum value after DCT

	// ////////////////////////////////////////
	// structs

	// represent a single Huffman code
	struct BitCode
	{
		BitCode() = default; // undefined state, must be initialized at a later time
		BitCode(uint16_t code_, uint8_t numBits_)
		: code(code_), numBits(numBits_) {}
		uint16_t code;       // JPEG's Huffman codes are limited to 16 bits
		uint8_t  numBits;    // number of valid bits
	};

	// wrapper for bit output operations
	struct BitWriter
	{
		// user-supplied callback that writes/stores one byte
		TooJpeg_cuda::WRITE_ONE_BYTE output;
		// initialize writer
		explicit BitWriter(TooJpeg_cuda::WRITE_ONE_BYTE output_) : output(output_) {}

		// store the most recently encoded bits that are not written yet
		struct BitBuffer
		{
			int32_t data    = 0; // actually only at most 24 bits are used
			uint8_t numBits = 0; // number of valid bits (the right-most bits)
		} buffer;

		// write Huffman bits stored in BitCode, keep excess bits in BitBuffer
		BitWriter& operator<<(const BitCode& data)
		{
			// append the new bits to those bits leftover from previous call(s)
			buffer.numBits += data.numBits;
			buffer.data   <<= data.numBits;
			buffer.data    |= data.code;

			// write all "full" bytes
			while (buffer.numBits >= 8)
			{
				// extract highest 8 bits
				buffer.numBits -= 8;
				auto oneByte = uint8_t(buffer.data >> buffer.numBits);
				output(oneByte);

				if (oneByte == 0xFF) // 0xFF has a special meaning for JPEGs (it's a block marker)
				output(0);         // therefore pad a zero to indicate "nope, this one ain't a marker, it's just a coincidence"

				// note: I don't clear those written bits, therefore buffer.bits may contain garbage in the high bits
				//       if you really want to "clean up" (e.g. for debugging purposes) then uncomment the following line
				//buffer.bits &= (1 << buffer.numBits) - 1;
			}
			return *this;
		}

		// write all non-yet-written bits, fill gaps with 1s (that's a strange JPEG thing)
		void flush()
		{
			// at most seven set bits needed to "fill" the last byte: 0x7F = binary 0111 1111
			*this << BitCode(0x7F, 7); // I should set buffer.numBits = 0 but since there are no single bits written after flush() I can safely ignore it
		}

		// NOTE: all the following BitWriter functions IGNORE the BitBuffer and write straight to output !
		// write a single byte
		BitWriter& operator<<(uint8_t oneByte)
		{
			output(oneByte);
			return *this;
		}

		// write an array of bytes
		template <typename T, int Size>
		BitWriter& operator<<(T (&manyBytes)[Size])
		{
			for (auto c : manyBytes)
				output(c);
			return *this;
		}

		// start a new JFIF block
		void addMarker(uint8_t id, uint16_t length)
		{
			output(0xFF); output(id);     // ID, always preceded by 0xFF
			output(uint8_t(length >> 8)); // length of the block (big-endian, includes the 2 length bytes as well)
			output(uint8_t(length & 0xFF));
		}
	};

	struct HeaderTables
	{
		uint8_t quantLuminance[8*8];
		uint8_t quantChrominance[8*8];
		BitCode huffmanLuminanceDC[256];
		BitCode huffmanLuminanceAC[256];
		BitCode huffmanChrominanceDC[256];
		BitCode huffmanChrominanceAC[256];
		BitCode codewords[2 * CodeWordLimit];
	};

	class PixelData
	{
	public:
		int16_t* quantY;
		int16_t* quantCb;
		int16_t* quantCr;
 
		uint8_t* posNonZeroY;
		uint8_t* posNonZeroCb;
		uint8_t* posNonZeroCr;

		uint32_t num_blocks;

		PixelData(uint32_t _num_pixels);
		~PixelData();
	};
	
	void debug_log(std::string s);
	
	PixelData::PixelData(uint32_t _num_pixels)
	{
		num_blocks = _num_pixels / 64;
		quantY = new int16_t[_num_pixels];
		quantCb = new int16_t[_num_pixels];
		quantCr = new int16_t[_num_pixels];

		posNonZeroY = new uint8_t[num_blocks];
		posNonZeroCb = new uint8_t[num_blocks];
		posNonZeroCr = new uint8_t[num_blocks];

	}
	
	PixelData::~PixelData()
	{
		debug_log("deleting pixel data");
		delete[] quantY;
		delete[] quantCb;
		delete[] quantCr;
 
		delete[] posNonZeroY;
		delete[] posNonZeroCb;
		delete[] posNonZeroCr;
	}
	

	void debug_log(std::string s)
	{
		if (PRINT_DEBUG)
		{
			std::cout << std::string(s) << std::endl;
		}
	}
	
	// ////////////////////////////////////////
	// functions / templates

	// same as std::min()
	template <typename Number>
	Number minimum(Number value, Number maximum)
	{
		return value <= maximum ? value : maximum;
	}

	// restrict a value to the interval [minimum, maximum]
	template <typename Number, typename Limit>
	Number clamp(Number value, Limit minValue, Limit maxValue)
	{
		if (value <= minValue) return minValue; // never smaller than the minimum
		if (value >= maxValue) return maxValue; // never bigger  than the maximum
		return value;                           // value was inside interval, keep it
	}

	void generate_quantization_tables (const unsigned char quality_,
		uint8_t quantLuminance[8*8], uint8_t quantChrominance[8*8])
	{
		// quality level must be in 1 ... 100
		auto quality = clamp<uint16_t>(quality_, 1, 100);
		// convert to an internal JPEG quality factor, formula taken from libjpeg
		quality = quality < 50 ? 5000 / quality : 200 - quality * 2;

		/* Probably not worth paralellizing this step since it's only 64 loops */
		for (auto i = 0; i < 8*8; i++)
		{
			int luminance   = (constants::DefaultQuantLuminance  [constants::ZigZagInv[i]] * quality + 50) / 100;
			int chrominance = (constants::DefaultQuantChrominance[constants::ZigZagInv[i]] * quality + 50) / 100;

			// clamp to 1..255
			quantLuminance  [i] = clamp(luminance,   1, 255);
			quantChrominance[i] = clamp(chrominance, 1, 255);
		}
	}

	void generate_quantization_tables_scaled(const uint8_t quantLuminance  [8*8], const uint8_t quantChrominance[8*8],
									        float* LumScale, float* ChromScale)
	{
		// adjust quantization tables with AAN scaling factors to simplify DCT
		for (auto i = 0; i < 8*8; i++)
		{
			auto row    = constants::ZigZagInv[i] / 8; // same as ZigZagInv[i] >> 3
			auto column = constants::ZigZagInv[i] % 8; // same as ZigZagInv[i] &  7

			// scaling constants for AAN DCT algorithm: AanScaleFactors[0] = 1, AanScaleFactors[k=1..7] = cos(k*PI/16) * sqrt(2)
			static const float AanScaleFactors[8] = { 1, 1.387039845f, 1.306562965f, 1.175875602f, 1, 0.785694958f, 0.541196100f, 0.275899379f };
			auto factor = 1 / (AanScaleFactors[row] * AanScaleFactors[column] * 8);
			LumScale  [constants::ZigZagInv[i]] = factor / quantLuminance  [i];
			ChromScale[constants::ZigZagInv[i]] = factor / quantChrominance[i];
		}


	}
	/* 
		inputs are the image data, width, height, color information, and whether to downsample or not, and quantization tables for Y and Cb/Cr
		outputs are quantized data, and position of the last nonzero element in each block
	*/

	PixelData process_pixels(const uint8_t* data, const int width, const int height, const bool isRGB, const bool downsample, HeaderTables tables) 
	{
		// initialize gpu constants in VRAM
		debug_log("initializing GPU");
		debug_log("first pixel:");
		debug_log(std::to_string(data[0]) + "," + std::to_string(data[1]) + "," + std::to_string(data[2]));
		gpu::initializeDevice();
		// prepare DCT scale matrix while converting RGB to YCbCr
		float* LumScaleDCT   = new float[constants::block_size];
		float* ChromScaleDCT = new float[constants::block_size];
		std::thread prep_blocks([tables,LumScaleDCT,ChromScaleDCT]()
		{
			debug_log("preping scale blocks");
			float*  LumScale   = new float[constants::block_size];
			float*  ChromScale = new float[constants::block_size];
			generate_quantization_tables_scaled(tables.quantLuminance,tables.quantChrominance,LumScale,ChromScale);
			for(auto i = 0; i < constants::block_size; i++)
			{
				LumScaleDCT[i]   = LumScale[i]   * constants::dct_correction_matrix[i];
				ChromScaleDCT[i] = ChromScale[i] * constants::dct_correction_matrix[i];
			}
			delete[] LumScale;
			delete[] ChromScale;
		});
		// convert image data
		debug_log("converting RGB to YCbCr");
		const auto n_padded = (width + (width % 8 == 0 ? 0 : 8 - (width % 8))) * (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
		PixelData output(n_padded);
		
		float* Y  = new float[n_padded];
		float* Cb = new float[n_padded]; 
		float* Cr = new float[n_padded];
		if(isRGB) {	
			if(downsample) {
				debug_log("color, downsampled"); 
				output.num_blocks = gpu::convertRGBtoYCbCr420(data,width,height,Y,Cb,Cr);
			} else {
				debug_log("color, no downsampling"); 
				output.num_blocks = gpu::convertRGBtoYCbCr444(data,width,height,Y,Cb,Cr);
			}
		} else {
			debug_log("BW"); 
			output.num_blocks = gpu::convertBWtoY(data,width,height,Y);
			delete Cb;
			delete Cr;
		}

		debug_log("first Y: " + std::to_string(Y[0]));


		// wait for the prep blocks thread to finish, we will need the blocks ready for the DCT
		prep_blocks.join();
		debug_log("DCT transforming and quantizing Y");
		gpu::transformBlock_many(Y, LumScaleDCT, output.num_blocks, output.posNonZeroY, output.quantY);

		if(isRGB) { //if image is RGB process chroma too
			// possibly run in parallel? detect multiple gpus?
			debug_log("DCT transforming and quantizing Cb");
			gpu::transformBlock_many(Cb, ChromScaleDCT, output.num_blocks / (downsample ? 4 : 1), output.posNonZeroCb, output.quantCb);
			debug_log("DCT transforming and quantizing Cr");
			gpu::transformBlock_many(Cr, ChromScaleDCT, output.num_blocks / (downsample ? 4 : 1), output.posNonZeroCr, output.quantCr);
		}
		// blocks are all processed, we are now ready for writing
		
		// memory cleanup
		delete[] Y;
		if(isRGB) {
			delete[] Cb;
			delete[] Cr;
		}
		delete[] LumScaleDCT;
		delete[] ChromScaleDCT;
		gpu::retireDevice();

		return output;
	}

	/* 
		writes and huffman encodes the block
	*/
	int16_t writeBlock(BitWriter& writer, int16_t* block, int16_t lastDC,
		const BitCode huffmanDC[256], const BitCode huffmanAC[256], const BitCode* codewords, int posNonZero)
	{
		/* 
			Step 5: Begin HuffmanEncoding
			Paralellizability: none, each block depends on previous
			Status: not done
		*/
		utility::print_array(64,block);

		// debug_log("Checking diff");
		// same "average color" as previous block ?
		auto DC = block[0];
		auto diff = DC - lastDC;
		if (diff == 0)
			writer << huffmanDC[0x00];   // yes, write a special short symbol
		else
		{
			auto bits = codewords[diff]; // nope, encode the difference to previous block's average color
			writer << huffmanDC[bits.numBits] << bits;
		}

		/* 
			Step 6: Write the huffman encoded bits
			Paralellizability: none (file io, bytes must be written in order)
			Status: not done
		*/

		// encode ACs (quantized[1..63])
		// debug_log("encoding ACs");
		auto offset = 0; // upper 4 bits count the number of consecutive zeros
		for (auto i = 1; i <= posNonZero; i++) // quantized[0] was already written, skip all trailing zeros, too
		{
			// zeros are encoded in a special way
			while (block[i] == 0) // found another zero ?
			{
				offset    += 0x10; // add 1 to the upper 4 bits
				// split into blocks of at most 16 consecutive zeros
				if (offset > 0xF0) // remember, the counter is in the upper 4 bits, 0xF = 15
				{
				writer << huffmanAC[0xF0]; // 0xF0 is a special code for "16 zeros"
				offset = 0;
				}
				i++;
			}

			auto encoded = codewords[block[i]];
			// combine number of zeros with the number of bits of the next non-zero value
			writer << huffmanAC[offset + encoded.numBits] << encoded; // and the value itself
			offset = 0;
		}

		// send end-of-block code (0x00), only needed if there are trailing zeros
		// debug_log("encoding trailing zeros");
		if (posNonZero < 8*8 - 1) // = 63
			writer << huffmanAC[0x00];

		return DC;
	}



	void writeBlock_many(BitWriter& writer, HeaderTables tables, PixelData* data, bool isRGB, bool downsample)
	{
		
		// for block in data
			// compare to DC of last block
			// encode non-zeros in block
			// encode zeros in block
		auto chroma_i = 0;
		auto num_blocks_str = std::to_string(data->num_blocks);
		int16_t lastYDC  = 0;
		int16_t lastCbDC = 0;
		int16_t lastCrDC = 0;
		for(auto i = 0; i < data->num_blocks;)
		{
			std::string msg("Writing block ");
			msg += std::to_string(i+1);
			msg += " of ";
			msg += num_blocks_str;
			debug_log(msg);
		
			chroma_i = i;
			for(auto j = 0; j < (downsample ? 4 : 1); j++)
			{
				// encode Y
				// debug_log("encoding Y");
				// debug_log(std::string("First data in block: ") + std::to_string(data.quantY[i * constants::block_size]));
				lastYDC = writeBlock(writer,&(data->quantY[i * constants::block_size]),lastYDC,tables.huffmanLuminanceDC,tables.huffmanLuminanceAC,tables.codewords,data->posNonZeroY[i]);
				i++;
			}
			// encode Cb
			lastCbDC = writeBlock(writer,&(data->quantY[chroma_i * constants::block_size]),lastCbDC,tables.huffmanChrominanceDC,tables.huffmanChrominanceAC,tables.codewords,data->posNonZeroCb[chroma_i]);
			// encode Cr
			lastCrDC = writeBlock(writer,&(data->quantY[chroma_i * constants::block_size]),lastCrDC,tables.huffmanChrominanceDC,tables.huffmanChrominanceAC,tables.codewords,data->posNonZeroCr[chroma_i]);
		}
		debug_log("finished writing blocks");
		
	
	}


	// Jon's code includes the pre-generated Huffman codes
	// I don't like these "magic constants" and compute them on my own :-)
	void generateHuffmanTable(const uint8_t numCodes[16], const uint8_t* values, BitCode result[256])
	{
		// process all bitsizes 1 thru 16, no JPEG Huffman code is allowed to exceed 16 bits
		auto huffmanCode = 0;
		for (auto numBits = 1; numBits <= 16; numBits++)
		{
			// ... and each code of these bitsizes
			for (auto i = 0; i < numCodes[numBits - 1]; i++) // note: numCodes array starts at zero, but smallest bitsize is 1
				result[*values++] = BitCode(huffmanCode++, numBits);

			// next Huffman code needs to be one bit wider
			huffmanCode <<= 1;
		}
	}

	HeaderTables writeHeader(BitWriter bitWriter, unsigned short width, unsigned short height, bool isRGB, unsigned char quality_, bool downsample, const char* comment)
	{
		// number of components
		const auto numComponents = isRGB ? 3 : 1;
		// note: if there is just one component (=grayscale), then only luminance needs to be stored in the file
		//       thus everything related to chrominance need not to be written to the JPEG
		//       I still compute a few things, like quantization tables to avoid a complete code mess

		// grayscale images can't be downsampled (because there are no Cb + Cr channels)
		if (!isRGB)
			downsample = false;

		// ////////////////////////////////////////
		// JFIF headers
		const uint8_t HeaderJfif[2+2+16] =
			{ 0xFF,0xD8,         // SOI marker (start of image)
			  0xFF,0xE0,         // JFIF APP0 tag
			  0,16,              // length: 16 bytes (14 bytes payload + 2 bytes for this length field)
			  'J','F','I','F',0, // JFIF identifier, zero-terminated
			  1,1,               // JFIF version 1.1
			  0,                 // no density units specified
			  0,1,0,1,           // density: 1 pixel "per pixel" horizontally and vertically
			  0,0 };             // no thumbnail (size 0 x 0)
		bitWriter << HeaderJfif;

		// ////////////////////////////////////////
		// comment (optional)
		if (comment != nullptr)
		{
			// look for zero terminator
			auto length = 0; // = strlen(comment);
			while (comment[length] != 0)
				length++;

			// write COM marker
			bitWriter.addMarker(0xFE, 2+length); // block size is number of bytes (without zero terminator) + 2 bytes for this length field
			// ... and write the comment itself
			for (auto i = 0; i < length; i++)
				bitWriter << comment[i];
		}

		// ////////////////////////////////////////
		// adjust quantization tables to desired quality
		uint8_t quantLuminance[8*8];
		uint8_t quantChrominance[8*8];
		generate_quantization_tables(quality_,quantLuminance,quantChrominance);
		
		
		// write quantization tables
		bitWriter.addMarker(0xDB, 2 + (isRGB ? 2 : 1) * (1 + 8*8)); // length: 65 bytes per table + 2 bytes for this length field
																	// each table has 64 entries and is preceded by an ID byte

		bitWriter   << 0x00 << quantLuminance;   // first  quantization table
		if (isRGB)
			bitWriter << 0x01 << quantChrominance; // second quantization table, only relevant for color images

		// ////////////////////////////////////////
		// write image infos (SOF0 - start of frame)
		bitWriter.addMarker(0xC0, 2+6+3*numComponents); // length: 6 bytes general info + 3 per channel + 2 bytes for this length field

		// 8 bits per channel
		bitWriter << 0x08
		// image dimensions (big-endian)
				<< (height >> 8) << (height & 0xFF)
				<< (width  >> 8) << (width  & 0xFF);

		// sampling and quantization tables for each component
		bitWriter << numComponents;       // 1 component (grayscale, Y only) or 3 components (Y,Cb,Cr)
		for (auto id = 1; id <= numComponents; id++)
			bitWriter <<  id                // component ID (Y=1, Cb=2, Cr=3)
		// bitmasks for sampling: highest 4 bits: horizontal, lowest 4 bits: vertical
					  << (id == 1 && downsample ? 0x22 : 0x11) // 0x11 is default YCbCr 4:4:4 and 0x22 stands for YCbCr 4:2:0
					  << (id == 1 ? 0 : 1); // use quantization table 0 for Y, table 1 for Cb and Cr

		// ////////////////////////////////////////
		// Huffman tables
		// DHT marker - define Huffman tables
		bitWriter.addMarker(0xC4, isRGB ? (2+208+208) : (2+208));
								// 2 bytes for the length field, store chrominance only if needed
								//   1+16+12  for the DC luminance
								//   1+16+162 for the AC luminance   (208 = 1+16+12 + 1+16+162)
								//   1+16+12  for the DC chrominance
								//   1+16+162 for the AC chrominance (208 = 1+16+12 + 1+16+162, same as above)

		// store luminance's DC+AC Huffman table definitions
		bitWriter << 0x00 // highest 4 bits: 0 => DC, lowest 4 bits: 0 => Y (baseline)
				  << DcLuminanceCodesPerBitsize
				  << DcLuminanceValues;
		bitWriter << 0x10 // highest 4 bits: 1 => AC, lowest 4 bits: 0 => Y (baseline)
				  << AcLuminanceCodesPerBitsize
				  << AcLuminanceValues;

		// compute actual Huffman code tables (see Jon's code for precalculated tables)
		BitCode huffmanLuminanceDC[256];
		BitCode huffmanLuminanceAC[256];
		generateHuffmanTable(DcLuminanceCodesPerBitsize, DcLuminanceValues, huffmanLuminanceDC);
		generateHuffmanTable(AcLuminanceCodesPerBitsize, AcLuminanceValues, huffmanLuminanceAC);

		// chrominance is only relevant for color images
		BitCode huffmanChrominanceDC[256];
		BitCode huffmanChrominanceAC[256];
		if (isRGB)
		{
			// store luminance's DC+AC Huffman table definitions
			bitWriter << 0x01 // highest 4 bits: 0 => DC, lowest 4 bits: 1 => Cr,Cb (baseline)
						<< DcChrominanceCodesPerBitsize
						<< DcChrominanceValues;
			bitWriter << 0x11 // highest 4 bits: 1 => AC, lowest 4 bits: 1 => Cr,Cb (baseline)
						<< AcChrominanceCodesPerBitsize
						<< AcChrominanceValues;

			// compute actual Huffman code tables (see Jon's code for precalculated tables)
			generateHuffmanTable(DcChrominanceCodesPerBitsize, DcChrominanceValues, huffmanChrominanceDC);
			generateHuffmanTable(AcChrominanceCodesPerBitsize, AcChrominanceValues, huffmanChrominanceAC);
		}

		// ////////////////////////////////////////
		// start of scan (there is only a single scan for baseline JPEGs)
		bitWriter.addMarker(0xDA, 2+1+2*numComponents+3); // 2 bytes for the length field, 1 byte for number of components,
														// then 2 bytes for each component and 3 bytes for spectral selection

		// assign Huffman tables to each component
		bitWriter << numComponents;
		for (auto id = 1; id <= numComponents; id++)
			// highest 4 bits: DC Huffman table, lowest 4 bits: AC Huffman table
			bitWriter << id << (id == 1 ? 0x00 : 0x11); // Y: tables 0 for DC and AC; Cb + Cr: tables 1 for DC and AC

		// constant values for our baseline JPEGs (which have a single sequential scan)
		static const uint8_t Spectral[3] = { 0, 63, 0 }; // spectral selection: must be from 0 to 63; successive approximation must be 0
		bitWriter << Spectral;


		// ////////////////////////////////////////
		// precompute JPEG codewords for quantized DCT
		BitCode  codewordsArray[2 * CodeWordLimit];          // note: quantized[i] is found at codewordsArray[quantized[i] + CodeWordLimit]
		BitCode* codewords = &codewordsArray[CodeWordLimit]; // allow negative indices, so quantized[i] is at codewords[quantized[i]]
		uint8_t numBits = 1; // each codeword has at least one bit (value == 0 is undefined)
		int32_t mask    = 1; // mask is always 2^numBits - 1, initial value 2^1-1 = 2-1 = 1
		for (int16_t value = 1; value < CodeWordLimit; value++)
		{
			// numBits = position of highest set bit (ignoring the sign)
			// mask    = (2^numBits) - 1
			if (value > mask) // one more bit ?
			{
				numBits++;
				mask = (mask << 1) | 1; // append a set bit
			}
			codewords[-value] = BitCode(mask - value, numBits); // note that I use a negative index => codewords[-value] = codewordsArray[CodeWordLimit  value]
			codewords[+value] = BitCode(       value, numBits);
		}

		HeaderTables tables;

		memcpy(tables.quantLuminance,   quantLuminance,   8 * 8 * sizeof(uint8_t));
		memcpy(tables.quantChrominance, quantChrominance, 8 * 8 * sizeof(uint8_t));
		
		memcpy(tables.huffmanLuminanceDC, huffmanLuminanceDC, 256 * sizeof(BitCode));
		memcpy(tables.huffmanLuminanceAC, huffmanLuminanceAC, 256 * sizeof(BitCode));

		memcpy(tables.huffmanChrominanceDC, huffmanChrominanceDC, 256 * sizeof(BitCode));
		memcpy(tables.huffmanChrominanceAC, huffmanChrominanceAC, 256 * sizeof(BitCode));

		memcpy(tables.codewords, codewords, 2 * CodeWordLimit * sizeof(BitCode));

		return tables;

	}

} // end of anonymous namespace

// -------------------- externally visible code --------------------

namespace TooJpeg_cuda
{
	// the only exported function ...
	bool writeJpeg(WRITE_ONE_BYTE output, const void* pixels_, unsigned short width, unsigned short height,
					 bool isRGB, unsigned char quality_, bool downsample, const char* comment)
	{
		// reject invalid pointers
		if (output == nullptr || pixels_ == nullptr)
			return false;
		// check image format
		if (width == 0 || height == 0)
			return false;
		// wrapper for all output operations
		BitWriter bitWriter(output);

		HeaderTables tables = writeHeader(bitWriter, width, height, isRGB, quality_, downsample, comment);

		// just convert image data from void*
		auto pixels = (const uint8_t*)pixels_;

		// the next two variables are frequently used when checking for image borders
		const auto maxWidth  = width  - 1; // "last row"
		const auto maxHeight = height - 1; // "bottom line"

		// process MCUs (minimum codes units) => image is subdivided into a grid of 8x8 or 16x16 tiles
		const auto sampling = downsample ? 2 : 1; // 1x1 or 2x2 sampling


		/* 
			steps taken in the loop:
			Step 1: convert rgb into YCbCr
			Parelellizability: strong (loops)

			Step 2: Encode Y
			Paralellizability: medium
				if we break it up into DCT, scaling, then writing, 
				we can DCT and scale all the Y block in GPU then
				finish the writing on the CPU
			
			Step 3: Perform Downsampling (is applicable)
			Paralellizability: strong (loops)
				can move up into first step of converting rgb to YCbCr

			Step 4: Encode Cb and Cr
			Paralellizability: medium (see step 3)

		*/
		debug_log("running GPU code");
		
		PixelData data = process_pixels(pixels,width,height,isRGB,downsample,tables);
		debug_log("writing data");
		writeBlock_many(bitWriter,tables,&data,isRGB,downsample);

		debug_log("data written, flushing buffers");


		bitWriter.flush(); // now image is completely encoded, write any bits still left in the buffer

		// ///////////////////////////
		// EOI marker
		bitWriter << 0xFF << 0xD9; // this marker has no length, therefore I can't use addMarker()
		return true;
	} // writeJpeg()
} // namespace TooJpeg
