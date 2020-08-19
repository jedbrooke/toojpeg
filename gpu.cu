#include "gpu.h"

namespace // anonymous namespace for helper functions
{ 
    bool isTransformConstsLoaded = false;
    float* dct_matrix_cuda;
    float* dct_matrix_transpose_cuda;
    uint8_t* ZigZagInv_cuda;
    
    void loadTransformConstants() // could maybe return the cuda error value if you want to do some quick exception handling
    {   
        cudaMalloc(&dct_matrix_cuda,           constants::block_size_mem);
        cudaMalloc(&dct_matrix_transpose_cuda, constants::block_size_mem);
        cudaMalloc(&ZigZagInv_cuda,            constants::block_size_mem);
        
        cudaMemcpy(dct_matrix_cuda,           constants::dct_matrix,           constants::block_size_mem,               cudaMemcpyHostToDevice);
        cudaMemcpy(dct_matrix_transpose_cuda, constants::dct_matrix_transpose, constants::block_size_mem,               cudaMemcpyHostToDevice);
        cudaMemcpy(ZigZagInv_cuda,            constants::ZigZagInv,            constants::block_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        isTransformConstsLoaded = true;
    }

    void unloadTransformConstants()
    {
        cudaFree(dct_matrix_cuda);
        cudaFree(dct_matrix_transpose_cuda);
        cudaFree(ZigZagInv_cuda);
    }

    std::string last_msg("");
    int repeat_count = 0;
    void debug_log(std::string s)
	{
		if (GPU_DEBUG)
		{
            if(last_msg.compare(s) == 0) { // if the current string is the same at the last one
                repeat_count += 1;
            } else {
                if(repeat_count > 0)
                {
                    std::cout << last_msg << "(" << repeat_count << ")" << std::endl;
                    repeat_count = 0;
                }
                std::cout << s << std::endl;
                last_msg = std::string(s.c_str());                
            }
            
		}
	}

    __global__
    void elementwise_mult_8x8_multi(const float* K, float* A, const uint32_t n)
    {   
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int c = index; c < n * 8*8; c+= stride)
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
    void DCT8x8_many(float* data, const uint32_t n, const float* scale)
    {
        float* temp;
        cudaMalloc(&temp,n * constants::block_size_mem);
        cudaMemset(temp,0,n * constants::block_size_mem);

        matmul_8x8_multi_step1<<<n,constants::block_size>>>(constants::dct_matrix,data,temp,n);

        cudaDeviceSynchronize();
        cudaMemset(data,0,n * constants::block_size_mem);

        matmul_8x8_multi_step2<<<n,constants::block_size>>>(temp,constants::dct_matrix_transpose,data,n);

        cudaDeviceSynchronize();

        elementwise_mult_8x8_multi<<<n,constants::block_size>>>(scale,data,n);

        cudaDeviceSynchronize();
        
        cudaFree(temp);

    }

    /* 
        data is n 8x8 blocks of data to be quantized
        quantized is n 8x8 blocks of ints returned
        this needs to be zigzagged
    */
    __global__
    void quantize_many(const float* data, const int n, uint8_t* ZigZagInv, int16_t* quantized)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int i = index; i < n * constants::block_size; i+= stride)
        {
            quantized[i] = __float2int_rn(data[(i / constants::block_size) + ZigZagInv[i % constants::block_size]]);
        }
    }

    /* 
        quantized is n 8x8 blocks
        posNonZeros is the return value,
        the pos of the last non zero value for each of the 8x8 blocks
        it's a uint8 since the number can only be [0...63] (5bits)

    */
    __global__
    void find_posNonZero_many(const int16_t* quantized, const uint32_t n, uint8_t* posNonZero)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(auto c = index; c < n; c+= stride)
        {
            const int16_t* block = &quantized[c * constants::block_size];
            // for loops are evil >:(
            for(auto i = constants::block_size - 1; i >= 0; i--) // go bakcwards until we find a non-zero, then we can end the loop
            {
                if (block[i] == 0)
                {
                    posNonZero[c] = i;
                    break;
                }
            }
        }
    }

    __global__ 
    void convertRGBtoY(const uint8_t* pixels, const int n, const int width, const int height, float* Y)
    {
        const auto index = (blockIdx.x * blockDim.x + threadIdx.x);
        const auto stride = blockDim.x * gridDim.x; 
        const auto width_padded  = (width  + (width  % 8 == 0 ? 0 : 8 - (width  % 8))); 
        const auto height_padded = (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
        for (auto i = index; i < n; i += stride)
        {
            // find x and y, if x >= width x=width - 1, if y >= height y = height - 1
            const auto x = (i % width_padded)  >= width  ? width  - 1 : (i % width_padded );
            const auto y = (i / height_padded) >= height ? height - 1 : (i / height_padded);
            const auto pos = (y*width + x) * 3; // mult 3 for 3 color components r,g,b
            Y[i] = (+0.299f * pixels[pos + 0] + 0.587f * pixels[pos + 1] + 0.114f * pixels[pos + 2]) - 128.f;
        } 
    }

    __global__ 
    void convertRGBtoCb(const uint8_t* pixels, const int n, const int width, const int height, float* Cb)
    {
        const auto index = (blockIdx.x * blockDim.x + threadIdx.x);
        const auto stride = blockDim.x * gridDim.x;
        const auto width_padded  = (width  + (width  % 8 == 0 ? 0 : 8 - (width  % 8))); 
        const auto height_padded = (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
        for (auto i = index; i < n; i += stride)
        {
            // find x and y, if x >= width x=width - 1, if y >= height y = height - 1
            const auto x = (i % width_padded)  >= width  ? width  - 1 : (i % width_padded );
            const auto y = (i / height_padded) >= height ? height - 1 : (i / height_padded);
            const auto pos = (y*width + x) * 3; // mult 3 for 3 color components r,g,b
            Cb[i] = -0.16874f * pixels[pos + 0] - 0.33126f * pixels[pos + 1] + 0.5f * pixels[pos + 2]; 
        } 
    }

    __global__ 
    void convertRGBtoCr(const uint8_t* pixels, const int n, const int width, const int height, float* Cr)
    {
        const auto index = (blockIdx.x * blockDim.x + threadIdx.x);
        const auto stride = blockDim.x * gridDim.x * 3;
        const auto width_padded  = (width  + (width  % 8 == 0 ? 0 : 8 - (width  % 8))); 
        const auto height_padded = (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
        for(auto i = index; i < n; i += stride)
        {
            // find x and y, if x >= width x=width - 1, if y >= height y = height - 1
            const auto x = (i % width_padded)  >= width  ? width  - 1 : (i % width_padded );
            const auto y = (i / height_padded) >= height ? height - 1 : (i / height_padded);
            const auto pos = (y*width + x) * 3; // mult 3 for 3 color components r,g,b
            Cr[i] = +0.5f * pixels[pos + 0] - 0.41869f * pixels[pos + 1] +0.5f * pixels[pos + 2];
        } 
    }

    __global__
    void reshape_data_to_blocks(float* data, const int width, const int height)
    {
        // process image 8 pixel rows at a time across all columns
        // memcpy chuck of 8 rows into temp array
        // async memcpy it back into source pointer
        const uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x);
        const uint32_t stride = blockDim.x * gridDim.x; // copy 8 rows at a time
        const uint32_t strip_size = width * 8 * sizeof(float);
        float* temp = new float[strip_size];
        for(uint32_t i = index; i < height / 8; i += stride) // for all the rows of blocks
        {
            float* strip_ptr = &data[i * 8 * width]; // constant pointer to the current strip
            memcpy(temp,strip_ptr,strip_size);
            // for loops are evil >:(
            for(uint32_t b = 0; b < width / 8; b++) // for each 8x8 block in the strip
            {
                // double for loop! what are you trying to do to me man
                for(uint8_t l = 0; l < 8; l++) // for each line in that block
                {
                    memcpy(&strip_ptr[b * constants::block_size + l * 8],&temp[l*8 + b * 8], 8 * sizeof(float));
                }
            }
        }
        delete[] temp;
    }

    void launchConversionKernel(const uint8_t* pixels, const int n, const int width, const int height, constants::conversions conversion, float* output)
    {
        // convert the color data
        auto const cu_blockSize = 256;
        auto const cu_numBlocks = (n / cu_blockSize) + 1;
        switch(conversion)
        {
            case constants::YConv:
                convertRGBtoY<<<cu_numBlocks,cu_blockSize>>>(pixels,n,width,height,output);
                break;
            case constants::CbConv:
                convertRGBtoCb<<<cu_numBlocks,cu_blockSize>>>(pixels,n,width,height,output);
                break;
            case constants::CrConv:
                convertRGBtoCr<<<cu_numBlocks,cu_blockSize>>>(pixels,n,width,height,output);
        }
        const auto width_padded  = (width  + (width  % 8 == 0 ? 0 : 8 - (width  % 8))); 
        const auto height_padded = (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
        cudaStreamSynchronize(0);
        // reshape the data in to 8x8 blocks
        reshape_data_to_blocks<<<(height / 64) + 1,8>>>(output,width_padded,height_padded);
        cudaStreamSynchronize(0);
    }


}

namespace gpu
{
    
    void initializeDevice()
    {
        loadTransformConstants();
    }
    void retireDevice()
    {
        debug_log(" ");
        unloadTransformConstants();
        isTransformConstsLoaded = false;
    }
    /* 
		data is (width * height) * 3
		data[i] = r, data[i + 1] = g, data[i + 2] = b
		Y is stored as (width/8 * height/8) * (8x8 Y block)
		etc for Cb, Cr
	*/
	int convertRGBtoYCbCr444(const uint8_t* data, const int width, const int height, float* Y, float* Cb, float* Cr)
	{
        // prepare memory
        uint8_t* pixels_cuda;
        float* Y_cuda;
        float* Cb_cuda;
        float* Cr_cuda;

        // n = total num items (width*height) * 3
        const auto n_datas = width * height * 3;
        // width and height are each rounded up to nearest multiple of 8 to prepare for converting data to 8x8 blocks
        const auto n_padded = (width + (width % 8 == 0 ? 0 : 8 - (width % 8))) * (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));

        // allocate all the memory on the GPU
        cudaMalloc(&pixels_cuda, n_datas * sizeof(uint8_t));
        cudaMemcpy(pixels_cuda, data, n_datas * sizeof(uint8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&Y_cuda,  n_padded * sizeof(float));
        cudaMalloc(&Cb_cuda, n_padded * sizeof(float));
        cudaMalloc(&Cr_cuda, n_padded * sizeof(float));

        // Y = rgb2Y(data)
        // Y = Y - 128.f
		// Cb = rgb2Cb(data)
		// Cr = rgb2Cr(data)
        // cudaDeviceSynchronize
        
        // if multithreaded. If we are facing memory limitations then we may need to do each channel separately
        bool multithreaded = false;
        if(multithreaded) {
            std::thread* threads = new std::thread[3];
        
            threads[0] = std::thread(launchConversionKernel, pixels_cuda, n_padded, width, height, constants::YConv , Y_cuda );
            threads[1] = std::thread(launchConversionKernel, pixels_cuda, n_padded, width, height, constants::CbConv, Cb_cuda);
            threads[2] = std::thread(launchConversionKernel, pixels_cuda, n_padded, width, height, constants::CrConv, Cr_cuda);
            
            for(uint8_t i = 0; i < 3; i++)
            {
                threads[i].join();
            }
        } else {    // else just launch the kernels one by one
            launchConversionKernel(pixels_cuda, n_padded, width, height, constants::YConv , Y_cuda );
            launchConversionKernel(pixels_cuda, n_padded, width, height, constants::CbConv, Cb_cuda);
            launchConversionKernel(pixels_cuda, n_padded, width, height, constants::CrConv, Cr_cuda);
        }
        

        
        
        // // make sure each array has enough allocated, since image may grow due to padding to 8x8 blocks
        // Y  = (float*) realloc(Y,  n_padded * sizeof(float));
        // Cb = (float*) realloc(Cb, n_padded * sizeof(float));
        // Cr = (float*) realloc(Cr, n_padded * sizeof(float));
        // assume pointers come with enough size

        // copy data from cuda back to 
        cudaMemcpyAsync(Y,  Y_cuda,  n_padded * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(Cb, Cb_cuda, n_padded * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(Cr, Cr_cuda, n_padded * sizeof(float), cudaMemcpyDeviceToHost);

        // make sure everything is done
        cudaDeviceSynchronize();

        debug_log("rgb2ycbcr done, first Y: " + std::to_string(Y[0]));

        // free the memory
        cudaFree(pixels_cuda);
        cudaFree(Y_cuda);
        cudaFree(Cb_cuda);
        cudaFree(Cr_cuda);

        // n = number of 8x8 blocks in Y
        // return n
        return n_padded / 64; // n_padded = width*height, divide each by 8 to get num blocks
	}

	/* 
		Y is stored as (width/8 * height/8) * (8x8 Y block)
		Cb/Cr is stored as (width/16 * height/16) * (1 Cb 8x8 block / 1 Cr 8x8 block)
	*/
	int convertRGBtoYCbCr420(const uint8_t* data, const int width, const int height, float* Y, float* Cb, float* Cr)
	{
		// Y = rgb2Y(data)
			// Y = Y - 128.f, probably in the same kernel so we dont need a deviceSynchronize
		// downscale RGB to 1/4 size with averages
		// wait for the downscale kernel to finish
		// Cb = rgb2Cb(data)
		// Cr = rgb2Cr(data)
		// cudaDeviceSynchronize
        // n = number of 8*8 blocks in Y, length of Cb,Cr is 1/4 N
        return 0;
	}

	/* 
		data is (width * height) BW pixel values, 
		Y is returned in data as n 8x8 blocks
	*/
	int convertBWtoY(const uint8_t* data, const int width, const int height, float* Y)
	{
        // Y = pixel - 128.f but in CUDA
        // pad to 8x8 and reshape data
        // int n = number of 8x8 blocks 
        return 0;
    }
    
    /* 
		data is an aray of n 8*8 blocks
		scale is an array of 8*8
		posNonZero will store the position of the last non zero value for each N blocks
		n is the number of blocks
	*/
	void transformBlock_many(const float* data, const float* scale, const uint32_t n, uint8_t* posNonZero, int16_t* quantized)
	{
        // DCT and Scale
        if(!isTransformConstsLoaded)
        {
            loadTransformConstants();
        }
        float* data_cuda;
        cudaMalloc(&data_cuda, n*constants::block_size_mem);
        cudaMemcpy(data_cuda,data, n*constants::block_size_mem, cudaMemcpyHostToDevice);

        float* scale_cuda;
        cudaMalloc(&scale_cuda, constants::block_size_mem);
        cudaMemcpy(scale_cuda,scale,constants::block_size_mem, cudaMemcpyHostToDevice);
        
        DCT8x8_many(data_cuda, n, scale_cuda);

        // quantize (process many blocks at a time with paralell inside each block too)
        int16_t* quantized_cuda;
        cudaMalloc(&quantized_cuda,n * constants::block_size_mem);
        quantize_many<<<n,constants::block_size>>>(data_cuda,n,ZigZagInv_cuda,quantized_cuda);
        cudaDeviceSynchronize();

        cudaFree(data_cuda);
        cudaFree(scale_cuda);

		// find pos non zero (paralell many blocks but serial inside block)
            // start counting from back and stop at first non-zero value, can skip most of the block then
        uint8_t* posNonZero_cuda;
        cudaMalloc(&posNonZero_cuda, n * sizeof(uint8_t));

        int cu_blockSize = 256;
        int cu_numBlocks = (n / cu_blockSize) + 1;

        find_posNonZero_many<<<cu_numBlocks,cu_blockSize>>>(quantized_cuda, n, posNonZero_cuda);
        
        // // prepare the destination pointers
        // quantized  = (int16_t*) realloc(quantized, n * constants::block_size * sizeof(int16_t));
        // posNonZero = (uint8_t*) realloc(posNonZero, n * sizeof(uint8_t));
        cudaDeviceSynchronize();

        // copy data back from the device to the cpu
        cudaMemcpyAsync(quantized,  quantized_cuda,  n * constants::block_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(posNonZero, posNonZero_cuda, n * sizeof(uint8_t),                         cudaMemcpyDeviceToHost);
        
        cudaDeviceSynchronize();

        debug_log("data transformed, 1st value: " + std::to_string(quantized[0]));
        
        cudaFree(quantized_cuda);
        cudaFree(posNonZero_cuda);
	}
}