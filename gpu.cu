#include "gpu.h"

namespace // anonymous namespace for helper functions
{ 
    bool isTransformConstsLoaded = false;
    float* dct_matrix_cuda;
    float* dct_matrix_transpose_cuda;
    float* ZigZagInv_cuda;
    
    void loadTransformConstants() // could maybe return the cuda error value if you want to do some quick exception handling
    {   
        cudaMalloc(&dct_matrix_cuda,           constants::block_size_mem);
        cudaMalloc(&dct_matrix_transpose_cuda, constants::block_size_mem);
        cudaMalloc(&ZigZagInv_cuda,            constants::block_size_mem);
        
        cudaMemcpy(dct_matrix_cuda,           constants::dct_matrix,           constants::block_size_mem, cudaMemcpyHostToDevice);
        cudaMemcpy(dct_matrix_transpose_cuda, constants::dct_matrix_transpose, constants::block_size_mem, cudaMemcpyHostToDevice);
        cudaMemcpy(ZigZagInv_cuda,            constants::ZigZagInv,            constants::block_size_mem, cudaMemcpyHostToDevice);
        isTransformConstsLoaded = true;
    }

    void unloadTransformConstants()
    {
        cudaFree(dct_matrix_cuda);
        cudaFree(dct_matrix_transpose_cuda);
        cudaFree(ZigZagInv_cuda);
    }

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
        cudaMalloc(&temp,n * constants::block_size_mem);
        cudaMemset(temp,0,array_size);

        matmul_8x8_multi_step1<<<n,constants::block_size>>>(dct_matrix,data,temp,n);

        cudaDeviceSynchronize();
        cudaMemset(data,0,n * constants::block_size_mem);

        matmul_8x8_multi_step2<<<n,constants::block_size>>>(temp,dct_matrix_transpose,data);

        cudaDeviceSynchronize();

        elementwise_mult_8x8_multi<<<n,constants::block_size>>>(scale,data);

        cudaDeviceSynchronize();
        
        cudaFree(temp);

    }

    /* 
        data is n 8x8 blocks of data to be quantized
        quantized is n 8x8 blocks of ints returned
        this needs to be zigzagged
    */
    __global__
    void quantize_many(const float* data, const int n, int16_t* quantized)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int i = index; i < n * constants::block_size; i+= stride)
        {
            quantized[i] = __float2int_rn(data[ZigZagInv_cuda[i]]);
        }
    }

    /* 
        quantized is n 8x8 blocks
        posNonZeros is the return value,
        the pos of the last non zero value for each of the 8x8 blocks
        it's a uint8 since the number can only be [0...63] (5bits)

    */
    __global__
    void find_posNonZero_many(const int16_t* quantized, const uint32_t n, uint8_t* posNonZeros)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(auto c = index; c < n; c+= stride)
        {
            int16_t* block = &quantized[c * constants::block_size];
            // for loops are evil >:(
            for(auto i = constants::block_size - 1; i >= 0; i--) // go bakcwards until we find a non-zero, then we can end the loop
            {
                if (block[i] == 0)
                {
                    posNonZeros[c] = i;
                    break;
                }
            }
        }
    }

    __global__ 
    void convertRGBtoY(const uint8_t* pixels, const int n, const int width, const int height, float* Y)
    {
        auto const index = blockIdx.x * blockDim.x + threadIdx.x;
        auto const stride = blockDim.x * gridDim.x * 3; // 3 color components r,g,b
        const auto width_padded  = (width  + (width  % 8 == 0 ? 0 : 8 - (width  % 8)))); 
        const auto height_padded = (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
        for(auto i = index; i < n; i += stride)
        {
            // find x and y, if x >= width x=width - 1, if y >= height y = height - 1
            const auto x = (i % width_padded)  >= width  ? width  - 1 : (i % width_padded );
            const auto y = (i / height_padded) >= height ? height - 1 : (i / height_padded);
            const auto pos = y*width + x;
            Y[i] = +0.299f * pixels[pos + 0] + 0.587f * pixels[pos + 1] + 0.114f * pixels[pos + 2];
            Y[i] =- 128.f;
        } 
    }

    __global__ 
    void convertRGBtoCb(const uint8_t* pixels, const int n, const int width, const int height, float* Cb)
    {
        auto const index = blockIdx.x * blockDim.x + threadIdx.x;
        auto const stride = blockDim.x * gridDim.x * 3; // 3 color components r,g,b
        const auto width_padded  = (width  + (width  % 8 == 0 ? 0 : 8 - (width  % 8)))); 
        const auto height_padded = (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
        for(auto i = index; i < n; i += stride)
        {
            // find x and y, if x >= width x=width - 1, if y >= height y = height - 1
            const auto x = (i % width_padded)  >= width  ? width  - 1 : (i % width_padded );
            const auto y = (i / height_padded) >= height ? height - 1 : (i / height_padded);
            const auto pos = y*width + x;
            Cb[i] = -0.16874f * pixels[pos + 0] - 0.33126f * pixels[pos + 1] + 0.5f * pixels[pos + 2]; 
        } 
    }

    __global__ 
    void convertRGBtoCr(const uint8_t* pixels, const int n, const int width, const int height, float* Cr)
    {
        auto const index = blockIdx.x * blockDim.x + threadIdx.x;
        auto const stride = blockDim.x * gridDim.x * 3; // 3 color components r,g,b
        const auto width_padded  = (width  + (width  % 8 == 0 ? 0 : 8 - (width  % 8)))); 
        const auto height_padded = (height + (height % 8 == 0 ? 0 : 8 - (height % 8)));
        for(auto i = index; i < n; i += stride)
        {
            // find x and y, if x >= width x=width - 1, if y >= height y = height - 1
            const auto x = (i % width_padded)  >= width  ? width  - 1 : (i % width_padded );
            const auto y = (i / height_padded) >= height ? height - 1 : (i / height_padded);
            const auto pos = y*width + x;
            Cr[i] = +0.5f * pixels[pos + 0] - 0.41869f * pixels[pos + 1] +0.5f * pixels[pos + 2];
        } 
    }

    __global__
    void reshape_data_to_blocks(float* data, const int width, const int height)
    {
        // process image 8 pixel rows at a time across all columns
        // memcpy chuck of 8 rows into temp array
        // async memcpy it back into source pointer
        auto const index = blockIdx.x * blockDim.x + threadIdx.x;
        auto const stride = blockDim.x * gridDim.x * 8; // copy 8 rows at a time
        const auto strip_size = width * 8 * sizeof(float);
        float* temp;
        cudaMalloc(temp,strip_size); // allocate enough for 8 rows
        for(auto i = index; i < height; i += stride) // for all the rows
        {
            float* const strip_ptr = &data[i * width]; // constant pointer to the current strip
            cudaMemcpy(temp,strip_ptr,strip_size,cudaMemcpyDeviceToDevice);
            // for loops are evil >:(, but this one is with asyncs so I guess it's ok
            for(auto b = 0; b < width / 8, b++) // for each 8x8 block in the strip
            {
                // double for loop! what are you trying to do to me man
                for(auto l = 0; l < 8; l++) // for each line in that block
                {
                    cudaMemcpyAsync(&strip_ptr[b * constants::block_size + l],temp[l*8 + b * 8], 8 * sizeof(float), cudaMemcpyDeviceToDevice);
                }
            }
        }
        cudaFree(temp);
    }

    void launchConversionKernel(const uint8_t* pixels, const int n, const int width, const int height, auto conversion, float* output)
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
        
        cudaStreamSynchronize(0);
        // reshape the data in to 8x8 blocks
        reshape_data_to_blocks<<<cu_numBlocks,cu_blockSize>>>(output,width,height);
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
        unloadTransformConstants();
        isTransformConstsLoaded = false;
    }
    /* 
		data is (width * height) * 3
		data[i] = r, data[i + 1] = g, data[i + 2] = b
		Y is stored as (width/8 * height/8) * (8x8 Y block)
		etc for Cb, Cr
	*/
	int convertRGBtoYCbCr444(uint8_t* data, const int width, const int height, float* Y, float* Cb, float* Cr)
	{
        // prepare memory
        uint8_t* pixels_cuda;
        float* Y_cuda;
        float* Cb_cuda;
        float* Cr_cuda;


        const auto n_datas = width * height * 3;
        // n = total num items (width*height) * 3
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
        std::thread Y_thread (launchConversionKernel, pixels_cuda, n_padded, width, height, constants::Yonv  , Y_cuda );
        std::thread Cb_thread(launchConversionKernel, pixels_cuda, n_padded, width, height, constants::CbConv, Cb_cuda);
        std::thread Cr_thread(launchConversionKernel, pixels_cuda, n_padded, width, height, constants::CrConv, Cr_cuda);
        std::thread threads[3] = {Y_thread,Cb_thread,Cr_thread};
        for(auto i = 0; i < 3; i++)
            threads[i].join();
        // else just launch the kernels one by one
        
        // copy data from cuda back to 
        cudaMemcpyAsync(Y,  Y_cuda,  n_padded * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(Cb, Cb_cuda, n_padded * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(Cr, Cr_cuda, n_padded * sizeof(float), cudaMemcpyDeviceToHost);

        // make sure everything is done
        cudaDeviceSynchronize();

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
	void transformBlock_many(float* const data, const float* const scale, const uint32_t n, uint8_t* const posNonZero, int16_t* quantized)
	{
        // Prepare scale matrix
            // elementwise mult the given scale matrix with the dct correction matrix
        float* new_scale;
        cudaMallocManaged(&new_scale,constants::block_size_mem);
        for(int i = 0; i < constants::block_size; i++)
            new_scale[i] = constants::dct_correction_matrix[i] * scale[i];

        // DCT and Scale
        if(!isTransformConstsLoaded)
        {
            loadTransformConstants();
        }
        float* data_cuda;
        cudaMalloc(&data_cuda, n*constants::block_size_mem);
        cudaMemcpy(data_cuda,data, n*constants::block_size_mem, cudaMemcpyHostToDevice);
        
        DCT8x8_many(data_cuda, n, scale);

        // quantize (process many blocks at a time with paralell inside each block too)
        int16_t* quantized_cuda;
        cudaMalloc(&quantized_cuda,n * constants::block_size_mem);
        quantize_many<<<n,constants::block_size>>>(data_cuda,n,quantized_cuda);
        cudaDeviceSynchronize();

        cudaFree(data_cuda);

		// find pos non zero (paralell many blocks but serial inside block)
            // start counting from back and stop at first non-zero value, can skip most of the block then
        uint8_t* posNonZeros_cuda;
        cudaMalloc(&posNonZeros_cuda, n * sizeof(uint8_t));

        int cu_blockSize = 256;
        int cu_numBlocks = (n / cu_blockSize) + 1

        find_posNonZero_many<<<cu_numBlocks,cu_blockSize>>>(quantized_cuda, n, posNonZeros_cuda);
        cudaDeviceSynchronize();

        // copy data back from the device to the cpu
        cudaMemcpy(quantized,   quantized_cuda,   n * constants::block_size_mem, cudaMemcpyDeviceToHost);
        cudaMemcpy(posNonZeros, posNonZeros_cuda, n * sizeof(uint8_t),           cudaMemcpyDeviceToHost);
        
        cudaFree(new_scale);
        cudaFree(quantized_cuda);
        cudaFree(posNonZeros_cuda);
	}
}