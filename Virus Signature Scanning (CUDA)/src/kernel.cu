#include <vector>
#include "defs.h"
#include <iostream>


__global__ void matchFile(const uint8_t* file_data, size_t file_len, char** signatures, size_t* lens, int* founds)
{
    // TODO: your code!
    size_t sig_len = lens[blockIdx.x]/2;
    char* signature = signatures[blockIdx.x];

    size_t start = threadIdx.x;
    size_t end = file_len - sig_len + 1;
    size_t stride = blockDim.x;


    for (size_t i = start; i < end; i += stride)
    {
        size_t j;
        for (j = 0; j != sig_len; ++j)
        {
            const char h1 = signature[j * 2];
            const char h2 = signature[j * 2 + 1];
            uint8_t b1;
            uint8_t b2;
            if (h1 >= '0' && h1<= '9')
                b1 = h1 - '0';
            else if (h1 == '?')
                b1 = file_data[i + j] >> 4;
            else
                b1 = h1 - 'a' + 10;

            if (h2 >= '0' && h2 <= '9')
                b2 = h2 - '0';
            else if (h2 == '?')
                b2 = file_data[i + j] & 15;
            else
                b2 = h2 - 'a' + 10;

            uint8_t byte = (b1 << 4) | b2;

            if (file_data[i + j] != byte) break;
        }

        if (j == sig_len)
            atomicExch(&founds[blockIdx.x], 1);
        
        if (founds[blockIdx.x] == 1) return;

    }
}

void runScanner(std::vector<Signature>& signatures, std::vector<InputFile>& inputs)
{
    {
        cudaDeviceProp prop;
        check_cuda_error(cudaGetDeviceProperties(&prop, 0));

        fprintf(stderr, "cuda stats:\n");
        fprintf(stderr, "  # of SMs: %d\n", prop.multiProcessorCount);
        fprintf(stderr, "  global memory: %.2f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
        fprintf(stderr, "  shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
        fprintf(stderr, "  constant memory: %zu bytes\n", prop.totalConstMem);
    }

    /*
       Here, we are creating one stream per file just for demonstration purposes;
       you should change this to fit your own algorithm and/or implementation.
     */
    std::vector<cudaStream_t> streams {};
    streams.resize(inputs.size());

    std::vector<uint8_t*> file_bufs {};
    std::vector<size_t*> file_size_bufs {};
    for(size_t i = 0; i < inputs.size(); i++)
    {
        cudaStreamCreate(&streams[i]);

        // allocate memory on the device for the file
        uint8_t* ptr = 0;
        check_cuda_error(cudaMalloc(&ptr, inputs[i].size));
        file_bufs.push_back(ptr);
    }

    // allocate memory for the signatures
    std::vector<char*> sig_bufs {};
    size_t* d_sig_size;
    size_t* h_sig_size = (size_t*)malloc(sizeof(size_t) * signatures.size());
    size_t size = signatures.size() * sizeof(size_t);
    check_cuda_error(cudaMalloc((void**) &d_sig_size, size));

    char** d_sig_bufs;
    char** h_sig_bufs = (char**)malloc(sizeof(char*) * signatures.size());
    size_t size_data = signatures.size() * sizeof(char*);
    check_cuda_error(cudaMalloc((void**) &d_sig_bufs, size_data));

    for(size_t i = 0; i < signatures.size(); i++)
    { 
        char* ptr = 0;
        h_sig_size[i] = signatures[i].size;
        check_cuda_error(cudaMalloc(&ptr, signatures[i].size));
        cudaMemcpy(ptr, signatures[i].data, signatures[i].size, cudaMemcpyHostToDevice);
        h_sig_bufs[i] = ptr;
    }
    cudaMemcpy(d_sig_size, h_sig_size, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig_bufs, h_sig_bufs, size_data, cudaMemcpyHostToDevice);
    // allocate memory for the results

    std::vector<int*> results;
    for (int i = 0; i != inputs.size(); ++i)
    {
        int* ptr = 0;
        cudaMallocManaged(&ptr, sizeof(int) * signatures.size());
        cudaMemset(ptr, 0, sizeof(int) * signatures.size());
        results.push_back(ptr);
    }

    for(size_t file_idx = 0; file_idx < inputs.size(); file_idx++)
    {
        // asynchronously copy the file contents from host memory
        // (the `inputs`) to device memory (file_bufs, which we allocated above)
        cudaMemcpyAsync(file_bufs[file_idx], inputs[file_idx].data, inputs[file_idx].size,
                cudaMemcpyHostToDevice, streams[file_idx]);    // pass in the stream here to do this async

            // launch the kernel!
            // your job: figure out the optimal dimensions
    
            /*
               This launch happens asynchronously. This means that the CUDA driver returns control
               to our code immediately, without waiting for the kernel to finish. We can then
               run another iteration of this loop to launch more kernels.

               Each operation on a given stream is serialised; in our example here, we launch
               all signatures on the same stream for a file, meaning that, in practice, we get
               a maximum of NUM_INPUTS kernels running concurrently.

               Of course, the hardware can have lower limits; on Compute Capability 8.0, at most
               128 kernels can run concurrently --- subject to resource constraints. This means
               you should *definitely* be doing more work per kernel than in our example!
             */
        matchFile<<<signatures.size(), 512, /* shared memory per block: */ 0, streams[file_idx]>>>(
                file_bufs[file_idx], inputs[file_idx].size,
                d_sig_bufs, d_sig_size, results[file_idx]);


            // example output printing. don't forget to change this!
            // printf("%s: %s\n", inputs[file_idx].name.c_str(), signatures[sig_idx].name.c_str());
    }

    cudaDeviceSynchronize();

    // print the results

    for (int i = 0; i != inputs.size(); ++i)
    {
        for (int j = 0; j != signatures.size(); ++j)
        {
            if (results[i][j] == 1) printf("%s: %s\n", inputs[i].name.c_str(), signatures[j].name.c_str());
        }
    }

    // free the device memory, though this is not strictly necessary
    // (the CUDA driver will clean up when your program exits)
    for(auto buf : file_bufs)
        cudaFree(buf);

    for(auto buf : sig_bufs)
        cudaFree(buf);

    for (auto buf: results)
        cudaFree(buf);

    // clean up streams (again, not strictly necessary)
    for(auto& s : streams)
        cudaStreamDestroy(s);
}
