#include<iostream>
#include<chrono>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<torch/extension.h>
#include<ATen/cuda/CUDAContext.h>
#include<c10/cuda/CUDAStream.h>



const int WARP_SIZE = 32;
const int BLOCK_SIZE = 16;


__global__ void SPMM_total(int *rowPtr, int *colIdx, float *values, int *offset, float *B, float *MT, int M, int N);
__global__ void Add_total_s(float *B, float *MB, int M, int N);
__global__ void Add_finish_s(float *MT, float *output, int M, int N);



torch::Tensor spmm_strassen_cuda(torch::Tensor A_rowPtr,
                            torch::Tensor A_colIdx,
                            torch::Tensor A_values,
                            torch::Tensor offset,
                            torch::Tensor X){
    int m = X.size(0);
    int n = X.size(1);
    int halfm = (m+1) >> 1;
    int halfn = (n+1) >> 1;
    
    // define C11 - C22
    torch::Tensor output = torch::zeros({m, n}, torch::kCUDA).contiguous();

    // M1-7
    torch::Tensor MT = torch::zeros({7 * halfm, halfn}, torch::kCUDA).contiguous();

    // define block and thread

    dim3 TPB(BLOCK_SIZE, BLOCK_SIZE);
    dim3 BPG((halfn + BLOCK_SIZE - 1) / BLOCK_SIZE, (halfm + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dim3 spTPB(WARP_SIZE, WARP_SIZE);
    dim3 spBPG((halfn + WARP_SIZE - 1) / WARP_SIZE, (halfm + WARP_SIZE - 1) / WARP_SIZE, 7);

    // intermediate results in a scope when finished, they are freed automatically
    {
        torch::Tensor MB = torch::zeros({7 * halfm, halfn}, torch::kCUDA).contiguous();

        // dense add
        // add is right
        Add_total_s<<<BPG, TPB>>>(X.data_ptr<float>(), MB.data_ptr<float>(), halfm, halfn);

        // std::cout<<MB<<"\n";

        // spmm
        SPMM_total<<<spBPG, spTPB>>>(A_rowPtr.data_ptr<int>(), A_colIdx.data_ptr<int>(), A_values.data_ptr<float>(), offset.data_ptr<int>(), 
                                     MB.data_ptr<float>(), MT.data_ptr<float>(), halfm, halfn);
        
    }

    Add_finish_s<<<BPG, TPB>>>(MT.data_ptr<float>(), output.data_ptr<float>(), halfm, halfn);

    return output;
}

// /////////////////////////////////
// // implementation
// /////////////////////////////////

// cuda implement

__global__ void Add_total_s(float *B, float *MB, int M, int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int NN = N << 1;
    __shared__ float SMB11[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMB12[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMB21[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMB22[BLOCK_SIZE][BLOCK_SIZE];
    if(row >= M || col >= N) return;
    // write to shared memory
    SMB11[threadIdx.y][threadIdx.x] = B[row * NN + col];
    SMB12[threadIdx.y][threadIdx.x] = B[row * NN + col + N];
    SMB21[threadIdx.y][threadIdx.x] = B[(row + M) * NN + col];
    SMB22[threadIdx.y][threadIdx.x] = B[(row + M) * NN + col + N];
    __syncthreads();

    //calculate
    MB[row * N + col] = SMB11[threadIdx.y][threadIdx.x] + SMB22[threadIdx.y][threadIdx.x];
    MB[(row + M) * N + col] = SMB11[threadIdx.y][threadIdx.x];
    MB[(row + M * 2) * N + col] = SMB12[threadIdx.y][threadIdx.x] - SMB22[threadIdx.y][threadIdx.x];
    MB[(row + M * 3) * N + col] = SMB21[threadIdx.y][threadIdx.x] - SMB11[threadIdx.y][threadIdx.x];
    MB[(row + M * 4) * N + col] = SMB22[threadIdx.y][threadIdx.x];
    MB[(row + M * 5) * N + col] = SMB11[threadIdx.y][threadIdx.x] + SMB12[threadIdx.y][threadIdx.x];
    MB[(row + M * 6) * N + col] = SMB21[threadIdx.y][threadIdx.x] + SMB22[threadIdx.y][threadIdx.x];

}

__global__ void Add_finish_s(float *MT, float *output, int M, int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int NN = N << 1;
    // shared mem
    __shared__ float SMT1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMT2[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMT3[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMT4[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMT5[BLOCK_SIZE][BLOCK_SIZE];

    if(row >= M || col >= N) return;

    int idx_M1 = row * N + col;
    int idx_M2 = (row + M) * N + col;
    int idx_M3 = (row + M * 2) * N + col;
    int idx_M4 = (row + M * 3) * N + col;
    int idx_M5 = (row + M * 4) * N + col;
    int idx_M6 = (row + M * 5) * N + col;
    int idx_M7 = (row + M * 6) * N + col;

    int idx_C11 = row * NN + col;
    int idx_C12 = row * NN + col + N;
    int idx_C21 = (row + M) * NN + col;
    int idx_C22 = (row + M) * NN + col + N;

    // write to shared mem
    SMT1[threadIdx.y][threadIdx.x] = MT[idx_M1];
    SMT2[threadIdx.y][threadIdx.x] = MT[idx_M2];
    SMT3[threadIdx.y][threadIdx.x] = MT[idx_M3];
    SMT4[threadIdx.y][threadIdx.x] = MT[idx_M4];
    SMT5[threadIdx.y][threadIdx.x] = MT[idx_M5];
    __syncthreads();

    output[idx_C11] = SMT1[threadIdx.y][threadIdx.x] + SMT4[threadIdx.y][threadIdx.x] - SMT5[threadIdx.y][threadIdx.x] + MT[idx_M7];
    output[idx_C12] = SMT3[threadIdx.y][threadIdx.x] + SMT5[threadIdx.y][threadIdx.x];
    output[idx_C21] = SMT2[threadIdx.y][threadIdx.x] + SMT4[threadIdx.y][threadIdx.x];
    output[idx_C22] = SMT1[threadIdx.y][threadIdx.x] - SMT2[threadIdx.y][threadIdx.x] + SMT3[threadIdx.y][threadIdx.x] + MT[idx_M6];

  
}

__global__ void SPMM_total(int *rowPtr, int *colIdx, float *values, int *offset, float *B, float *MT, int M, int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int osr = blockIdx.z * (M + 1);
    int oso = blockIdx.z * M;
    int osc = offset[blockIdx.z];
    if(row < M && col < N){
        float sum = 0.0;
        #pragma unroll
        for(int i = rowPtr[row + osr] ; i < rowPtr[row + osr + 1] ; i++){
            int colA = colIdx[i + osc];
            sum += values[i + osc] * B[(colA + oso) * N + col];
        }
        MT[(row + oso) * N + col] = sum;
    }
}