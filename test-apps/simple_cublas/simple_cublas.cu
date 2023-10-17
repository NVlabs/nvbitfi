#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#if USE_TENSOR_CORES == 1
#define BLAS_MODE CUBLAS_TF32_TENSOR_OP_MATH
#else
#define BLAS_MODE CUBLAS_PEDANTIC_MATH
#endif

static void checkCublasErrorsCall(cublasStatus_t error, int line, const char *file) {
    if (error == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::cerr << "CUDA CUBLAS error: " << error << line << ":" << file << std::endl;
    exit(EXIT_FAILURE);
}

#define checkCublasErrors(error) checkCublasErrorsCall(error, __LINE__, __FILE__);

void call_gemm(size_t dim, const half *v1_dev, const half *v2_dev, half *output_dev, cublasHandle_t blas_handle,
               half &alpha, half &beta) {
    checkCublasErrors(cublasHgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                  &alpha, v1_dev, dim, v2_dev, dim, &beta, output_dev, dim));
}

void call_gemm(size_t dim, const float *v1_dev, const float *v2_dev, float *output_dev, cublasHandle_t blas_handle,
               float &alpha, float &beta) {
    checkCublasErrors(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                  &alpha, v1_dev, dim, v2_dev, dim, &beta, output_dev, dim));
}


template<typename real_t>
void run_gemm(size_t dim, real_t val_v1, real_t val_v2) {
    auto size = dim * dim;
    std::cout << "Testing CUBLAS with array size:" << size << std::endl;
#if USE_TENSOR_CORES == 1
    std::cout << "Using tensor cores\n";
#else
    std::cout << "Using default math\n";
#endif
    real_t *v1_dev, *v2_dev, *output_dev;
    std::vector<real_t> v1(size, val_v1), v2(size, val_v2), output(size);
    auto byte_count = sizeof(real_t) * size;
    cudaMalloc(&v1_dev, byte_count);
    cudaMalloc(&v2_dev, byte_count);
    cudaMalloc(&output_dev, byte_count);
    cudaMemcpy(v1_dev, v1.data(), byte_count, cudaMemcpyHostToDevice);
    cudaMemcpy(v2_dev, v2.data(), byte_count, cudaMemcpyHostToDevice);
    cudaMemset(output_dev, 0x0, byte_count);

    cublasHandle_t blas_handle;
    checkCublasErrors(cublasCreate(&blas_handle));
    checkCublasErrors(cublasSetMathMode(blas_handle, BLAS_MODE));

    real_t alpha = 1.0f, beta = 0.0f;
    call_gemm(dim, v1_dev, v2_dev, output_dev, blas_handle, alpha, beta);

    cudaMemcpy(output.data(), output_dev, byte_count, cudaMemcpyDeviceToHost);
    auto expected_val = (double) dim;
    for (auto i = 0; i < output.size(); i++) {
        auto out = double(output[i]);
        if (out != expected_val) {
            std::cout << "M[" << i << "]:" << out << std::endl;
        }
    }

    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(output_dev);
    std::cout << "Finished computation\n";
}

int main(int argc, char *argv[]) {
    setbuf(stdout, nullptr); // Disable stdout buffering
    auto size = 256;
    run_gemm<half>(size, 1.0f, 1.0f);
    return 0;
}
