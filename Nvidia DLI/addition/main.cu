#include <iostream>
#include <cmath>

using namespace std;

__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1<<20; // this is 1 million elements wtf
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    //initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    //run kernel on 1m elements on the cpu
    add<<<numBlocks, blockSize>>>(N, x, y);

    //wait for gpu to finish before accessing on host
    cudaDeviceSynchronize();

    //check for errors all vals shd be 3.0f
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    cout << "Max error: " << maxError << endl;


    cudaFree(x);
    cudaFree(y);
    return 0;
}