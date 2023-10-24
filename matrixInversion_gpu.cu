#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <random>
#include <atomic>

using namespace std;

#define blocksize 32

/*storing matrix*/
void matrix_read(double *L, int dimension){
	FILE *fp;
	int row, col;

	fp = fopen("randomMatrix_2.txt", "r");//open output file
	if (fp == NULL)//open failed
		return;

	for (row = 0; row < dimension; row++){
		for (col = 0; col < dimension; col++)
		if (fscanf(fp, "%f,", &L[row * dimension + col]) == EOF) break;//read data

		if (feof(fp)) break;//if the file is over
	}

	fclose(fp);//close file

}

void generateRandomDoubleArray(double* arr, int arraySize) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < arraySize; ++i) {
        arr[i] = dist(gen);
    }
}

__global__ void nodiag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
  
	if (x < n && y < n){
  	if (x == i && x!=y){
  		I[x*n + y] /= A[i*n + i];
  		A[x*n + y] /= A[i*n + i];
  	}
	}
}

__global__ void diag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}

}

__global__ void set_zero(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

__global__ void matrix_inversion_kernel(double *input_A, double *input_I, double *output_A, double *output_I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
  int block_num = blockDim.x*blockDim.y;
  double temp_A;
  double temp_I;
  
	if (x < n && y < n){
   if(x==i){
     if(x==y){
       temp_A = 1;
       temp_I = 1/input_A[i*n + i];
     }else{
       temp_A = input_A[i*n + y]/input_A[i*n + i];
       temp_I = input_I[i*n + y]/input_A[i*n + i];
     }
   }else{
     temp_A = input_A[x*n + y] - input_A[i*n + y]*input_A[x*n + i]/input_A[i*n + i];
     temp_I = input_I[x*n + y] - input_I[i*n + y]*input_A[x*n + i]/input_A[i*n + i];
   }
	}
 __syncthreads();
 if (x < n && y < n){
   output_A[x*n + y] = temp_A;
   output_I[x*n + y] = temp_I;
 }
}

void savetofile(double *A, string s, int n, int h)
{
	std::ofstream plik;
	plik.open(s, std::ofstream::out | std::ofstream::trunc);

	for (int j = 0; j<h; j++){
		for (int i = 0; i<h; i++){
			plik << A[j*n + i] << "\t";
		}
		plik << endl;
	}
	plik.close();
}

int main()
{
  std::ofstream plik;
	plik.open("results.txt", std::ofstream::out | std::ofstream::trunc);
 
  int dimention_max = 8191;
  for(int n = dimention_max; n < dimention_max + 1; n++){
 
  	//const int n = 4;
  	// creating input
  	double *iL = new double[n*n];
  	double *L = new double[n*n];
  
    generateRandomDoubleArray(L, n*n);
    //for(int i = 0; i < n*n; i++){
    //  double temp = ((double)(i+2));
    //  L[i] = temp;
      //cout << L[i] << endl;
    //}
   
  	//matrix_read(L, n);
  	savetofile(L, "L.txt", n, n);
    savetofile(L, "L.txt", n, n);
    
  	cout << "inv\n";
  	double *d_A1 ,*d_A2, *d_L, *I, *dI1, *dI2;
  	float time;
  	cudaError_t err;
  	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
  	int ddsize = n*n*sizeof(double);
  
  	dim3 threadsPerBlock(blocksize, blocksize);
  	//dim3 numBlocks(n / blocksize, n / blocksize);
    int gridsize = ((n % blocksize) == 0) ? n / blocksize : n / blocksize + 1;
    dim3 numBlocks(gridsize, gridsize);
  	// memory allocation    
  	err = cudaMalloc((void**)&d_A1, ddsize);
  	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
  	err = cudaMalloc((void**)&dI1, ddsize);
  	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    err = cudaMalloc((void**)&d_A2, ddsize);
  	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
  	err = cudaMalloc((void**)&dI2, ddsize);
  	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
  	I = new double[n*n];
  
  	for (int i = 0; i<n; i++){
  		for (int j = 0; j<n; j++){
  			if (i == j) I[i*n + i] = 1.0;
  			else I[i*n + j] = 0.0;
  		}
  	}
  
  	//copy data from CPU to GPU
  	err = cudaMemcpy(d_A1, L, ddsize, cudaMemcpyHostToDevice);
  	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
  	err = cudaMemcpy(dI1, I, ddsize, cudaMemcpyHostToDevice);
  	if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
   
    int syncValue_1 = 0;
    int syncValue_2 = 0;
    int* d_syncValue_1;
    int* d_syncValue_2;
    cudaMalloc(&d_syncValue_1, sizeof(int));
    cudaMalloc(&d_syncValue_2, sizeof(int));
    cudaMemcpy(d_syncValue_1, &syncValue_1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syncValue_2, &syncValue_2, sizeof(int), cudaMemcpyHostToDevice);
  
  	//timer start
  	cudaEventRecord(start, 0);
  
  	// L^(-1)    
  	//for (int i = 0; i<n; i++){
  	//	nodiag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
  	//	diag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
  	//	gaussjordan << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
  	//	set_zero << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
  	//}
    for (int i = 0; i<n; i++){
     if(i % 2 == 0){
       matrix_inversion_kernel<<<numBlocks, threadsPerBlock>>>(d_A1, dI1, d_A2, dI2, n, i);
     }
     else{
       matrix_inversion_kernel<<<numBlocks, threadsPerBlock>>>(d_A2, dI2, d_A1, dI1, n, i);
     }
    }
  
  	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&time, start, stop);
  	cudaEventDestroy(start);
  	cudaEventDestroy(stop);
  
  	//copy data from GPU to CPU
    if(n % 2 == 0){
      err = cudaMemcpy(iL, dI1, ddsize, cudaMemcpyDeviceToHost);
  	  if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
  	  err = cudaMemcpy(I, d_A1, ddsize, cudaMemcpyDeviceToHost);
  	  if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    }else{
      err = cudaMemcpy(iL, dI2, ddsize, cudaMemcpyDeviceToHost);
  	  if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
  	  err = cudaMemcpy(I, d_A2, ddsize, cudaMemcpyDeviceToHost);
  	  if (err != cudaSuccess){ cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl; }
    }
  	
  
  	cout << "Dim" << n << "Cuda Time - inverse: " << time << "ms\n";
    plik << time << endl;
  	savetofile(iL, "inv.txt", n, n);
  	//savetofile(I, "I.txt", n, n);
  	cudaFree(d_A1);
  	cudaFree(dI1);
    cudaFree(d_A2);
  	cudaFree(dI2);
    cudaFree(d_syncValue_1);
    cudaFree(d_syncValue_2);
  
  	double *c = new double[n*n];
  	for (int i = 0; i<n; i++)  
  	for (int j = 0; j<n; j++)  
  	{
  		c[i*n+j] = 0;  //put the initial value to zero
  		for (int x = 0; x<n; x++)  
  			c[i*n + j] = c[i*n + j] + L[i*n+x] * iL[x*n + j];  //matrix multiplication
  	}
  	savetofile(c, "c.txt", n, n);
  
  	delete[]I;
  	delete[]L;
  	delete[]iL;
  }
	//system("Pause");
	return 0;
}