//MIT License
//
//Copyright(c) 2020 Zheng Jiaqi @NUSComputing
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#include "cuda_runtime.h"
#include <stdio.h>

#define TOID(x, y, n) (y) * (n) + (x)

__host__ __device__ double scalar_product(double x1, double y1, double x2, double y2){
	return x1 * x2 + y1 * y2;
}

__host__ __device__ void barycentric_coordinate(
	double x1, double y1, double x2, double y2,
	double x3, double y3, double x0, double y0,
	double &w3, double &w1, double &w2
){
	double v0x = x2 - x1, v0y = y2 - y1;
	double v1x = x3 - x1, v1y = y3 - y1;
	double v2x = x0 - x1, v2y = y0 - y1;

	double d00 = scalar_product(v0x, v0y, v0x, v0y);
	double d01 = scalar_product(v0x, v0y, v1x, v1y);
	double d11 = scalar_product(v1x, v1y, v1x, v1y);
	double d20 = scalar_product(v2x, v2y, v0x, v0y);
	double d21 = scalar_product(v2x, v2y, v1x, v1y);

	double denom = d00 * d11 - d01 * d01;
	if (denom == 0) {
		w1 = w2 = w3 = -1;
		return;
	}
	w1 = (d11 * d20 - d01 * d21) / denom;
	w2 = (d00 * d21 - d01 * d20) / denom;
	w3 = 1.0 - w1 - w2;
}

__global__ void kernelDiscretization(
	double *points,
	double *weight,
	int *triangle,
	int num_tri,
	float *density,
	double scale,
	int n
){
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int id = TOID(tx, ty, n);
	double x = tx * scale, y = ty * scale;
	float res = 0;

	for (int i = 0; i < num_tri; ++i) {
		int p1 = triangle[i*3], p2 = triangle[i*3+1], p3 = triangle[i*3+2];
		double x1, x2, x3, y1, y2, y3, w1, w2, w3;
		x1 = points[p1 << 1], y1 = points[p1 << 1 | 1];
		x2 = points[p2 << 1], y2 = points[p2 << 1 | 1];
		x3 = points[p3 << 1], y3 = points[p3 << 1 | 1];
		barycentric_coordinate(x1, y1, x2, y2, x3, y3, x, y, w1, w2, w3);
		if (w1 < 0 || w2 < 0 || w3 < 0) continue;
		density[id] = w1 * weight[p1] + w2 * weight[p2] + w3 * weight[p3];
		return;
	}
	density[id] = 0;
	return;
}

void discretization_d(
	double *points,
	double *weight,
	int num_point,
	int *triangle,
	int num_tri,
	float *density,
	double scale,
	int n
){
	double *points_d, *weight_d;
	float *density_d;
	int *triangle_d;

	cudaMalloc((void **) &points_d,		num_point * sizeof(double) * 2);
	cudaMalloc((void **) &weight_d,		num_point * sizeof(double));
	cudaMalloc((void **) &triangle_d,	num_tri * sizeof(int) * 3);
	cudaMalloc((void **) &density_d,	n * n * sizeof(float));

	cudaMemcpy(points_d,	points, num_point * sizeof(double) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_d,	weight, num_point * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(triangle_d,	triangle, num_tri * sizeof(int) * 3, cudaMemcpyHostToDevice);
	
	dim3 block(16, 16);
	dim3 grid(n/block.x, n/block.y);
	kernelDiscretization <<< grid, block >>> (points_d, weight_d, triangle_d, num_tri, density_d, scale, n);

	cudaMemcpy(density, density_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(points_d);
	cudaFree(weight_d);
	cudaFree(triangle_d);
	cudaFree(density_d);
}
