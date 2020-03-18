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

#include <stdio.h>
#include <unordered_map>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <helper_timer.h>
#include <vector>
#include <algorithm>

// Parameters for CUDA kernel executions
#define BLOCKX			16
#define BLOCKY			16
#define BLOCKSIZE		64
#define BAND			256			// For simplicity, just assume we never need to work with a smaller texture.
#define THRESHOLD		1e-5
#define MARKER			-32768
#define TOID(x, y, n)	((y) * (n) + (x))

#define debug_error 1
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Global Variables
short2 **pbaTextures, *pbaMargin;       // Two textures used to compute 2D Voronoi Diagram
short2 *pbaVoronoi, *pbaTemp;
float **pbaDensity;
float **pbaPrefixX, **pbaPrefixY, **pbaPrefixW;
float *pbaTotalX, *pbaTotalY, *pbaTotalW;
float *pbaEnergyTex, pbaEnergy_h;
float pbaOmega;
bool *constrainMask_d;

int pbaScale;
int pbaBuffer;              // Current buffer
int pbaMemSize;             // Size (in bytes) of a texture
int pbaTexSize;             // Texture size (squared texture)

// Fill an array with some value
__global__ void kernelFillShort(short2* arr, short value, int texSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	arr[__mul24(y, texSize) + x] = make_short2(value, value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Parallel Banding Algorithm plus //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelFloodDown(short2 *input, short2 *output, int size, int bandSize)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * bandSize;
	int id = TOID(tx, ty, size);

	short2 pixel1, pixel2;

	pixel1 = make_short2(MARKER, MARKER);

	for (int i = 0; i < bandSize; i++, id += size) {
		pixel2 = input[id];

		if (pixel2.x != MARKER)
			pixel1 = pixel2;

		output[id] = pixel1;
	}
}

__global__ void kernelFloodUp(short2 *input, short2 *output, int size, int bandSize)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = (blockIdx.y + 1) * bandSize - 1;
	int id = TOID(tx, ty, size);

	short2 pixel1, pixel2;
	int dist1, dist2;

	pixel1 = make_short2(MARKER, MARKER);

	for (int i = 0; i < bandSize; i++, id -= size) {
		dist1 = abs(pixel1.y - ty + i);

		pixel2 = input[id];
		dist2 = abs(pixel2.y - ty + i);

		if (dist2 < dist1)
			pixel1 = pixel2;

		output[id] = pixel1;
	}
}

__global__ void kernelPropagateInterband(short2 *input, short2 *output, int size, int bandSize)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int inc = bandSize * size;
	int ny, nid, nDist;
	short2 pixel;

	// Top row, look backward
	int ty = blockIdx.y * bandSize;
	int topId = TOID(tx, ty, size);
	int bottomId = TOID(tx, ty + bandSize - 1, size);
	int tid = blockIdx.y * size + tx;
	int bid = tid + (size * size / bandSize);

	pixel = input[topId];
	int myDist = abs(pixel.y - ty);
	output[tid] = pixel;

	for (nid = bottomId - inc; nid >= 0; nid -= inc) {
		pixel = input[nid];

		if (pixel.x != MARKER) {
			nDist = abs(pixel.y - ty);

			if (nDist < myDist)
				output[tid] = pixel;

			break;
		}
	}

	// Last row, look downward
	ty = ty + bandSize - 1;
	pixel = input[bottomId];
	myDist = abs(pixel.y - ty);
	output[bid] = pixel;

	for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc) {
		pixel = input[nid];

		if (pixel.x != MARKER) {
			nDist = abs(pixel.y - ty);

			if (nDist < myDist)
				output[bid] = pixel;

			break;
		}
	}
}

__global__ void kernelUpdateVertical(short2 *color, short2 *margin, short2 *output, int size, int bandSize)
{
	__shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * bandSize;

	short2 top = margin[blockIdx.y * size + tx];
	short2 bottom = margin[(blockIdx.y + size / bandSize) * size + tx];
	short2 pixel;

	int dist, myDist;

	int id = TOID(tx, ty, size);

	int n_step = bandSize / blockDim.x;
	for (int step = 0; step < n_step; ++step) {
		int y_start = blockIdx.y * bandSize + step * blockDim.x;
		int y_end = y_start + blockDim.x;

		for (ty = y_start; ty < y_end; ++ty, id += size) {
			pixel = color[id];
			myDist = abs(pixel.y - ty);

			dist = abs(top.y - ty);
			if (dist < myDist) { myDist = dist; pixel = top; }

			dist = abs(bottom.y - ty);
			if (dist < myDist) pixel = bottom;

			block[threadIdx.x][ty - y_start] = make_short2(pixel.y, pixel.x);
		}

		__syncthreads();

		int tid = TOID(blockIdx.y * bandSize + step * blockDim.x + threadIdx.x, \
			blockIdx.x * blockDim.x, size);

		for (int i = 0; i < blockDim.x; ++i, tid += size) {
			output[tid] = block[i][threadIdx.x];
		}

		__syncthreads();
	}
}

#define LL long long
__device__ bool dominate(LL x1, LL y1, LL x2, LL y2, LL x3, LL y3, LL x0)
{
	LL k1 = y2 - y1, k2 = y3 - y2;
	return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - (x0 << 1))) * k2 > \
		(k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - (x0 << 1))) * k1;
}
#undef LL

__global__ void kernelProximatePoints(short2 *input, short2 *stack, int size, int bandSize)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = __mul24(blockIdx.y, bandSize);
	int id = TOID(tx, ty, size);
	int lasty = -1;
	short2 last1, last2, current;

	last1.y = -1; last2.y = -1;

	for (int i = 0; i < bandSize; i++, id += size) {
		current = input[id];

		if (current.x != MARKER) {
			while (last2.y >= 0) {
				if (!dominate(last1.x, last2.y, last2.x, \
					lasty, current.x, current.y, tx))
					break;

				lasty = last2.y; last2 = last1;

				if (last1.y >= 0)
					last1 = stack[TOID(tx, last1.y, size)];
			}

			last1 = last2; last2 = make_short2(current.x, lasty); lasty = current.y;

			stack[id] = last2;
		}
	}

	// Store the pointer to the tail at the last pixel of this band
	if (lasty != ty + bandSize - 1)
		stack[TOID(tx, ty + bandSize - 1, size)] = make_short2(MARKER, lasty);
}

__global__ void kernelCreateForwardPointers(short2 *input, short2 *output, int size, int bandSize)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = __mul24(blockIdx.y + 1, bandSize) - 1;
	int id = TOID(tx, ty, size);
	int lasty = -1, nexty;
	short2 current;

	// Get the tail pointer
	current = input[id];

	if (current.x == MARKER)
		nexty = current.y;
	else
		nexty = ty;

	for (int i = 0; i < bandSize; i++, id -= size)
		if (ty - i == nexty) {
			current = make_short2(lasty, input[id].y);
			output[id] = current;

			lasty = nexty;
			nexty = current.y;
		}

	// Store the pointer to the head at the first pixel of this band
	if (lasty != ty - bandSize + 1)
		output[id + size] = make_short2(lasty, MARKER);
}

__global__ void kernelMergeBands(short2 *color, short2 *link, short2 *output, int size, int bandSize)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int band1 = blockIdx.y * 2;
	int band2 = band1 + 1;
	int firsty, lasty;
	short2 last1, last2, current;
	// last1 and last2: x component store the x coordinate of the site, 
	// y component store the backward pointer
	// current: y component store the x coordinate of the site, 
	// x component store the forward pointer

	// Get the two last items of the first list
	lasty = __mul24(band2, bandSize) - 1;
	last2 = make_short2(color[TOID(tx, lasty, size)].x,
		link[TOID(tx, lasty, size)].y);

	if (last2.x == MARKER) {
		lasty = last2.y;

		if (lasty >= 0)
			last2 = make_short2(color[TOID(tx, lasty, size)].x,
				link[TOID(tx, lasty, size)].y);
		else
			last2 = make_short2(MARKER, MARKER);
	}

	if (last2.y >= 0) {
		// Second item at the top of the stack
		last1 = make_short2(color[TOID(tx, last2.y, size)].x,
			link[TOID(tx, last2.y, size)].y);
	}

	// Get the first item of the second band
	firsty = __mul24(band2, bandSize);
	current = make_short2(link[TOID(tx, firsty, size)].x,
		color[TOID(tx, firsty, size)].x);

	if (current.y == MARKER) {
		firsty = current.x;

		if (firsty >= 0)
			current = make_short2(link[TOID(tx, firsty, size)].x,
				color[TOID(tx, firsty, size)].x);
		else
			current = make_short2(MARKER, MARKER);
	}

	// Count the number of item in the second band that survive so far. 
	// Once it reaches 2, we can stop. 
	int top = 0;

	while (top < 2 && current.y >= 0) {
		// While there's still something on the left
		while (last2.y >= 0) {

			if (!dominate(last1.x, last2.y, last2.x, \
				lasty, current.y, firsty, tx))
				break;

			lasty = last2.y; last2 = last1;
			top--;

			if (last1.y >= 0)
				last1 = make_short2(color[TOID(tx, last1.y, size)].x,
					link[TOID(tx, last1.y, size)].y);
		}

		// Update the current pointer 
		output[TOID(tx, firsty, size)] = make_short2(current.x, lasty);

		if (lasty >= 0)
			output[TOID(tx, lasty, size)] = make_short2(firsty, last2.y);

		last1 = last2; last2 = make_short2(current.y, lasty); lasty = firsty;
		firsty = current.x;

		top = max(1, top + 1);

		// Advance the current pointer to the next one
		if (firsty >= 0)
			current = make_short2(link[TOID(tx, firsty, size)].x,
				color[TOID(tx, firsty, size)].x);
		else
			current = make_short2(MARKER, MARKER);
	}

	// Update the head and tail pointer. 
	firsty = __mul24(band1, bandSize);
	lasty = __mul24(band2, bandSize);
	current = link[TOID(tx, firsty, size)];

	if (current.y == MARKER && current.x < 0) { // No head?
		last1 = link[TOID(tx, lasty, size)];

		if (last1.y == MARKER)
			current.x = last1.x;
		else
			current.x = lasty;

		output[TOID(tx, firsty, size)] = current;
	}

	firsty = __mul24(band1, bandSize) + bandSize - 1;
	lasty = __mul24(band2, bandSize) + bandSize - 1;
	current = link[TOID(tx, lasty, size)];

	if (current.x == MARKER && current.y < 0) { // No tail?
		last1 = link[TOID(tx, firsty, size)];

		if (last1.x == MARKER)
			current.y = last1.y;
		else
			current.y = firsty;

		output[TOID(tx, lasty, size)] = current;
	}
}

__global__ void kernelDoubleToSingleList(short2 *color, short2 *link, short2 *output, int size)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = blockIdx.y;
	int id = TOID(tx, ty, size);

	output[id] = make_short2(color[id].x, link[id].y);
}

__global__ void kernelColor(short2 *input, short2 *output, int size)
{
	__shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

	int col = threadIdx.x;
	int tid = threadIdx.y;
	int tx = __mul24(blockIdx.x, blockDim.x) + col;
	int dx, dy, lasty;
	unsigned int best, dist;
	short2 last1, last2;

	lasty = size - 1;

	last2 = input[TOID(tx, lasty, size)];

	if (last2.x == MARKER) {
		lasty = last2.y;
		last2 = input[TOID(tx, lasty, size)];
	}

	if (last2.y >= 0)
		last1 = input[TOID(tx, last2.y, size)];

	int y_start, y_end, n_step = size / blockDim.x;
	for (int step = 0; step < n_step; ++step) {
		y_start = size - step * blockDim.x - 1;
		y_end = size - (step + 1) * blockDim.x;

		for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
			dx = last2.x - tx; dy = lasty - ty;
			best = dist = __mul24(dx, dx) + __mul24(dy, dy);

			while (last2.y >= 0) {
				dx = last1.x - tx; dy = last2.y - ty;
				dist = __mul24(dx, dx) + __mul24(dy, dy);

				if (dist > best)
					break;

				best = dist; lasty = last2.y; last2 = last1;

				if (last2.y >= 0)
					last1 = input[TOID(tx, last2.y, size)];
			}

			block[threadIdx.x][ty - y_end] = make_short2(lasty, last2.x);
		}

		__syncthreads();

		int iinc = size * blockDim.y;
		int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x + tid, size);
		for (int i = tid; i < blockDim.x; i += blockDim.y, id += iinc) {
			output[id] = block[i][threadIdx.x];
		}

		__syncthreads();
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Centroidal Voronoi Tessellation ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelZoomIn(short2 *input, short2 *output, int size, int scale)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int id = TOID(tx, ty, size);
	int tid = TOID(tx << scale, ty << scale, size << scale);

	short2 pixel = input[id];

	output[tid] = (pixel.x == MARKER) ? make_short2(MARKER, MARKER) : make_short2(pixel.x << scale, pixel.y << scale);
}

__global__ void kernelDensityScaling(float *input, float *output, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	float density = 0;

	for (int x = (tx << 1); x < (tx << 1) + 2; ++x) {
		for (int y = (ty << 1); y < (ty << 1) + 2; ++y) {
			density += input[TOID(x, y, size << 1)];
		}
	}

	output[TOID(tx, ty, size)] = density / 4.0;
}

// compute the prefix sum of weight, x*weight and y*weight for each row
extern __shared__ float tmpScan[];
__global__ void kernelComputeWeightedPrefixX(float *prefixW, float *prefixX, float *prefixY, float *density, int texWidth)
{
	float *tmpX = tmpScan;
	float *tmpY = tmpX + blockDim.x;
	float *tmpWeight = tmpY + blockDim.x;

	float pW, pX, pY;

	int tid = threadIdx.x;
	int tx, ty = blockIdx.x;
	float lastX = 0.0f, lastY = 0.0f, lastW = 0.0f;
	int id = __mul24(ty, texWidth);

	for (int xx = 0; xx < texWidth; xx += blockDim.x) {
		tx = xx + tid;
		pW = density[id + tx];
		pX = lastX + tx * pW;
		pY = lastY + ty * pW;
		pW = lastW + pW;
		tmpWeight[tid] = pW; tmpX[tid] = pX; tmpY[tid] = pY;
		__syncthreads();

		for (int step = 1; step < blockDim.x; step *= 2) { // parallel prefix sum within a block
			if (tid >= step) {
				pW += tmpWeight[tid - step];
				pX += tmpX[tid - step];
				pY += tmpY[tid - step];
			}

			__syncthreads();
			tmpWeight[tid] = pW; tmpX[tid] = pX; tmpY[tid] = pY;
			__syncthreads();
		}

		prefixX[id + tx] = tmpX[tid];
		prefixY[id + tx] = tmpY[tid];
		prefixW[id + tx] = tmpWeight[tid];

		if (tid == 0) {
			lastX = tmpX[blockDim.x - 1];
			lastY = tmpY[blockDim.x - 1];
			lastW = tmpWeight[blockDim.x - 1];
		}
		__syncthreads();
	}
}

// 2D -> 1D Voronoi Diagram
extern __shared__ short sharedVor[];
__global__ void kernelVoronoi1D(short2 *input, int *output, int texWidth)
{
	int tid = threadIdx.x;
	int tx, ty = blockIdx.x;
	int id = __mul24(ty, texWidth);

	// Initialize
	for (tx = tid; tx < texWidth; tx += blockDim.x)
		sharedVor[tx] = MARKER;

	__syncthreads();

	// Mark
	for (tx = tid; tx < texWidth; tx += blockDim.x) {
		short2 pixel = input[id + tx];

		sharedVor[pixel.x] = pixel.y;
	}

	__syncthreads();

	// Write
	id /= 2;
	for (tx = tid; tx < texWidth / 2; tx += blockDim.x)
		output[id + tx] = ((int *)sharedVor)[tx];
}

__global__ void kernelTotal_X(short2 *voronoi, float *prefixX, float *prefixY, float *prefixW, \
	float *totalX, float *totalY, float *totalW, int texWidth)
{
	// Shared array to store the sums
	__shared__ float sharedTotalX[BAND];  // BAND = 256
	__shared__ float sharedTotalY[BAND];
	__shared__ float sharedTotalW[BAND];
	__shared__ int startBlk[100], endBlk[100];	// 100 blocks is more than enough

	int count;
	int tid = threadIdx.x;
	int tx, ty = blockIdx.x, offset;
	int id = __mul24(ty, texWidth);
	short2 me, other;

	int margin = tid * BAND;

	if (margin < texWidth) {
		startBlk[tid] = 0;
		endBlk[tid] = texWidth;

		for (tx = 0; tx < texWidth; tx += blockDim.x) {
			me = voronoi[id + tx];

			if (me.x >= margin) {
				startBlk[tid] = max(0, tx - int(blockDim.x));
				break;
			}
		}

		for (; tx < texWidth; tx += blockDim.x) {
			me = voronoi[id + tx];

			if (me.x >= margin + BAND) {
				endBlk[tid] = tx;
				break;
			}
		}
	}

	__syncthreads();

	count = 0;

	// We process one BAND at a time. 
	for (margin = 0; margin < texWidth; margin += BAND, count++) {
		// Only for the first iteration of tx
		// Make sure we detect the boundary at tx = 0
		other.x = -1;

		// Left edge, scan through the row
		for (tx = startBlk[count] + tid; tx < endBlk[count]; tx += blockDim.x) {
			if (tx > 0)
				other = voronoi[id + tx - 1];

			me = voronoi[id + tx];
			offset = me.x - margin;

			// margin <= me.x < margin + BAND  &&  the closest site of the previous pixel is different
			if (offset >= 0 && offset < BAND && other.x < me.x) {
				if (tx > 0) {
					sharedTotalX[offset] = prefixX[id + tx - 1];
					sharedTotalY[offset] = prefixY[id + tx - 1];
					sharedTotalW[offset] = prefixW[id + tx - 1];
				}
				else {
					sharedTotalX[offset] = 0.0f;
					sharedTotalY[offset] = 0.0f;
					sharedTotalW[offset] = 0.0f;
				}
			}
		}

		__syncthreads();

		// Right edge
		for (tx = startBlk[count] + tid; tx < endBlk[count]; tx += blockDim.x) {
			me = voronoi[id + tx];
			offset = me.x - margin;

			if (tx < texWidth - 1)
				other = voronoi[id + tx + 1];
			else
				other.x = texWidth;

			// margin <= me.x < margin + BAND  &&  the closest site of the next pixel is different
			if (offset >= 0 && offset < BAND && me.x < other.x) {
				sharedTotalX[offset] = prefixX[id + tx] - sharedTotalX[offset];
				sharedTotalY[offset] = prefixY[id + tx] - sharedTotalY[offset];
				sharedTotalW[offset] = prefixW[id + tx] - sharedTotalW[offset];
			}
		}

		__syncthreads();

		// Write
		for (tx = tid; tx < BAND; tx += blockDim.x)
			if (margin + tx < texWidth) {
				totalX[id + margin + tx] = sharedTotalX[tx];
				totalY[id + margin + tx] = sharedTotalY[tx];
				totalW[id + margin + tx] = sharedTotalW[tx];
			}
	}
}

__global__ void kernelScan_Y(short *voronoi, float *totalX, float *totalY, float *totalW, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * BLOCKSIZE;
	int id = TOID(tx, ty, size), tid;
	short pixel, last = MARKER;
	float tmpX = 0.0, tmpY = 0.0, tmpW = 0.0;

	for (int i = 0; i < BLOCKSIZE; ++i, ++ty, id += size) {
		__syncthreads();
		pixel = voronoi[id];

		if (pixel != last) {
			if (last != MARKER) {
				tid = TOID(tx, last, size);
				atomicAdd(totalX + tid, tmpX);
				atomicAdd(totalY + tid, tmpY);
				atomicAdd(totalW + tid, tmpW);
			}
			tmpX = tmpY = tmpW = 0.0;
			last = pixel;
		}

		if (pixel != MARKER && pixel != ty) {
			tmpX += totalX[id];
			tmpY += totalY[id];
			tmpW += totalW[id];
		}
	}

	if (last != MARKER) {
		tid = TOID(tx, last, size);
		atomicAdd(totalX + tid, tmpX);
		atomicAdd(totalY + tid, tmpY);
		atomicAdd(totalW + tid, tmpW);
	}
}

__global__ void kernelDebug_Y(short *voronoi, float *totalX, float *totalY, float *totalW, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = 0;
	int id = TOID(tx, ty, size);
	short pixel, last = MARKER;

	for (; ty < size; ++ty, id += size) {
		pixel = voronoi[id];

		if (pixel != last) {
			if (last != MARKER) printf("%d %d\n", tx, last);
			last = pixel;
		}
	}

	if (last != MARKER) printf("%d %d\n", tx, last);
}

__global__ void kernelUpdateSites(short *voronoi, float *totalX, float *totalY, float *totalW, float *density,
	short2 *output, bool *mask, int size, float omega)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	float pX, pY, pW;

	int id = TOID(tx, ty, size);
	short pixel, seed = voronoi[id];

	if (seed != ty) return;

	pX = totalX[id];
	pY = totalY[id];
	pW = totalW[id];

	float _x = pX / pW, _y = pY / pW;

	short2 rc = make_short2(tx + (_x - tx) * omega + 0.5f, ty + (_y - ty) * omega + 0.5f);

	rc.x = max(min(rc.x, size - 1), 0);
	rc.y = max(min(rc.y, size - 1), 0);

	if (mask[id] || density[TOID(rc.x, rc.y, size)] == 0) rc = make_short2(tx, ty);

	id = TOID(rc.x, rc.y, size);
	output[id] = rc;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelCalcEnergy(short2 *voronoi, float *density, float *nrgTex, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int id = TOID(tx, ty, size);

	short2 site = voronoi[id];

	float dx = (site.x - tx) * 1.0f / size;
	float dy = (site.y - ty) * 1.0f / size;

	float dist = dx * dx + dy * dy;

	nrgTex[id] = density[id] * dist;
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void kernelReduce(float *input, float *output, unsigned int n)
{
	__shared__ float sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) { sdata[tid] += input[i] + input[i + blockSize]; i += gridSize; }
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) output[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Initialization ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<int,int> m_1, m_2, m_3, m_4;

void gcvtInitialization(int textureSize)
{
	m_1[256] = 4, m_2[256] = 32, m_3[256] = 16;
	m_1[512] = 8, m_2[512] = 32, m_3[512] = 16;
	m_1[1024] = 16, m_2[1024] = 32, m_3[1024] = 16;
	m_1[2048] = 32, m_2[2048] = 32, m_3[2048] = 8;
	m_1[4096] = 64, m_2[4096] = 32, m_3[4096] = 8;
	m_1[8192] = 128, m_2[8192] = 32, m_3[8192] = 4;

    pbaTexSize  = textureSize;

    pbaMemSize  = pbaTexSize * pbaTexSize * sizeof(short2);

    pbaTextures = (short2 **) malloc(2 * sizeof(short2 *));
    pbaDensity  = (float  **) malloc(10 * sizeof(float  *));
    pbaPrefixX  = (float  **) malloc(10 * sizeof(float  *));
    pbaPrefixY  = (float  **) malloc(10 * sizeof(float  *));
    pbaPrefixW  = (float  **) malloc(10 * sizeof(float  *));

    cudaMalloc((void **) &pbaTextures[0],   pbaMemSize);
    cudaMalloc((void **) &pbaTextures[1],   pbaMemSize);
    cudaMalloc((void **) &pbaMargin,        m_1[pbaTexSize] * pbaTexSize * sizeof(short2));
    cudaMalloc((void **) &pbaTotalX,        pbaTexSize * pbaTexSize * sizeof(float)); 
    cudaMalloc((void **) &pbaTotalY,        pbaTexSize * pbaTexSize * sizeof(float)); 
    cudaMalloc((void **) &pbaTotalW,        pbaTexSize * pbaTexSize * sizeof(float)); 
    cudaMalloc((void **) &pbaEnergyTex,     pbaTexSize * pbaTexSize * sizeof(float));
	cudaMalloc((void **) &constrainMask_d,	pbaTexSize * pbaTexSize * sizeof(bool));

    for(int i = 0; i < 10; ++i) {
    if((pbaTexSize>>i) < 256) break;
	cudaMalloc((void **) &pbaDensity[i],    (pbaTexSize>>i) * (pbaTexSize>>i) * sizeof(float));
    cudaMalloc((void **) &pbaPrefixX[i],    (pbaTexSize>>i) * (pbaTexSize>>i) * sizeof(float)); 
    cudaMalloc((void **) &pbaPrefixY[i],    (pbaTexSize>>i) * (pbaTexSize>>i) * sizeof(float)); 
    cudaMalloc((void **) &pbaPrefixW[i],    (pbaTexSize>>i) * (pbaTexSize>>i) * sizeof(float)); 
    }
}

// Deallocate all allocated memory
void pbaCVDDeinitialization()
{
    cudaFree(pbaTextures[0]);
    cudaFree(pbaTextures[1]);
    cudaFree(pbaMargin);

    for(int i = 0; i < 10; ++i) {
    if((pbaTexSize>>i) < 256) break;
    cudaFree(pbaDensity[i]);
    cudaFree(pbaPrefixX[i]);
    cudaFree(pbaPrefixY[i]);
    cudaFree(pbaPrefixW[i]);
    }
    
    cudaFree(pbaTotalX);
    cudaFree(pbaTotalY);
    cudaFree(pbaTotalW);
    cudaFree(pbaEnergyTex);
	cudaFree(constrainMask_d);

    free(pbaTextures);
    free(pbaDensity);
    free(pbaPrefixX);
    free(pbaPrefixY);
    free(pbaPrefixW);
}

// Copy input to GPU 
void pba2DInitializeInput(float *density, bool *mask)
{
    cudaMemcpy(pbaDensity[0], density, pbaTexSize * pbaTexSize * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(constrainMask_d, mask, pbaTexSize * pbaTexSize * sizeof(bool), cudaMemcpyHostToDevice);

    pbaVoronoi = pbaTextures[0];
    pbaTemp    = pbaTextures[1];
    pbaBuffer  = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Parallel Banding Algorithm plus //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

// Phase 1 of PBA. m1 must divides texture size and equal or less than size / 64
void pba2DPhase1(int m1) 
{
    dim3 block = dim3(BLOCKSIZE);   
    dim3 grid = dim3(pbaTexSize / block.x, m1); 

    kernelFloodDown<<< grid, block >>>(pbaTextures[pbaBuffer], pbaTextures[pbaBuffer], pbaTexSize, pbaTexSize / m1); 

    kernelFloodUp<<< grid, block >>>(pbaTextures[pbaBuffer], pbaTextures[pbaBuffer], pbaTexSize, pbaTexSize / m1); 

    kernelPropagateInterband<<< grid, block >>>(pbaTextures[pbaBuffer], pbaMargin, pbaTexSize, pbaTexSize / m1);

    kernelUpdateVertical<<< grid, block >>>(pbaTextures[pbaBuffer], pbaMargin, pbaTextures[1^pbaBuffer], pbaTexSize, pbaTexSize / m1);
}

// Phase 2 of PBA. m2 must divides texture size
void pba2DPhase2(int m2) 
{
    // Compute proximate points locally in each band
    dim3 block = dim3(BLOCKSIZE);
    dim3 grid = dim3(pbaTexSize / block.x, m2);

    kernelProximatePoints<<< grid, block >>>(pbaTextures[1^pbaBuffer], pbaTextures[pbaBuffer], pbaTexSize, pbaTexSize / m2); 

    kernelCreateForwardPointers<<< grid, block >>>(pbaTextures[pbaBuffer], pbaTextures[pbaBuffer], pbaTexSize, pbaTexSize / m2); 

    // Repeatly merging two bands into one
    for (int noBand = m2; noBand > 1; noBand /= 2) {
        grid = dim3(pbaTexSize / block.x, noBand / 2); 
        kernelMergeBands<<< grid, block >>>(pbaTextures[1^pbaBuffer], pbaTextures[pbaBuffer], pbaTextures[pbaBuffer], pbaTexSize, pbaTexSize / noBand); 
    }

    // Replace the forward link with the X coordinate of the seed to remove
    // the need of looking at the other texture. We need it for coloring.
    grid = dim3(pbaTexSize / block.x, pbaTexSize); 
    kernelDoubleToSingleList<<< grid, block >>>(pbaTextures[1^pbaBuffer], pbaTextures[pbaBuffer], pbaTextures[pbaBuffer], pbaTexSize); 
}

// Phase 3 of PBA. m3 must divides texture size and equal or less than 64
void pba2DPhase3(int m3) 
{
    dim3 block = dim3(BLOCKSIZE, m3); 
    dim3 grid = dim3(pbaTexSize / block.x);
    
    kernelColor<<< grid, block >>>(pbaTextures[pbaBuffer], pbaTextures[1^pbaBuffer], pbaTexSize); 
}

void pba2DCompute(int m1, int m2, int m3)
{
    pba2DPhase1(m1);

    pba2DPhase2(m2); 

    pba2DPhase3(m3); 

    pbaVoronoi = pbaTextures[1^pbaBuffer]; 
    pbaTemp = pbaTextures[pbaBuffer]; 
    pbaBuffer = 1^pbaBuffer;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Centroidal Voronoi Tessellation ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void pbaCVDDensityScaling(int k)
{
    dim3 block(BLOCKX, BLOCKY);

    for(int i = 1; i < k; ++i) {
        dim3 grid((pbaTexSize >> i) / block.x, (pbaTexSize >> i) / block.y);
        kernelDensityScaling<<< grid, block >>>(pbaDensity[i - 1], pbaDensity[i], pbaTexSize >> i);
    }
}

void pbaCVDComputeWeightedPrefix(int k) 
{
    dim3 block(BLOCKSIZE); 

	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
    int ns = BLOCKSIZE * 3 * sizeof(float); 
    for(int i = 0; i < k; ++i) {
        dim3 grid(pbaTexSize >> i);
        kernelComputeWeightedPrefixX<<< grid, block, ns >>>(pbaPrefixW[i], pbaPrefixX[i], pbaPrefixY[i], pbaDensity[i], pbaTexSize >> i); 
	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
    }
}

void pbaCVDComputeCentroid()
{
	dim3 block(BLOCKSIZE);
    dim3 grid(pbaTexSize);

    int ns = pbaTexSize * sizeof(short); 

    kernelVoronoi1D<<< grid, block, ns >>>(pbaVoronoi, (int *) pbaTemp, pbaTexSize); 

	kernelTotal_X<<< grid, block >>>(pbaVoronoi, pbaPrefixX[pbaScale], pbaPrefixY[pbaScale], pbaPrefixW[pbaScale], \
                                     pbaTotalX, pbaTotalY, pbaTotalW, pbaTexSize); 

    block = dim3(BLOCKSIZE); 
    grid = dim3(pbaTexSize / block.x, pbaTexSize / block.x);
	kernelScan_Y<<< grid, block >>>((short *) pbaTemp, pbaTotalX, pbaTotalY, pbaTotalW, pbaTexSize);
}

void pbaCVDUpdateSites() 
{
    dim3 block(BLOCKX, BLOCKY); 
    dim3 grid(pbaTexSize / block.x, pbaTexSize / block.y); 

    kernelFillShort<<< grid, block >>>(pbaVoronoi, MARKER, pbaTexSize);

	kernelUpdateSites<<< grid, block >>>((short *) pbaTemp, pbaTotalX, pbaTotalY, pbaTotalW, pbaDensity[pbaScale], \
                                         pbaVoronoi, constrainMask_d, pbaTexSize, pbaOmega);
}

void pbaCVDZoomIn()
{
    dim3 block(BLOCKX, BLOCKY);
    dim3 grid1(pbaTexSize / block.x, pbaTexSize / block.y);
    dim3 grid2((pbaTexSize << 1) / block.x, (pbaTexSize << 1) / block.y);

    kernelFillShort<<< grid2, block >>>(pbaTemp, MARKER, pbaTexSize << 1);

    kernelZoomIn<<< grid1, block >>>(pbaVoronoi, pbaTemp, pbaTexSize, 1);

    pbaBuffer = 1^pbaBuffer;

    short2 *tmp_ptr = pbaVoronoi;
    pbaVoronoi = pbaTemp;
    pbaTemp = tmp_ptr;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Calculate CVT Engergy Function //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

float pbaCVDCalcEnergy()
{
    dim3 block(BLOCKX, BLOCKY);
    dim3 grid(pbaTexSize / block.x, pbaTexSize / block.y);

    kernelCalcEnergy<<< grid, block >>>(pbaVoronoi, pbaDensity[pbaScale], pbaEnergyTex, pbaTexSize);

    const int blockSize = 512;
    int n = pbaTexSize * pbaTexSize;
    int blocksPerGrid;

    do {
        blocksPerGrid = min(int(std::ceil((1.*n) / blockSize)), 32768);
        kernelReduce<blockSize><<< blocksPerGrid, blockSize >>>(pbaEnergyTex, pbaEnergyTex, n);
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1) {
        kernelReduce<blockSize><<< 1, blockSize >>>(pbaEnergyTex, pbaEnergyTex, n);
    }

    cudaMemcpy(&pbaEnergy_h, pbaEnergyTex, sizeof(float), cudaMemcpyDeviceToHost);

    return pbaEnergy_h * powf(2.0, pbaScale * 2.0);
}


int gcvtIterations;
void gCVT(short *Voronoi, float *density_d, bool *mask, int size, int depth, int maxIter)
{
	gcvtInitialization(size);

	for (int i = 0; i < depth; ++i) if ((pbaTexSize >> i) < 256) { depth = i; break; }

	pba2DInitializeInput(density_d, mask);
	pbaCVDDensityScaling(depth);
	pbaCVDComputeWeightedPrefix(depth);

	pbaScale = 0;
	gcvtIterations = 0;
	pbaTexSize >>= depth;

	pbaTexSize <<= 1;
	cudaMemcpy(pbaVoronoi, Voronoi, pbaTexSize * pbaTexSize * sizeof(short2), cudaMemcpyHostToDevice);
	pbaTexSize >>= 1;

	float Energy, lastEnergy = 1e18, diffEnergy, gradientEnergy;
	std::vector <int> switch_iter; switch_iter.clear();

	pbaOmega = 2.0;

	for (pbaScale = depth - 1; ~pbaScale; --pbaScale) {
		pbaTexSize <<= 1;
		do {

			pba2DCompute(m_1[pbaTexSize], m_2[pbaTexSize], m_3[pbaTexSize]);

			if (gcvtIterations % 10 == 0)
				Energy = pbaCVDCalcEnergy();

			pbaCVDComputeCentroid();

			pbaCVDUpdateSites();

			gcvtIterations++;

			if (gcvtIterations % 10 == 0) {
				diffEnergy = lastEnergy - Energy;
				gradientEnergy = diffEnergy / 10.0;

				//printf("Iter %d: %f %f\n", gcvtIterations, Energy, gradientEnergy);

				pbaOmega = min(2.0, 1.0 + diffEnergy);
				if (pbaScale) {
					if (gradientEnergy < 3e-1) break;
				}
				else {
					if (gradientEnergy < THRESHOLD) break;
				}

				lastEnergy = Energy;
			}

		} while (gcvtIterations < maxIter);

		switch_iter.push_back(gcvtIterations);

		if (pbaScale) pbaCVDZoomIn();
	}

	pba2DCompute(m_1[pbaTexSize], m_2[pbaTexSize], m_3[pbaTexSize]);
	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(Voronoi, pbaVoronoi, pbaMemSize, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());

	pbaCVDDeinitialization();
}