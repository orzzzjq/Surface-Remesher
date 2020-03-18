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

#ifndef GCVT_H
#define GCVT_H

#define TOID(x, y, n)  ((y) * (n) + (x))
#define MARKER	-32768

extern void gCVT(short *Voronoi, float *density_d, bool *mask, int size, int depth, int maxIter);

//#define visual_initialization
#ifdef visual_initialization
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

void visual_points(short *Voronoi, int n)
{
	Mat img(n, n, CV_32F, Scalar(1));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Voronoi[TOID(i, j, n) * 2] == i && Voronoi[TOID(i, j, n) * 2 + 1] == j) {
				img.at<float>(i, j) = 0.0;
			} else {
				img.at<float>(i, j) = 1.0;
			}
		}
	}
	img.convertTo(img, CV_8U, 255);
	imwrite("initial_sites.jpg", img);
}
#endif

// Random Point Generator
// Random number generator, obtained from http://oldmill.uchicago.edu/~wilder/Code/random/
unsigned long z, w, jsr, jcong; // Seeds
void randinit(unsigned long x_)
{z = x_; w = x_; jsr = x_; jcong = x_;}
unsigned long znew()
{return (z = 36969 * (z & 0xfffful) + (z >> 16));}
unsigned long wnew()
{return (w = 18000 * (w & 0xfffful) + (w >> 16));}
unsigned long MWC()
{return ((znew() << 16) + wnew());}
unsigned long SHR3()
{jsr ^= (jsr << 17); jsr ^= (jsr >> 13); return (jsr ^= (jsr << 5));}
unsigned long CONG()
{return (jcong = 69069 * jcong + 1234567);}
unsigned long rand_int()         // [0,2^32-1]
{return ((MWC() ^ CONG()) + SHR3());}
double random()     // [0,1)
{return ((double)rand_int() / (double(ULONG_MAX) + 1));}

void randomPoints(
	short *Voronoi,
	float *density,
	int num,
	int size
){
	double mx = 0, avg = 0, cnt = 0;
	for (int i = 0; i < size * size; ++i) {
		if (density[i] > mx) mx = density[i];
		if (density[i] != 0) {
			cnt += 1;
			avg += density[i];
		}
	}
	mx = std::min(mx, avg/cnt * 100.);

	int x, y;
	double z;
	for (int i = 0; i < num; ++i) {
		do {
			x = random() * size;
			y = random() * size;
			z = random() * mx;
		} while (Voronoi[TOID(x, y, size) * 2] != MARKER 
			|| density[TOID(x, y, size)] <= z);
		Voronoi[TOID(x, y, size) * 2    ] = x;
		Voronoi[TOID(x, y, size) * 2 + 1] = y;
	}
}

void putConstrains(
	short *Voronoi,
	bool *mask,
	int n
){
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (mask[TOID(i, j, n)]) {
				Voronoi[TOID(i, j, n) * 2    ] = i;
				Voronoi[TOID(i, j, n) * 2 + 1] = j;
			}
			else {
				Voronoi[TOID(i, j, n) * 2] = Voronoi[TOID(i, j, n) * 2 + 1] = MARKER;
			}
		}
	}
}

void centroidalVoronoi(
	short *Voronoi,
	float *density,
	bool *constrainMask,
	int vertices,
	int imageSize,
	int depth,
	int maxIter
){
	putConstrains(Voronoi, constrainMask, imageSize);
	randomPoints(Voronoi, density, vertices, imageSize);
#ifdef visual_initialization
	visual_points(Voronoi, imageSize);
#endif
	gCVT(Voronoi, density, constrainMask, imageSize, depth, maxIter);
}

//generateMask(mesh_2d, constrain_point, constrainMask, imageSize, scale, leftbound, lowerbound);
template <typename Mesh, typename Varr, typename real>
void generateMask(
	const Mesh& mesh,
	const Varr& points,
	bool *mask,
	int imageSize,
	real scale,
	real l,
	real b
){
	memset(mask, 0, sizeof(mask));
	for (auto v : points) {
		auto p = mesh.point(v);
		int x = int((p.x() - l) / scale);
		int y = int((p.y() - b) / scale);
		mask[TOID(x, y, imageSize)] = true;
	}
}

#endif