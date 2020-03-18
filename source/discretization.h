// Autho//MIT License
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

#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

//#define visual_density_map
#ifdef visual_density_map
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

void visual_density(float *density, int n)
{
	Mat img(n, n, CV_32F, Scalar(1));

	float mx = 0, avg = 0, cnt = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (mx < density[j*n + i]) {
				mx = density[j*n + i];
			}
			if (density[j*n + i] != 0) {
				avg += density[j*n + i];
				cnt += 1.0;
			}
			//printf("%f\n", density[j*n + i]);
		}
	}
	avg /= cnt;
	printf("max %f  avg %f\n", mx, avg);

	//mx /= 100.0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			img.at<float>(i, j) = std::min(float(1.0), float((mx - density[j*n + i]) / mx));
			//printf("%f\n", img.at<float>(i, j));
		}
	}
	img.convertTo(img, CV_8U, 255);
	imwrite("density_map.jpg", img);
}
#endif

extern void discretization_d(double *points, double *weight, int num_point, int *triangle, int num_tri,
								float *density, double scale, int n);

template <typename Mesh, typename WtMap>
void discretization(
	Mesh& mesh,
	WtMap& wt,
	float* density,
	int n,
	double& scale,
	double& left,
	double& lower
){
	typedef typename boost::graph_traits<Mesh>::vertex_descriptor	vertex_2d;
	typedef typename boost::graph_traits<Mesh>::halfedge_descriptor	halfedge_2d;
	typedef typename boost::graph_traits<Mesh>::face_descriptor		face_2d;


	double *points = (double *)malloc(2 * mesh.number_of_vertices() * sizeof(double));
	double *weight = (double *)malloc(mesh.number_of_vertices() * sizeof(double));
	int *triangle = (int *)malloc(3 * mesh.number_of_faces() * sizeof(int));
	int num_tri = 0, num_point = 0;
	boost::unordered_map<vertex_2d, int> vertex_id;
	double u = -1e8, b = 1e8, l = 1e8, r = -1e8;
	for (auto v : mesh.vertices()) {
		Point_2 p = mesh.point(v);
		u = u > p.y() ? u : p.y();
		b = b < p.y() ? b : p.y();
		l = l < p.x() ? l : p.x();
		r = r > p.x() ? r : p.x();
	}

	left = l, lower = b;
	if (u - b > r - l) scale = (u - b) / (n - 1);
	else scale = (r - l) / (n - 1);

	for (auto v : mesh.vertices()) {
		Point_2 p = mesh.point(v);
		points[num_point << 1] = p.x() - l;
		points[num_point<<1|1] = p.y() - b;
		weight[num_point] = wt[v];
		vertex_id[v] = num_point;
		num_point++;
	}

	for (auto f : mesh.faces()) {
		auto h = halfedge(f, mesh);
		int off = 0;
		for (auto v : vertices_around_face(h, mesh)) {
			triangle[num_tri * 3 + (off++)] = vertex_id[v];
		}
		num_tri++;
	}

	discretization_d(points, weight, mesh.number_of_vertices(), triangle, mesh.number_of_faces(), density, scale, n);

	free(points);
	free(weight);
	free(triangle);
#ifdef visual_density_map
	visual_density(density, n);
#endif
}

#endif