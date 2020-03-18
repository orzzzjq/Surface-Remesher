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

#ifndef DELAUNAY_H
#define DELAUNAY_H

//delaunayInput(mesh_2d, Voronoi, constrainMask, constrain_point, constrain_edge, DelInput.pointVec, DelInput.constraintVec, scale, leftbound, lowerbound);
#define TOID(x, y, n)  ((y) * (n) + (x))

template <typename Mesh, typename Varr, typename Earr, typename delPoint, typename delSeg>
void delaunayInput(
	const Mesh& mesh,
	short *Voronoi,
	bool *mask,
	const Varr& c_point,
	const Earr& c_edge,
	delPoint& inPoint,
	delSeg& inSeg,
	int imageSize,
	double scale,
	double l,
	double b
){
	typedef boost::graph_traits<SurfMesh>::vertex_descriptor		vertex;

	Point2 pt;

	//printf("%f %f\n\n", l, b);
	for (int i = 0; i < imageSize; ++i) {
		for (int j = 0; j < imageSize; ++j) {
			if (!mask[TOID(i, j, imageSize)]
				&& Voronoi[TOID(i, j, imageSize) * 2] == i
				&& Voronoi[TOID(i, j, imageSize) * 2 + 1] == j) {
				pt._p[0] = i * scale + l;
				pt._p[1] = j * scale + b;
				inPoint.push_back(pt);
			}
		}
	}

	std::unordered_map<vertex, int> vid;

	for (auto v : c_point) {
		auto p = mesh.point(v);
		pt._p[0] = p.x();
		pt._p[1] = p.y();
		inPoint.push_back(pt);
		vid[v] = inPoint.size() - 1;
	}

	Segment seg;

	for (auto e : c_edge) {
		seg._v[0] = vid[e.first];
		seg._v[1] = vid[e.second];
		inSeg.push_back(seg);
	}

	vid.clear();
}

#endif
