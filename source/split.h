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

#ifndef SPLIT_H
#define SPLIT_H

#include "constrains.h"

template <typename Mesh, typename SeamEdge, typename SeamVec, typename SeamFile>
void addSeams(
	const Mesh& surface,
	SeamEdge& seam_edges,
	SeamVec& seam_vec,
	SeamFile& seamfile
){
	typedef boost::graph_traits<SurfMesh>::vertex_descriptor		vertex_descriptor;

	std::ifstream in(seamfile);
	std::vector<vertex_descriptor> vertexVec;

	for (auto v : surface.vertices())
		vertexVec.push_back(v);

	int s, t;
	while (in >> s >> t) {
		auto v1 = vertexVec[s];
		auto v2 = vertexVec[t];
		auto tmed = CGAL::edge(v1, v2, surface);
		if (!tmed.second) continue;
		if (!CGAL::is_border(tmed.first, surface)) {
			if (get(seam_edges, tmed.first) == true) continue;
			put(seam_edges, tmed.first, true);
			seam_vec.push_back(tmed.first);
		}
	}

	vertexVec.clear();
	in.close();
}

template <typename Mesh, typename SeamEdge, typename SeamVertex>
void getSeamVertices(
	const Mesh& surface,
	const SeamEdge& seam_edges,
	SeamVertex& seam_vertices
){
	for (auto e : surface.edges()) {
		if (seam_edges[e] == true) {
			auto h = halfedge(e, surface);
			seam_vertices[source(h, surface)] = true;
			seam_vertices[target(h, surface)] = true;
		}
	}
}

template <typename Mesh, typename cVec>
void detectConstrains(
	const Mesh& surface,
	cVec& vec
){
	for (auto e : surface.edges()) {
		if (is_border(e, surface)/* || is_sharp(e, surface, 60.0)*/) {
			vec.push_back(e);
		}
	}
}

#endif
