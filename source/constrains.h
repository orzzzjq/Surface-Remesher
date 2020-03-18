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

#ifndef CONSTRAINS_H
#define CONSTRAINS_H

#define TOID(x, y, n) (y) * (n) + (x)

template <typename HE, typename Mesh, typename FT>
bool internal_is_sharp(
	const HE& he,
	const Mesh& mesh,
	FT cos_angle
){
	auto f1 = face(he, mesh);
	auto f2 = face(opposite(he, mesh), mesh);

	const typename Kernel::Vector_3& n1 = PMP::compute_face_normal(f1, mesh);
	const typename Kernel::Vector_3& n2 = PMP::compute_face_normal(f2, mesh);

	if (n1 * n2 <= cos_angle) return true;
	else return false;
}

template <typename Edge, typename Mesh, typename FT>
bool is_sharp(
	const Edge& e,
	const Mesh& mesh,
	FT angle_in_deg
){
	FT cos_angle ( std::cos ( CGAL::to_double(angle_in_deg) * CGAL_PI / 180.) );
	auto he = halfedge(e, mesh);
	if (is_border_edge(he, mesh)
		|| angle_in_deg == FT()
		|| (angle_in_deg != FT(180) && internal_is_sharp(he, mesh, cos_angle))) return true;
	else return false;
}

template <typename Mesh, typename Map, typename Earr, typename Varr>
void detect_constrains(
	const Mesh& mesh,
	const Map& map,
	Earr& c_edge,
	Varr& c_point
){
	for (auto e : mesh.edges()) {
		if (is_border(e, mesh) /*|| is_sharp(e, mesh, 60.0)*/) {
			auto he = halfedge(e, mesh);
			auto s = source(he, mesh);
			auto t = target(he, mesh);
			c_edge.push_back({map[s], map[t]});
			c_point.insert(map[s]);
			c_point.insert(map[t]);
		}
	}
}

#endif