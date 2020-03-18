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

#ifndef WEIGHTING_H
#define WEIGHTING_H

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMP = CGAL::Surface_mesh_parameterization;

double cross(const Kernel::Point_2& A, const Kernel::Point_2& B) {
	return A.x() * B.y() - B.x() * A.y();
}

double get_area(const Kernel::Point_2& A, const Kernel::Point_2& B, const Kernel::Point_2& C) {
	return fabs(cross(Kernel::Point_2(B.x() - A.x(), B.y() - A.y()), Kernel::Point_2(C.x() - A.x(), C.y() - A.y()))) / 2.;
}

template <typename Mesh3D, typename Mesh2D, typename _32_map, typename weight_map>
void weighting_vertices_with_area(
	const Mesh3D& mesh_3d,
	const Mesh2D& mesh_2d,
	const _32_map& vertex_3d_to_2d,
	weight_map& vertex_weight
){
	typedef typename boost::graph_traits<Mesh3D>::vertex_descriptor		vertex_3d;
	typedef typename boost::graph_traits<Mesh3D>::halfedge_descriptor	halfedge_3d;
	typedef typename boost::graph_traits<Mesh3D>::face_descriptor		face_3d;
	typedef typename boost::graph_traits<Mesh2D>::vertex_descriptor		vertex_2d;
	typedef typename boost::graph_traits<Mesh2D>::halfedge_descriptor	halfedge_2d;
	typedef typename boost::graph_traits<Mesh2D>::face_descriptor		face_2d;

	boost::unordered_map<vertex_2d, double> area_3d, area_2d;

	for (face_3d f3d : mesh_3d.faces()) {
		double area = PMP::face_area(f3d, mesh_3d);

		halfedge_3d h3d = halfedge(f3d, mesh_3d);
		for (vertex_3d v3d : vertices_around_face(h3d, mesh_3d)) {
			area_3d[get(vertex_3d_to_2d, v3d)] += area;
		}
	}

	for (face_2d f2d : mesh_2d.faces()) {
		std::vector<vertex_2d> vec;
		halfedge_2d h2d = halfedge(f2d, mesh_2d);
		for (vertex_2d v2d : vertices_around_face(h2d, mesh_2d)) {
			vec.push_back(v2d);
		}
		
		double area = get_area(mesh_2d.point(vec[0]), mesh_2d.point(vec[1]), mesh_2d.point(vec[2]));

		for (vertex_2d v2d : vertices_around_face(h2d, mesh_2d)) {
			area_2d[v2d] += area;
		}
	}

	double inf = 1e10, mi = 1e10, mx = 0;
	std::vector<vertex_2d> inf_vex;
	for (vertex_2d v2d : mesh_2d.vertices()) {
		double weight = area_3d[v2d] / area_2d[v2d];
		if (weight > inf) {
			inf_vex.push_back(v2d);
			continue;
		}

		put(vertex_weight, v2d, weight);
		if (weight > mx) mx = weight;
		if (weight < mi) mi = weight;
	}

	for (auto it : inf_vex) {
		vertex_weight[it] = mx;
	}

	area_2d.clear();
	area_3d.clear();
	inf_vex.clear();
}

template <typename P>
double mydot(const P& A, const P& B) {
	return A.x() * B.x() + A.y() * B.y() + A.z() * B.z();
}

template <typename P>
double mysquare(const P& A) {
	return A.x() * A.x() + A.y() * A.y() + A.z() * A.z();
}

template <typename Edge, typename Mesh, typename Ppmap>
double calc_curvature(const Edge& e, const Mesh& mesh, const Ppmap& ppmap) {
	auto v1 = source(halfedge(e, mesh), mesh);
	auto v2 = target(halfedge(e, mesh), mesh);
	auto p1 = get(ppmap, v1);
	auto p2 = get(ppmap, v2);
	auto n1 = PMP::compute_vertex_normal(v1, mesh);
	auto n2 = PMP::compute_vertex_normal(v2, mesh);
	
	return fabs(mydot(n2 - n1, p2 - p1) / mysquare(p2 - p1));
}

template <typename Mesh3D, typename Mesh2D, typename _32_map, typename weight_map>
void weighting_vertices_with_curvature(
	const Mesh3D& mesh_3d,
	const Mesh2D& mesh_2d,
	const _32_map& vertex_3d_to_2d,
	weight_map& vertex_weight
){
	typedef typename boost::graph_traits<Mesh3D>::vertex_descriptor		vertex_3d;
	typedef typename boost::graph_traits<Mesh3D>::halfedge_descriptor	halfedge_3d;
	typedef typename boost::graph_traits<Mesh3D>::face_descriptor		face_3d;
	typedef typename boost::graph_traits<Mesh2D>::vertex_descriptor		vertex_2d;
	typedef typename boost::graph_traits<Mesh2D>::halfedge_descriptor	halfedge_2d;
	typedef typename boost::graph_traits<Mesh2D>::face_descriptor		face_2d;

	boost::unordered_map<vertex_2d, double> curvature, count;
	auto ppmap = get(CGAL::vertex_point, mesh_3d);

	for (auto e : mesh_3d.edges()) {
		auto v1 = source(halfedge(e, mesh_3d), mesh_3d);
		auto v2 = target(halfedge(e, mesh_3d), mesh_3d);

		auto w = calc_curvature(e, mesh_3d, ppmap);
		curvature[vertex_3d_to_2d[v1]] += w;
		curvature[vertex_3d_to_2d[v2]] += w;
		count[vertex_3d_to_2d[v1]] += 1.0;
		count[vertex_3d_to_2d[v2]] += 1.0;
	}

	for (auto v : mesh_3d.vertices()) {
		vertex_weight[vertex_3d_to_2d[v]] = curvature[vertex_3d_to_2d[v]] / count[vertex_3d_to_2d[v]];
	}

	curvature.clear();
	count.clear();
}

template <typename Mesh3D, typename Mesh2D, typename _32_map, typename weight_map>
void weighting_vertices_combine(
	const Mesh3D& mesh_3d,
	const Mesh2D& mesh_2d,
	const _32_map& vertex_3d_to_2d,
	weight_map& vertex_weight
) {
	typedef typename boost::graph_traits<Mesh3D>::vertex_descriptor		vertex_3d;
	typedef typename boost::graph_traits<Mesh3D>::halfedge_descriptor	halfedge_3d;
	typedef typename boost::graph_traits<Mesh3D>::face_descriptor		face_3d;
	typedef typename boost::graph_traits<Mesh2D>::vertex_descriptor		vertex_2d;
	typedef typename boost::graph_traits<Mesh2D>::halfedge_descriptor	halfedge_2d;
	typedef typename boost::graph_traits<Mesh2D>::face_descriptor		face_2d;

	boost::unordered_map<vertex_2d, double> curvature, count;
	auto ppmap = get(CGAL::vertex_point, mesh_3d);

	for (auto e : mesh_3d.edges()) {
		auto v1 = source(halfedge(e, mesh_3d), mesh_3d);
		auto v2 = target(halfedge(e, mesh_3d), mesh_3d);

		auto w = calc_curvature(e, mesh_3d, ppmap);
		curvature[vertex_3d_to_2d[v1]] += w;
		curvature[vertex_3d_to_2d[v2]] += w;
		count[vertex_3d_to_2d[v1]] += 1.0;
		count[vertex_3d_to_2d[v2]] += 1.0;
	}

	boost::unordered_map<vertex_2d, double> area_3d, area_2d;
	
	for (face_3d f3d : mesh_3d.faces()) {
		double area = PMP::face_area(f3d, mesh_3d);

		halfedge_3d h3d = halfedge(f3d, mesh_3d);
		for (vertex_3d v3d : vertices_around_face(h3d, mesh_3d)) {
			area_3d[get(vertex_3d_to_2d, v3d)] += area;
		}
	}

	for (face_2d f2d : mesh_2d.faces()) {
		std::vector<vertex_2d> vec;
		halfedge_2d h2d = halfedge(f2d, mesh_2d);
		for (vertex_2d v2d : vertices_around_face(h2d, mesh_2d)) {
			vec.push_back(v2d);
		}

		double area = get_area(mesh_2d.point(vec[0]), mesh_2d.point(vec[1]), mesh_2d.point(vec[2]));

		for (vertex_2d v2d : vertices_around_face(h2d, mesh_2d)) {
			area_2d[v2d] += area;
		}
	}

	double inf = 1e10, eps = 1e-3, mi = 1e10, mx = 0;
	double alpha = 0.01, beta = 1.0;
	for (auto v3d : mesh_3d.vertices()) {
		auto v2d = vertex_3d_to_2d[v3d];
		double area = area_3d[v2d] / area_2d[v2d];
		double curv = curvature[v2d] / count[v2d];
		double weight = alpha * area + beta * curv;
		
		if (weight < inf && weight > eps) {
			if (weight > mx) mx = weight;
			if (weight < mi) mi = weight;
		}

		put(vertex_weight, v2d, weight);
	}

	for (auto v3d : mesh_3d.vertices()) {
		auto v2d = vertex_3d_to_2d[v3d];
		if (vertex_weight[v2d] > mx) vertex_weight[v2d] = mx;
		if (vertex_weight[v2d] < mi) vertex_weight[v2d] = mi;
	}

	curvature.clear();
	count.clear();
	area_2d.clear();
	area_3d.clear();
}


#endif
