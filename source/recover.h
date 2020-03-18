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

#ifndef RECOVER_H
#define RECOVER_H

double dot(double x1, double y1, double x2, double y2) {
	return x1 * x2 + y1 * y2;
}

void barycentric(
	double x1, double y1, double x2, double y2,
	double x3, double y3, double x0, double y0,
	double &w3, double &w1, double &w2
) {
	double v0x = x2 - x1, v0y = y2 - y1;
	double v1x = x3 - x1, v1y = y3 - y1;
	double v2x = x0 - x1, v2y = y0 - y1;

	double d00 = dot(v0x, v0y, v0x, v0y);
	double d01 = dot(v0x, v0y, v1x, v1y);
	double d11 = dot(v1x, v1y, v1x, v1y);
	double d20 = dot(v2x, v2y, v0x, v0y);
	double d21 = dot(v2x, v2y, v1x, v1y);

	double denom = d00 * d11 - d01 * d01;
	if (denom == 0) {
		w1 = w2 = w3 = -1;
		return;
	}
	w1 = (d11 * d20 - d01 * d21) / denom;
	w2 = (d00 * d21 - d01 * d20) / denom;
	w3 = 1.0 - w1 - w2;
}

struct tuple {
	double _w[3];
	tuple() {}
	tuple(double a, double b, double c) {
		_w[0] = a, _w[1] = b, _w[2] = c;
	}
};

template <typename Point, typename Mesh, typename F_location>
bool locate(const Point& p, const Mesh& mesh, F_location& f_loc) {
	typedef typename boost::graph_traits<Mesh2D>::vertex_descriptor			vertex_descriptor;

	for (auto f : mesh.faces()) {
		auto hd = halfedge(f, mesh);
		std::vector<Point> _p;
		for (auto vd : vertices_around_face(hd, mesh)) {
			_p.push_back(mesh.point(vd));
		}

		tuple bc;
		barycentric(_p[0].x(), _p[0].y(), _p[1].x(), _p[1].y(), _p[2].x(), _p[2].y(), p.x(), p.y(), bc._w[0], bc._w[1], bc._w[2]);

		if (bc._w[0] < 0 || bc._w[1] < 0 || bc._w[2] < 0) continue;

		f_loc.first = f;
		f_loc.second = bc;
		return true;
	}
	return false;
}

template <typename Mesh2D, typename SurfMesh, typename SeamMesh, typename Map23, typename PointVec, typename TriVec, typename cPoints>
void recover(
	const Mesh2D& mesh_2d,
	const SeamMesh& mesh_3d,
	const Map23& vertex_2d_to_3d,
	const PointVec& point,
	const TriVec& triangle,
	const cPoints& c_points,
	SurfMesh& resMesh
){
	typedef typename boost::graph_traits<Mesh2D>::vertex_descriptor			vertex_descriptor;
	typedef typename boost::graph_traits<Mesh2D>::halfedge_descriptor		halfedge_descriptor;
	typedef typename boost::graph_traits<Mesh2D>::face_descriptor			face_descriptor;

	std::pair<face_descriptor, tuple> f_loc;
	std::unordered_map<int, vertex_descriptor> vid;
	auto ppmap = get(CGAL::vertex_point, mesh_3d);
	Point_3 _p3[3];
	Point_2 _p2[3];
	int id = 0;

	for (int i = 0; i < point.size() - c_points.size(); ++i) {
		auto pt = point[i];
		locate(Point_2(pt._p[0], pt._p[1]), mesh_2d, f_loc);
		//printf("%f %f %f\n", f_loc.second._w[0], f_loc.second._w[1], f_loc.second._w[2]);

		//std::cout << f_loc.first << std::endl;
		std::vector<Point_3> vec;
		auto hd = halfedge(f_loc.first, mesh_2d);
		for (auto vd : vertices_around_face(hd, mesh_2d)) {
			vec.push_back(get(ppmap, vertex_2d_to_3d[vd]));
		}
		
		double _x = 0.0, _y = 0.0, _z = 0.0;
		for (int j = 0; j < 3; ++j)	_x += vec[j].x() * f_loc.second._w[j];
		for (int j = 0; j < 3; ++j) _y += vec[j].y() * f_loc.second._w[j];
		for (int j = 0; j < 3; ++j) _z += vec[j].z() * f_loc.second._w[j];

		vid[id++] = resMesh.add_vertex(Point_3(_x, _y, _z));
	}

	for (auto v : c_points) {
		vid[id++] = resMesh.add_vertex(get(ppmap, vertex_2d_to_3d[v]));
	}

	for (auto tri : triangle) {
		for (int i = 0; i < 3; ++i) {
			auto pt = point[tri._v[i]];
			_p2[i] = Point_2(pt._p[0], pt._p[1]);
		}

		auto c = CGAL::centroid(_p2[0], _p2[1], _p2[2]);

		if (!locate(c, mesh_2d, f_loc)) continue;
		//const Point_3& p = to_p2(Point_2(c.x(), c.y()));
		//f_loc = PMP::locate_with_AABB_tree(p, tree, mesh_2d);

		//if (!PMP::is_in_face(f_loc.second, mesh_2d)) continue;
		
		std::vector<vertex_descriptor> vec;
		for (int i = 0; i < 3; ++i) {
			vec.push_back(vid[tri._v[i]]);
		}
		resMesh.add_face(vec);
	}

	vid.clear();
}

#endif