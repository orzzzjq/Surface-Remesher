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

#ifndef TRIANGULATION_H
#define TRIANGULATION_H

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMP = CGAL::Surface_mesh_parameterization;

template <typename Mesh3D, typename Mesh2D, typename VertexUVMap, typename _32_map, typename _23_map>
void build_2d_triangulation(
	const Mesh3D& mesh_3d,
	typename boost::graph_traits<Mesh3D>::halfedge_descriptor bhd,
	const VertexUVMap uvmap,
	Mesh2D& mesh_2d,
	_32_map& vertex_3d_to_2d,
	_23_map& vertex_2d_to_3d)
{
	typedef typename boost::graph_traits<Mesh3D>::vertex_descriptor		vertex_3d;
	typedef typename boost::graph_traits<Mesh3D>::halfedge_descriptor	halfedge_3d;
	typedef typename boost::graph_traits<Mesh3D>::face_descriptor		face_3d;
	typedef typename boost::graph_traits<Mesh2D>::vertex_descriptor		vertex_2d;
	typedef typename boost::graph_traits<Mesh2D>::halfedge_descriptor	halfedge_2d;
	typedef typename boost::graph_traits<Mesh2D>::face_descriptor		face_2d;

	boost::unordered_set<vertex_3d> vertices;
	std::vector<face_3d> faces;

	SMP::internal::Containers_filler<Mesh3D> fc(mesh_3d, vertices, &faces);
	PMP::connected_component(
		face(opposite(bhd, mesh_3d), mesh_3d),
		mesh_3d,
		boost::make_function_output_iterator(fc));

	// add vertices
	for (vertex_3d vd3 : vertices) {
		Point_2 tmp = get(uvmap, vd3);
		vertex_2d vd2 = mesh_2d.add_vertex(Point_2(tmp.x(), tmp.y()));
		put(vertex_3d_to_2d, vd3, vd2);
		put(vertex_2d_to_3d, vd2, vd3);
	}

	// add faces
	for (face_3d fd : faces) {
		halfedge_3d hd = halfedge(fd, mesh_3d);
		std::vector<vertex_2d> vec;
		for (vertex_3d vd3 : vertices_around_face(hd, mesh_3d)) {
			vec.push_back(get(vertex_3d_to_2d, vd3));
		}
		mesh_2d.add_face(vec);
	}

	vertices.clear();
	faces.clear();
}

#endif
