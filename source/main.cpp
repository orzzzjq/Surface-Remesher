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

#include <CGAL/Surface_mesh_parameterization/Discrete_conformal_map_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Discrete_authalic_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Square_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Two_vertices_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/internal/Containers_filler.h>
#include <CGAL/Surface_mesh_parameterization/ARAP_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/LSCM_parameterizer_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Surface_mesh_parameterization/parameterize.h>
#include <CGAL/Surface_mesh_parameterization/IO/File_off.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/smooth_mesh.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/locate.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <boost/property_map/property_map.hpp>
#include <boost/function_output_iterator.hpp>
#include <CGAL/boost/graph/Seam_mesh.h>
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/disable_warnings.h>
#include <CGAL/Simple_cartesian.h>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <CGAL/Unique_hash_map.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/circulator.h>
#include <CGAL/AABB_tree.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>

#include "gDel2D/gDel2D/GpuDelaunay.h"
#include "gDel2D/gDel2D/PerfTimer.h"
#include "gDel2D/gDel2D/CPU/PredWrapper.h"
#include "gDel2D/DelaunayChecker.h"
#include "delaunay.h"

#if defined(_WIN32)
#include <Windows.h>
#endif

//#define OUTPUT_PARAMETERIZATION
//#define VISUALIZE_TRIANGULATION
#ifdef VISUALIZE_TRIANGULATION
#include "gDel2D/Visualizer.h"
#endif

typedef CGAL::Simple_cartesian<double>	Kernel;

#include "split.h"
#include "weighting.h"
#include "triangulation.h"
#include "discretization.h"
#include "constrains.h"
#include "gcvt.h"
#include "recover.h"

typedef Kernel::Point_2					Point_2;
typedef Kernel::Point_3					Point_3;
typedef CGAL::Surface_mesh<Point_3>		SurfMesh;
typedef CGAL::Surface_mesh<Point_2>		Mesh2D;

typedef boost::graph_traits<SurfMesh>::edge_descriptor			SM_edge_descriptor;
typedef boost::graph_traits<SurfMesh>::halfedge_descriptor		SM_halfedge_descriptor;
typedef boost::graph_traits<SurfMesh>::vertex_descriptor		SM_vertex_descriptor;

typedef boost::graph_traits<Mesh2D>::edge_descriptor			edge_2d;
typedef boost::graph_traits<Mesh2D>::halfedge_descriptor		halfedge_2d;
typedef boost::graph_traits<Mesh2D>::vertex_descriptor			vertex_2d;

typedef CGAL::Unique_hash_map<SM_halfedge_descriptor, Point_2>	UV_uhm;
typedef CGAL::Unique_hash_map<SM_edge_descriptor, bool>			Seam_edge_uhm;
typedef CGAL::Unique_hash_map<SM_vertex_descriptor, bool>		Seam_vertex_uhm;

typedef boost::associative_property_map<UV_uhm>					UV_pmap;
typedef boost::associative_property_map<Seam_edge_uhm>			Seam_edge_pmap;
typedef boost::associative_property_map<Seam_vertex_uhm>		Seam_vertex_pmap;

typedef CGAL::Seam_mesh<SurfMesh, Seam_edge_pmap, Seam_vertex_pmap>	SeamMesh;

typedef boost::graph_traits<SeamMesh>::vertex_descriptor		vertex_descriptor;
typedef boost::graph_traits<SeamMesh>::edge_descriptor			edge_descriptor;
typedef boost::graph_traits<SeamMesh>::halfedge_descriptor		halfedge_descriptor;
typedef boost::graph_traits<SeamMesh>::face_descriptor			face_descriptor;

typedef boost::unordered_map<vertex_descriptor, vertex_2d>		_3d_to_2d_uhm;
typedef boost::unordered_map<vertex_2d, vertex_descriptor>		_2d_to_3d_uhm;
typedef boost::unordered_map<vertex_2d, double>					weight_uhm;

typedef boost::associative_property_map<_3d_to_2d_uhm>			_3d_to_2d_pmap;
typedef boost::associative_property_map<_2d_to_3d_uhm>			_2d_to_3d_pmap;
typedef boost::associative_property_map<weight_uhm>				weight_pmap;

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMP = CGAL::Surface_mesh_parameterization;

typedef PMP::Face_location<SurfMesh, Kernel::FT>				Face_Location;

int imageSize	= 2048;
int depth		= 1;
int maxIter		= 1000;
int vertices    = 20000;
double scale, leftbound, lowerbound;
float *densityMap;
short *Voronoi;
bool *constrainMask;

GDel2DInput		DelInput;
GDel2DOutput	DelOutput;

int main(int argc, char** argv)
{
	// input surface mesh
	const char* filename = argc > 1 ? argv[1] : "data/horse.off";
	std::ifstream input(filename);
	SurfMesh surface;
	if (!input || !(input >> surface) || surface.is_empty()) {
		std::cerr << "Not a valid .off file." << std::endl;
		return EXIT_FAILURE;
	}
	input.close();

	double edge_length_limit = 0.72;
	
	// split long edges
	std::vector<SM_edge_descriptor> constrain_vec;
	detectConstrains(surface, constrain_vec);
	PMP::split_long_edges(constrain_vec, edge_length_limit, surface);

	// create seam mesh
	const char* seamfile = "data/horse.selection.txt";
	Seam_edge_uhm seam_edge_uhm(false);
	Seam_edge_pmap seam_edges(seam_edge_uhm);
	Seam_vertex_uhm seam_vertex_uhm(false);
	Seam_vertex_pmap seam_vertices(seam_vertex_uhm);
	std::vector<SM_edge_descriptor> seam_vec;
	addSeams(surface, seam_edges, seam_vec, seamfile);
	PMP::split_long_edges(seam_vec, edge_length_limit, surface, PMP::parameters::edge_is_constrained_map(seam_edges));
	getSeamVertices(surface, seam_edges, seam_vertices);

	SeamMesh seam_mesh(surface, seam_edges, seam_vertices);

	// the property map stores the 2D points in the uv parameter space
	UV_uhm uv_uhm;
	UV_pmap uv_coord(uv_uhm);

	// a halfedge on the border
	halfedge_descriptor border_halfedge = PMP::longest_border(seam_mesh).first;

	// planar parameterization
	//typedef SMP::Square_border_uniform_parameterizer_3<SeamMesh> Border_parameterizer;
	typedef SMP::ARAP_parameterizer_3<SeamMesh> Parameterizer;
	SMP::parameterize(seam_mesh, Parameterizer(), border_halfedge, uv_coord);
	printf("Parameterization done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);
#ifdef OUTPUT_PARAMETERIZATION
	std::ofstream out("parameterization.off");
	SMP::IO::output_uvmap_to_off(seam_mesh, border_halfedge, uv_coord, out);
#endif

	// create 2d triangulation
	Mesh2D mesh_2d;
	_3d_to_2d_uhm _32_uhm;
	_3d_to_2d_pmap vertex_3d_to_2d(_32_uhm);
	_2d_to_3d_uhm _23_uhm;
	_2d_to_3d_pmap vertex_2d_to_3d(_23_uhm);
	build_2d_triangulation(seam_mesh, border_halfedge, uv_coord, mesh_2d, vertex_3d_to_2d, vertex_2d_to_3d);
	printf("Triangulation done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);

	// weighting vertices
	weight_uhm wgt_uhm;
	weight_pmap vertex_weight(wgt_uhm);
	weighting_vertices_with_area(seam_mesh, mesh_2d, vertex_3d_to_2d, vertex_weight);
	printf("Weighting done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);
	
	// discretization
	densityMap = (float *)malloc(imageSize * imageSize * sizeof(float));
	discretization(mesh_2d, vertex_weight, densityMap, imageSize, scale, leftbound, lowerbound);
	printf("Discretization done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);

	// detect constrains
	std::vector<std::pair<vertex_2d, vertex_2d> > constrain_edge;
	std::unordered_set<vertex_2d> constrain_point;
	detect_constrains(seam_mesh, vertex_3d_to_2d, constrain_edge, constrain_point);
	printf("Detect constrains done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);

	// construct centroidal Voronoi diagram
	Voronoi = (short *)malloc(sizeof(short) * imageSize * imageSize * 2);
	constrainMask = (bool *)malloc(sizeof(bool) * imageSize * imageSize);
	generateMask(mesh_2d, constrain_point, constrainMask, imageSize, scale, leftbound, lowerbound);
	centroidalVoronoi(Voronoi, densityMap, constrainMask, vertices, imageSize, depth, maxIter);
	printf("Centroidal Voronoi tessellation done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);

	// construct constraint Delaunay triangulation
#ifdef VISUALIZE_TRIANGULATION
	Visualizer *vis = Visualizer::instance();
	if (vis->isEnable()) {
		vis->init(argc, argv, "gDel2D");
		vis->printHelp();
	}
	Visualizer::instance()->pause();
#endif
	GpuDel gpuDel;
	Point2HVec().swap(DelInput.pointVec);
	SegmentHVec().swap(DelInput.constraintVec);
	TriHVec().swap(DelOutput.triVec);
	TriOppHVec().swap(DelOutput.triOppVec);
	delaunayInput(mesh_2d, Voronoi, constrainMask, constrain_point, constrain_edge, DelInput.pointVec, DelInput.constraintVec, imageSize, scale, leftbound, lowerbound);
	//printf("%d points, %d segments", DelInput.pointVec.size(), DelInput.constraintVec.size());
	gpuDel.compute(DelInput, &DelOutput);
	printf("Constraint Delaunay triangulation done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);

	// Parameter space -> surface
	SurfMesh resultMesh;
	recover(mesh_2d, seam_mesh, vertex_2d_to_3d, DelInput.pointVec, DelOutput.triVec, constrain_point, resultMesh);
	printf("Recover done. %f s\n", clock()*1.0 / CLOCKS_PER_SEC);

	// compute average edge length
	double edge_sum = 0, edge_avg = 0;
	for (auto e : resultMesh.edges()) {
		edge_sum += PMP::edge_length(e, resultMesh);
	}
	edge_avg = edge_sum / resultMesh.number_of_edges();
	printf("Average edge length: %f\n", edge_sum / resultMesh.number_of_halfedges());

	// Output to off file
	std::ofstream ostream("result.off");
	ostream << resultMesh;

	free(densityMap);
	free(Voronoi);
	free(constrainMask);

#ifdef VISUALIZE_TRIANGULATION
	// Visualize the constraint Delaunay triangualtion
	Visualizer::instance()->resume();
	Visualizer::instance()->addFrame(DelInput.pointVec, DelInput.constraintVec, DelOutput.triVec);
	vis->run();
#endif

	return EXIT_SUCCESS;
}
