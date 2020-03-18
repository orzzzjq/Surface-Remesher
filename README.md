<p align="center">
  <img src="picture/_.jpg">
</p>
Surface Remesher remeshes a surface mesh using the centroidal Voronoi tessellation. The input is a triangulated surface mesh. We first cut the surface into a topological disk, then parameterize it in a planar space. We compute the centroidal Voronoi tessellation (CVT) in the parameter space with respect to a density distribution, and construct the constrained Delaunay triangulation (CDT) from the resulted CVT. The final optimized surface mesh is then obtained from CDT.


We use these packages:
1. CGAL 5.0.2 to store the mesh structure,
2. with Eigen 3.3.7 to parameterize the surface mesh;
3. gCVT to compute the centroidal Voronoi tessellation on the GPU;
4. gDel2D to compute the constrained Delaunay triangulation on the GPU.
