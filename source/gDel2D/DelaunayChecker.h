#pragma once 

#include "gDel2D/GpuDelaunay.h"
#include "gDel2D/CPU/PredWrapper.h"

class DelaunayChecker
{
private: 
	GDel2DInput&  _input; 
    GDel2DOutput& _output; 

	PredWrapper2D _predWrapper; 

    int getVertexCount();
    int getSegmentCount();
    int getTriangleCount();

public: 
    DelaunayChecker( GDel2DInput& input, GDel2DOutput& output ); 

    void checkEuler();
    void checkAdjacency();
    void checkOrientation();
    void checkDelaunay();
    void checkConstraints(); 
}; 

const int TriSegNum = 3;
const int TriSeg[ TriSegNum ][2] = {
    { 0, 1 },
    { 1, 2 },
    { 2, 0 }
};

