#pragma once

#include "gDel2D/CommonTypes.h"
#include "HashTable.h"
#include "RandGen.h"

const int GridSize = 8192;

enum Distribution
{
    UniformDistribution,
    GaussianDistribution,
    DiskDistribution,
    ThinCircleDistribution,
    CircleDistribution,
    GridDistribution,
    EllipseDistribution,
    TwoLineDistribution
};

const std::string DistStr[] =
{
    "Uniform",
    "Gaussian",
    "Disk",
    "ThinCircle",
    "Circle",
    "Grid", 
    "Ellipse",
    "TwoLines"
};

typedef HashTable< Point2, int > PointTable; 

class InputCreator 
{
private: 
    RandGen randGen;

    void randCirclePoint( double radius, double &x, double &y );

public: 
    void makePoints( 
        int             pointNum, 
        Distribution    dist, 
        Point2HVec&     pointVec, 
        int             seed = 0 
        );
    void readPoints( 
        std::string     inFilename, 
        Point2HVec&     pointVec,
        SegmentHVec&    constraintVec,
        int             maxConstraintNum = -1
        );
};