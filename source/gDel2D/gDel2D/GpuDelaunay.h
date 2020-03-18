#pragma once

#include<iomanip>
#include<iostream>

#include "CommonTypes.h"
#include "PerfTimer.h"

#include "GPU/CudaWrapper.h"
#include "GPU/HostToKernel.h"
#include "GPU/DPredWrapper.h"
#include "GPU/SmallCounters.h"

////
// Consts
////

const int BlocksPerGrid         = 512;
const int ThreadsPerBlock       = 128;
const int PredBlocksPerGrid     = 64;
const int PredThreadsPerBlock   = 32;
const int PredTotalThreadNum    = PredBlocksPerGrid * PredThreadsPerBlock;

////
// Input / Output
////

struct GDel2DOutput
{
    TriHVec     triVec; 
    TriOppHVec  triOppVec;
    Point2      ptInfty; 

    // Statistics
    Statistics stats; 
}; 

struct GDel2DInput
{
    Point2HVec  pointVec; 
    SegmentHVec constraintVec; 

    bool insAll;       // Insert all before flipping
    bool noSort;       // Sort input points (unused)
    bool noReorder;    // Reorder the triangle before flipping
    
    ProfLevel profLevel; 

    bool isProfiling( ProfLevel level ) const
    {
        return ( profLevel >= level ); 
    }

    GDel2DInput()
    {
        // Default setting

        insAll      = false; 
        noSort      = false; 
        noReorder   = false;
        
        profLevel   = ProfDefault; 
    }
};

////
// Main class
////

class GpuDel
{
private:
    const GDel2DInput* _input; 
    GDel2DOutput*      _output; 

    // Input
    Point2DVec  _pointVec;
    SegmentDVec _constraintVec; 
    int         _pointNum; 
    int         _triMax; 
    RealType    _minVal; 
    RealType    _maxVal; 

    // Output - Size proportional to triNum
    TriDVec     _triVec;
    TriOppDVec  _oppVec;
    CharDVec    _triInfoVec;

    // State
    bool        _doFlipping; 
    ActTriMode  _actTriMode;
    int         _insNum; 

    // Supplemental arrays - Size proportional to triNum
    IntDVec     _actTriVec; 
    Int2DVec    _triMsgVec;
    FlipDVec    _flipVec; 
    IntDVec     _triConsVec; 
    IntDVec     _actConsVec; 

    MemoryPool    _memPool; 

    // Supplemental arrays - Size proportional to vertNum
    IntDVec      _orgPointIdx; 
    IntDVec      _vertTriVec;

    // Very small
    IntHVec       _orgFlipNum; 
    SmallCounters _counters; 
	Point2		  _ptInfty; 
    int           _infIdx; 
    int           _availPtNum; 
    DPredWrapper  _dPredWrapper; 

    // Diagnostic - Only used when enabled
    IntDVec      __circleCountVec;
    IntDVec      __rejFlipVec;

    Diagnostic   _diagLogCompact, _diagLogCollect; 
    Diagnostic*  _diagLog; 
                 
    IntHVec      _numActiveVec; 
    IntHVec      _numFlipVec; 
    IntHVec      _numCircleVec; 

    RealHVec     _timeCheckVec; 
    RealHVec     _timeFlipVec; 

    // Timing
    CudaTimer    _profTimer[ ProfLevelCount ]; 

private:
    // Helpers
    void constructInitialTriangles();
    void bootstrapInsertion( Tri firstTri );
    void markSpecialTris();
    void expandTri( int newTriNum );
    void splitTri();
    void initProfiling(); 
    void doFlippingLoop( CheckDelaunayMode checkMode );
    bool doFlipping( CheckDelaunayMode checkMode );
	void shiftTri( IntDVec& triToVert, IntDVec& splitTriVec );
    void relocateAll();

    void startTiming( ProfLevel level ); 
    void stopTiming( ProfLevel level, double& accuTime ); 
    void pauseTiming( ProfLevel level );
    void restartTiming( ProfLevel level, double& accuTime ); 
    void shiftOppVec( IntDVec &shiftVec, TriOppDVec &dataVec, int size );
    void compactTris(); 
    void dispatchCheckDelaunay
    ( 
    CheckDelaunayMode   checkMode, 
    int                 orgActNum, 
    IntDVec&            triVoteVec
    ); 

	template< typename T > 
    void shiftExpandVec( IntDVec &shiftVec, DevVector< T > &dataVec, int size );

    void initForConstraintInsertion(); 
    bool markIntersections();
    bool doConsFlipping( int& flipNum );
    void updatePairStatus();
    void checkConsFlipping( IntDVec& triVoteVec );

    // Main
    void initForFlip(); 
    void splitAndFlip();
    void doInsertConstraints(); 
    void outputToHost(); 
    void cleanup(); 

public:
    void compute( const GDel2DInput& input, GDel2DOutput *output );
}; // class GpuDel

