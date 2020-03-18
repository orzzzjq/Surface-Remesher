#include "../GpuDelaunay.h"

#include<iomanip>
#include<iostream>

#include "KerCommon.h"
#include "KerDivision.h"
#include "KerPredicates.h"
#include "ThrustWrapper.h"

#include "../../Visualizer.h"

////
// GpuDel methods
////
void GpuDel::cleanup()
{
    thrust_free_all(); 

    _memPool.free(); 

    _pointVec.free(); 
    _constraintVec.free(); 
    _triVec.free(); 
    _oppVec.free(); 
    _triInfoVec.free(); 
    _orgPointIdx.free(); 
    _vertTriVec.free();
    _counters.free(); 
    _actConsVec.free(); 
        
    _orgFlipNum.clear(); 

    _dPredWrapper.cleanup(); 

    __circleCountVec.free(); 
    __rejFlipVec.free(); 

    _numActiveVec.clear(); 
    _numFlipVec.clear(); 
    _numCircleVec.clear(); 
    _timeCheckVec.clear(); 
    _timeFlipVec.clear(); 
}

void GpuDel::compute
(
const GDel2DInput&  input,
GDel2DOutput*       output
)
{
    // Set L1 for kernels
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    _input  = &input; 
    _output = output; 

    initProfiling(); 

    startTiming( ProfNone ); 

    initForFlip();
    splitAndFlip();
	outputToHost(); 

    stopTiming( ProfNone, _output->stats.totalTime ); 

    if ( _input->isProfiling( ProfDetail ) )
    {
        std::cout << " FlipCompact time: "; 
        _diagLogCompact.printTime(); 
        
        std::cout << std::endl; 
        std::cout << " FlipCollect time: "; 
        _diagLogCollect.printTime(); 

        std::cout << std::endl; 
    }

    cleanup(); 

    return;
}

void GpuDel::startTiming( ProfLevel level )
{
    if ( _input->isProfiling( level ) )
        _profTimer[ level ].start(); 
}

void GpuDel::pauseTiming( ProfLevel level )
{
    if ( _input->isProfiling( level ) )
        _profTimer[ level ].pause(); 
}

void GpuDel::stopTiming( ProfLevel level, double &accuTime )
{
    if ( _input->isProfiling( level ) )
    {
        _profTimer[ level ].stop(); 

        accuTime += _profTimer[ level ].value(); 
    }
}

void GpuDel::restartTiming( ProfLevel level, double &accuTime )
{
    stopTiming( level, accuTime ); 
    startTiming( level ); 
}

struct CompareX
{
	__device__ bool operator()( const Point2 &a, const Point2 &b ) const
	{
		return a._p[0] < b._p[0]; 
	}
};

struct Get2Ddist
{
	Point2		_a; 
	RealType	abx, aby; 

	Get2Ddist( const Point2 &a, const Point2 &b ) : _a(a)
	{
		abx = b._p[0] - a._p[0]; 
		aby = b._p[1] - a._p[1]; 
	}

	__device__ int operator()( const Point2 &c ) 
	{
		RealType acx = c._p[0] - _a._p[0]; 
		RealType acy = c._p[1] - _a._p[1]; 

		RealType dist = abx * acy - aby * acx; 

		return __float_as_int( fabs((float) dist) ); 
	}
};

RealType orient2dzero( const RealType *pa, const RealType *pb, const RealType *pc );

void GpuDel::constructInitialTriangles()
{
	// First, choose two extreme points along the X axis
	typedef Point2DVec::iterator Point2DIter; 

	thrust::pair< Point2DIter, Point2DIter > ret = thrust::minmax_element( 
        _pointVec.begin(), _pointVec.end(), CompareX() ); 

    int v0 = ret.first - _pointVec.begin(); 
	int v1 = ret.second - _pointVec.begin(); 

	const Point2 p0 = _pointVec[v0]; 
	const Point2 p1 = _pointVec[v1]; 

	// Find the furthest point from v0v1
	IntDVec distVec = _memPool.allocateAny<int>( _pointNum ); 

	distVec.resize( _pointVec.size() ); 

	thrust::transform( _pointVec.begin(), _pointVec.end(), distVec.begin(), Get2Ddist( p0, p1 ) ); 

	const int v2	= thrust::max_element( distVec.begin(), distVec.end() ) - distVec.begin(); 
	const Point2 p2 = _pointVec[v2]; 

    _memPool.release( distVec ); 

    if ( _input->isProfiling( ProfDebug ) )
	{
		std::cout << "Leftmost: " << v0 << " --> " << p0._p[0] << " " << p0._p[1] << std::endl; 
		std::cout << "Rightmost: " << v1 << " --> " << p1._p[0] << " " << p1._p[1] << std::endl; 
		std::cout << "Furthest 2D: " << v2 << " --> " << p2._p[0] << " " << p2._p[1] << std::endl; 
	}

	// Check to make sure the 4 points are not co-planar
	RealType ori = orient2dzero( p0._p, p1._p, p2._p ); 

	if ( ori == 0.0 )
	{
		std::cout << "Input too degenerate!!!\n" << std::endl; 
		exit(-1); 
	}

	if ( ortToOrient( ori ) == OrientNeg ) 
		std::swap( v0, v1 ); 

	// Compute the centroid of v0v1v2v3, to be used as the kernel point. 
	_ptInfty._p[0] = ( p0._p[0] + p1._p[0] + p2._p[0] ) / 3.0; 
	_ptInfty._p[1] = ( p0._p[1] + p1._p[1] + p2._p[1] ) / 3.0; 

    // Add the infinity point to the end of the list
    _infIdx = _pointNum - 1; 

    _pointVec.resize( _pointNum ); 
    _pointVec[ _infIdx ] = _ptInfty; 

	if ( _input->isProfiling( ProfDiag ) ) 
	{
		std::cout << "Kernel: " << _ptInfty._p[0] << " " << _ptInfty._p[1] << std::endl; 
	}

    // Initialize the predicate wrapper!!!
    _dPredWrapper.init( 
        toKernelPtr( _pointVec ), 
        _pointNum, 
        _input->noSort ? NULL : toKernelPtr( _orgPointIdx ), 
        _infIdx ); 

    setPredWrapperConstant( _dPredWrapper ); 

    // Create the initial triangulation
    Tri firstTri = { v0, v1, v2 }; 

    _triVec.expand( 4 );
    _oppVec.expand( 4 );
    _triInfoVec.expand( 4 );

    // Put the initial tets at the Inf list
    kerMakeFirstTri<<< 1, 1 >>>(
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _triInfoVec ),
		firstTri, _infIdx
		);
    CudaCheckError();

    // Locate initial positions of points
    _vertTriVec.resize( _pointNum );

    IntDVec exactCheckVec = _memPool.allocateAny<int>( _pointNum ); 

    _counters.renew(); 

    kerInitPointLocationFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertTriVec ), 
        toKernelPtr( exactCheckVec ), 
        _counters.ptr(), 
        firstTri 
        ); 

    kerInitPointLocationExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        toKernelPtr( _vertTriVec ), 
        toKernelPtr( exactCheckVec ), 
        _counters.ptr(), 
        firstTri
        ); 
    CudaCheckError();

    _memPool.release( exactCheckVec ); 

    _availPtNum = _pointNum - 4; 

    Visualizer::instance()->addFrame( _pointVec, SegmentDVec(), _triVec, _infIdx ); 
}

void GpuDel::initForFlip()
{
    startTiming( ProfDefault ); 

    _pointNum   = _input->pointVec.size() + 1;	// Plus the infinity point
    _triMax     = (int) ( _pointNum * 2 );

    // Copy points to GPU
    _pointVec.resize( _pointNum );  // 1 additional slot for the infinity point
    _pointVec.copyFromHost( _input->pointVec );

    // Copy constraints to GPU
    _constraintVec.copyFromHost( _input->constraintVec ); 

    // Allocate space
    _triVec.resize( _triMax );
    _oppVec.resize( _triMax );
    _triInfoVec.resize( _triMax );
    _counters.init( CounterNum ); 

    if ( _constraintVec.size() > 0 ) 
        _actConsVec.resize( _constraintVec.size() ); 

    if ( _input->isProfiling( ProfDiag ) )
    {
        __circleCountVec.resize( _triMax );
        __rejFlipVec.resize( _triMax );
    }

    // Preallocate some buffers in the pool
    _memPool.reserve<FlipItem>( _triMax );  // flipVec
    _memPool.reserve<int2>( _triMax );      // triMsgVec
    _memPool.reserve<int>( _pointNum );     // vertSphereVec
    _memPool.reserve<int>( _triMax );       // actTriVec
    _memPool.reserve<int>( _triMax );       // Two more for common use
    _memPool.reserve<int>( _triMax );       //

    if ( _constraintVec.size() > 0 ) 
        _memPool.reserve<int>( _triMax ); 

	// Find the min and max coordinate value
    typedef thrust::device_ptr< RealType > RealPtr; 
	RealPtr coords( ( RealType* ) toKernelPtr( _pointVec ) ); 
    thrust::pair< RealPtr, RealPtr> ret
        = thrust::minmax_element( coords, coords + _pointVec.size() * 2 ); 

    _minVal = *ret.first; 
    _maxVal = *ret.second; 

    if ( _input->isProfiling( ProfDebug ) ) 
    {
        std::cout << "_minVal = " << _minVal << ", _maxVal == " << _maxVal << std::endl; 
    }

    // Sort points along space curve
    if ( !_input->noSort )
    {
        stopTiming( ProfDefault, _output->stats.initTime ); 
        startTiming( ProfDefault ); 

        IntDVec valueVec = _memPool.allocateAny<int>( _pointNum );  
        valueVec.resize( _pointVec.size() );

        _orgPointIdx.resize( _pointNum ); 
        thrust::sequence( _orgPointIdx.begin(), _orgPointIdx.end(), 0 ); 

        thrust_transform_GetMortonNumber( 
            _pointVec.begin(), _pointVec.end(), valueVec.begin(),
            _minVal, _maxVal );

        thrust_sort_by_key( valueVec.begin(), valueVec.end(), 
            make_zip_iterator( make_tuple( 
                _orgPointIdx.begin(), _pointVec.begin() ) ) ); 

        _memPool.release( valueVec ); 

        stopTiming( ProfDefault, _output->stats.sortTime ); 
        startTiming( ProfDefault ); 
    }

    // Create first upper-lower triangles
	constructInitialTriangles(); 

    stopTiming( ProfDefault, _output->stats.initTime ); 

    return;
}

void GpuDel::doFlippingLoop( CheckDelaunayMode checkMode )
{
    startTiming( ProfDefault ); 

    _flipVec    = _memPool.allocateAny<FlipItem>( _triMax ); 
    _triMsgVec  = _memPool.allocateAny<int2>( _triMax ); 
    _actTriVec  = _memPool.allocateAny<int>( _triMax ); 

    _triMsgVec.assign( _triMax, make_int2( -1, -1 ) ); 

    int flipLoop = 0; 

    _actTriMode = ActTriMarkCompact;
    _diagLog    = &_diagLogCompact; 

    while ( doFlipping( checkMode ) ) 
        ++flipLoop; 

    stopTiming( ProfDefault, _output->stats.flipTime ); 

    relocateAll(); 

    _memPool.release( _triMsgVec ); 
    _memPool.release( _flipVec ); 
    _memPool.release( _actTriVec ); 
}

void GpuDel::initProfiling()
{
    _output->stats.reset(); 

    _diagLogCompact.reset(); 
    _diagLogCollect.reset(); 

    _numActiveVec.clear(); 
    _numFlipVec.clear(); 
    _timeCheckVec.clear(); 
    _timeFlipVec.clear(); 
}

void GpuDel::initForConstraintInsertion()
{
    if ( !_input->noSort )
    {
        // Update vertex indices of constraints
        IntDVec mapVec = _memPool.allocateAny<int>( _pointNum ); 
        mapVec.resize( _pointNum ); 

        thrust_scatterSequenceMap( _orgPointIdx, mapVec ); 

        thrust::device_ptr<int> segInt( (int *) toKernelPtr( _constraintVec ) ); 
        thrust::gather( segInt, segInt + _constraintVec.size() * 2, mapVec.begin(), segInt ); 

        _memPool.release( mapVec ); 

    //    // Sort the constraints
    //    const int constraintNum = _constraintVec.size(); 

    //    IntDVec keyVec = _memPool.allocateAny<int>( constraintNum ); 
    //    keyVec.resize( constraintNum ); 

    //    thrust::transform( _constraintVec.begin(), _constraintVec.end(), keyVec.begin(), GetConstraintMinVert() ); 

    //    thrust::sort_by_key( keyVec.begin(), keyVec.end(), _constraintVec.begin() ); 

    //    _memPool.release( keyVec ); 
    }

    // Construct 
    _vertTriVec.resize( _pointNum ); 

    kerMapTriToVert<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _triVec ), 
        toKernelPtr( _vertTriVec )
        ); 
    CudaCheckError(); 

    // Initialize list of active constraints
    thrust::sequence( _actConsVec.begin(), _actConsVec.end() ); 
}

bool GpuDel::markIntersections() 
{
    _counters.renew(); 

    kerMarkTriConsIntersectionFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actConsVec ), 
        toKernelPtr( _constraintVec ), 
        toKernelPtr( _triVec ), 
        toKernelPtr( _oppVec ),
        toKernelPtr( _triInfoVec ),
        toKernelPtr( _vertTriVec ), 
        toKernelPtr( _triConsVec ), 
        _counters.ptr()
        ); 

    kerMarkTriConsIntersectionExact<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actConsVec ), 
        toKernelPtr( _constraintVec ), 
        toKernelPtr( _triVec ), 
        toKernelPtr( _oppVec ),
        toKernelPtr( _triInfoVec ),
        toKernelPtr( _vertTriVec ), 
        toKernelPtr( _triConsVec ), 
        _counters.ptr()
        ); 
    CudaCheckError(); 

    return ( _counters[ CounterFlag ] == 1 ); 
}

void GpuDel::updatePairStatus()
{
    IntDVec exactVec = _memPool.allocateAny<int>( _triMax );  

    _counters.renew(); 

    kerUpdatePairStatusFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actTriVec ),
        toKernelPtr( _triConsVec ),
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _triInfoVec ),
        toKernelPtr( exactVec ), 
        _counters.ptr()
        );

    kerUpdatePairStatusExact<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actTriVec ),
        toKernelPtr( _triConsVec ),
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _triInfoVec ),
        toKernelPtr( exactVec ), 
        _counters.ptr()
        );
    CudaCheckError();

    _memPool.release( exactVec ); 
}

void GpuDel::checkConsFlipping( IntDVec& triVoteVec )
{
    IntDVec exactVec = _memPool.allocateAny<int>( _triMax );  

    _counters.renew(); 

    kerCheckConsFlippingFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actTriVec ), 
        toKernelPtr( _triConsVec ),
        toKernelPtr( _triInfoVec ), 
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ), 
        toKernelPtr( triVoteVec ),
        toKernelPtr( exactVec ), 
        _counters.ptr()
        );

    kerCheckConsFlippingExact<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actTriVec ), 
        toKernelPtr( _triConsVec ),
        toKernelPtr( _triInfoVec ), 
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ), 
        toKernelPtr( triVoteVec ),
        toKernelPtr( exactVec ), 
        _counters.ptr()
        );
    CudaCheckError();

    _memPool.release( exactVec ); 
}

bool GpuDel::doConsFlipping( int &flipNum )
{
    const int triNum  = _triVec.size();
    const int actNum  = _actTriVec.size(); 

    ///////
    // Vote for flips
    ///////
#pragma region Diagnostic
    if ( _input->isProfiling( ProfDiag ) )
        __rejFlipVec.assign( triNum, 0 );
#pragma endregion

    updatePairStatus(); 

    IntDVec triVoteVec = _memPool.allocateAny<int>( _triMax ); 
    triVoteVec.assign( triNum, INT_MAX );

    checkConsFlipping( triVoteVec ); 

    ////
    // Mark rejected flips
    ////
    IntDVec flipToTri = _memPool.allocateAny<int>( _triMax ); 

    flipToTri.resize( actNum );

    kerMarkRejectedConsFlips<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _actTriVec ),
        toKernelPtr( _triConsVec ), 
        toKernelPtr( triVoteVec ),
        toKernelPtr( _triInfoVec ), 
        toKernelPtr( _oppVec ),
        toKernelPtr( flipToTri ),
        _input->isProfiling( ProfDiag ) ? toKernelPtr( __rejFlipVec ) : NULL );
    CudaCheckError();

    _memPool.release( triVoteVec ); 

    ////
    // Compact flips
    ////
    IntDVec temp = _memPool.allocateAny<int>( _triMax, true ); 
    flipNum = compactIfNegative( flipToTri, temp ); 

    if ( 0 == flipNum ) 
    {
        _memPool.release( flipToTri ); 
        return false; 
    }

    ////
    // Expand flip vector
    ////
    int orgFlipNum = _flipVec.size(); 
    int expFlipNum = orgFlipNum + flipNum; 

    if ( expFlipNum > _flipVec.capacity() ) 
    {
        _flipVec.resize( 0 ); 
        _triMsgVec.assign( _triMax, make_int2( -1, -1 ) ); 

        orgFlipNum = 0; 
        expFlipNum = flipNum; 
    }

    _flipVec.grow( expFlipNum ); 

    // See doFlipping
    _triMsgVec.resize( _triVec.size() ); 

    ////
    // Flipping
    ////
#pragma region Diagnostic
    if ( _input->isProfiling( ProfDiag ) )
    {
        const int rejFlipNum = thrust_sum( __rejFlipVec );

        std::cout << "  ConsFlips: " << flipNum << " ( " << rejFlipNum << " )" 
            << std::endl;
    }
#pragma endregion

    // 32 ThreadsPerBlock is optimal
    kerFlip<<< BlocksPerGrid, 32 >>>( 
        toKernelArray( flipToTri ),
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ),
        NULL,
        toKernelPtr( _triMsgVec ),
        NULL,
        toKernelPtr( _flipVec ),
        toKernelPtr( _triConsVec ), 
        toKernelPtr( _vertTriVec ),
        orgFlipNum, 0
        ); 
    CudaCheckError(); 

    ////
    // Update oppTri
    ////
    kerUpdateOpp<<< BlocksPerGrid, 32 >>>(
        toKernelPtr( _flipVec ) + orgFlipNum,
        toKernelPtr( _oppVec ),
        toKernelPtr( _triMsgVec ),
        toKernelPtr( flipToTri ),
        orgFlipNum, flipNum
        ); 
    CudaCheckError();

    _memPool.release( flipToTri ); 

/////////////////////////////////////////////////////////////////////

    return true;   
}

void GpuDel::doInsertConstraints() 
{
    startTiming( ProfDefault ); 

    initForConstraintInsertion(); 

    const int triNum = _triVec.size(); 

    _triConsVec = _memPool.allocateAny<int>( triNum ); 
    _triConsVec.assign( triNum, -1 ); 

    _flipVec    = _memPool.allocateAny<FlipItem>( _triMax ); 
    _triMsgVec  = _memPool.allocateAny<int2>( _triMax ); 
    _actTriVec  = _memPool.allocateAny<int>( _triMax ); 

    _triMsgVec.assign( _triMax, make_int2( -1, -1 ) ); 

    int outerLoop  = 0; 
    int flipLoop   = 0; 
    int totFlipNum = 0; 
    int flipNum; 

    while ( markIntersections() ) 
    {
        if ( _input->isProfiling( ProfDiag ) )
            std::cout << "Iter " << ( outerLoop+1 ) << std::endl; 

        // VISUALIZATION
        if ( Visualizer::instance()->isEnable() ) 
        {
            pauseTiming( ProfNone ); 
            pauseTiming( ProfDefault ); 

            IntHVec triColorVec; 
            _triConsVec.copyToHost( triColorVec ); 

            for ( int i = 0; i < triColorVec.size(); ++i ) 
                if ( triColorVec[i] != -1 ) 
                    triColorVec[i] >>= 4; 

            Visualizer::instance()->addFrame( _pointVec, _constraintVec, _triVec, triColorVec, _infIdx ); 

            startTiming( ProfDefault ); 
            startTiming( ProfNone ); 
        }

        // Collect active triangles
        thrust_copyIf_IsNotNegative( _triConsVec, _actTriVec ); 

        int innerLoop = 0; 

        while ( doConsFlipping( flipNum ) )
        {
            totFlipNum += flipNum; 

            // VISUALIZATION
            if ( Visualizer::instance()->isEnable() ) 
            {
                pauseTiming( ProfNone ); 
                pauseTiming( ProfDefault ); 

                IntHVec triColorVec; 
                _triConsVec.copyToHost( triColorVec ); 

                for ( int i = 0; i < triColorVec.size(); ++i ) 
                    if ( triColorVec[i] != -1 ) 
                        triColorVec[i] >>= 4; 

                Visualizer::instance()->addFrame( _pointVec, _constraintVec, _triVec, triColorVec, _infIdx ); 

                startTiming( ProfDefault ); 
                startTiming( ProfNone ); 
            }

            ++flipLoop; 
            ++innerLoop; 

            if ( innerLoop == 5 ) break; 

            //if ( flipLoop == 1 ) break; 
        }

        ++outerLoop; 

        // Mark all the possibly modified triangles as Alive + Changed (3). 
        thrust_scatterConstantMap( _actTriVec, _triInfoVec, 3 ); 

        //if ( outerLoop == 5 ) break; 
    }

    //if ( outerLoop >= 20 )
    //{
    //    for ( int i = 0; i < _actTriVec.size(); ++i ) 
    //        std::cout << _actTriVec[i] << " "; 
    //    std::cout << std::endl; 
    //}

    if ( _input->isProfiling( ProfDiag ) )
        std::cout << "ConsFlip: Outer loop = " << outerLoop 
        << ", inner loop = " << flipLoop 
        << ", total flip = " << totFlipNum 
        << std::endl; 

    _memPool.release( _triConsVec ); 
    _memPool.release( _triMsgVec ); 
    _memPool.release( _actTriVec );        
    _memPool.release( _flipVec );        

    stopTiming( ProfDefault, _output->stats.constraintTime ); 
}

void GpuDel::splitAndFlip()
{
    int insLoop = 0;

    _doFlipping = !_input->insAll; 

    //////////////////
    while ( _availPtNum > 0 )
    //////////////////
    {
        ////////////////////////
        splitTri();
        ////////////////////////

        if ( _doFlipping ) 
            doFlippingLoop( CircleFastOrientFast ); 

        ++insLoop;
    }

    //////////////////////////////
    if ( !_doFlipping ) 
        doFlippingLoop( CircleFastOrientFast ); 

    markSpecialTris(); 
    doFlippingLoop( CircleExactOrientSoS ); 

    //////////////////////////////
    // Insert constraints if needed
    if ( _constraintVec.size() > 0 ) 
        doInsertConstraints(); 

    doFlippingLoop( CircleFastOrientFast ); 

    markSpecialTris(); 
    doFlippingLoop( CircleExactOrientSoS ); 

#pragma region Diagnostic
    if ( _input->isProfiling( ProfDiag ) )
    {
        std::cout << "\nInsert loops: " << insLoop << std::endl;

        std::cout << "Compact: " << std::endl; 
        _diagLogCompact.printCount(); 

        std::cout << "Collect: " << std::endl; 
        _diagLogCollect.printCount(); 
    }
#pragma endregion

    return;
}

void GpuDel::markSpecialTris()
{
    startTiming( ProfDetail ); 

    kerMarkSpecialTris<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _triInfoVec ), 
        toKernelPtr( _oppVec )
        );    
    CudaCheckError(); 

    stopTiming( ProfDetail, _diagLog->_t[ 0 ] ); 
}

void GpuDel::expandTri( int newTriNum )
{
    //*** Expand triangles
    _triVec.expand( newTriNum );
    _oppVec.expand( newTriNum );
    _triInfoVec.expand( newTriNum );
}

void GpuDel::splitTri()
{
    const int MaxSamplePerTri = 100; 

    startTiming( ProfDefault ); 

    ////
    // Rank points
    ////
    int triNum   = _triVec.size();
    int noSample = _pointNum; 
    
    if ( noSample / triNum > MaxSamplePerTri )
        noSample = triNum * MaxSamplePerTri; 

    IntDVec triCircleVec = _memPool.allocateAny<int>( _triMax ); 
    triCircleVec.assign( triNum, INT_MIN );

    IntDVec vertCircleVec = _memPool.allocateAny<int>( _pointNum ); 
    vertCircleVec.resize( noSample );

    kerVoteForPoint<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertTriVec ),
        toKernelPtr( _triVec ),
        toKernelPtr( vertCircleVec ),
        toKernelPtr( triCircleVec ),
        noSample
        );
    CudaCheckError();

    IntDVec triToVert = _memPool.allocateAny<int>( _triMax ); 
    triToVert.assign( triNum, INT_MAX );

    kerPickWinnerPoint<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertTriVec ),
        toKernelPtr( vertCircleVec ),
        toKernelPtr( triCircleVec ),
        toKernelPtr( triToVert ),
        noSample
        );
    CudaCheckError();

    _memPool.release( vertCircleVec ); 
    _memPool.release( triCircleVec ); 

    ////
    // Collect triangles with insertions
    ////
    IntDVec splitTriVec = _memPool.allocateAny<int>( _pointNum ); 
    _insNum = thrust_copyIf_TriHasVert( triToVert, splitTriVec ); 

    const int extraTriNum   = DIM * _insNum;
    const int splitTriNum   = triNum + extraTriNum;

    if ( _input->isProfiling( ProfDiag ) )
    {
        std::cout << "Insert: " << _insNum 
        << " Tri from: " << triNum
        << " to: " << splitTriNum << std::endl;
    }

    // If there's just a few points
    if ( _availPtNum - _insNum < _insNum && 
        _insNum < 0.1 * _pointNum ) 
    {
        _doFlipping = false; 
        //std::cout << "Stop flipping!" << std::endl; 
    }

    if ( !_input->noReorder && _doFlipping ) 
    {
        stopTiming( ProfDefault, _output->stats.splitTime ); 

        shiftTri( triToVert, splitTriVec ); 

        triNum = -1;    // Mark that we have shifted the array

        startTiming( ProfDefault ); 
    }

    ////
    // Make map
    ////
    IntDVec insTriMap = _memPool.allocateAny<int>( _triMax ); 

    insTriMap.assign( ( triNum < 0 ) ? splitTriNum : triNum, -1 );

    thrust_scatterSequenceMap( splitTriVec, insTriMap ); 

    ////
    // Expand if space needed
    ////
    expandTri( splitTriNum );

    ////
    // Update the location of the points
    ////
    stopTiming( ProfDefault, _output->stats.splitTime ); 
    startTiming( ProfDefault ); 

    IntDVec exactCheckVec = _memPool.allocateAny<int>( _pointNum ); 

    _counters.renew(); 

    kerSplitPointsFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertTriVec ),
        toKernelPtr( triToVert ),
        toKernelPtr( _triVec ),
        toKernelPtr( insTriMap ),
        toKernelPtr( exactCheckVec ), 
        _counters.ptr(),
        triNum, _insNum
        );

    kerSplitPointsExactSoS<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        toKernelPtr( _vertTriVec ),
        toKernelPtr( triToVert ),
        toKernelPtr( _triVec ),
        toKernelPtr( insTriMap ),
        toKernelPtr( exactCheckVec ), 
        _counters.ptr(),
        triNum, _insNum
        );
    CudaCheckError();

    _memPool.release( exactCheckVec ); 

    stopTiming( ProfDefault, _output->stats.relocateTime ); 
    startTiming( ProfDefault ); 

    ////
    // Split old into new triangle and copy them to new array
    ////
    kerSplitTri<<< BlocksPerGrid, 32 >>>(
        toKernelArray( splitTriVec ),
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _triInfoVec ),
        toKernelPtr( insTriMap ),
        toKernelPtr( triToVert ),
        triNum, _insNum
        );
    CudaCheckError();

    _memPool.release( triToVert ); 
    _memPool.release( insTriMap );
    _memPool.release( splitTriVec ); 

    _availPtNum -= _insNum; 

    stopTiming( ProfDefault, _output->stats.splitTime ); 

    Visualizer::instance()->addFrame( _pointVec, SegmentDVec(), _triVec, _infIdx ); 

    return;
}

bool GpuDel::doFlipping( CheckDelaunayMode checkMode )
{
    startTiming( ProfDetail ); 

    ++_diagLog->_flipLoop; 

    const int triNum  = _triVec.size();

    ////
    // Compact active triangles
    ////

    switch ( _actTriMode ) 
    {
    case ActTriMarkCompact: 
        thrust_copyIf_IsActiveTri( _triInfoVec, _actTriVec ); 
        break; 

    case ActTriCollectCompact: 
        IntDVec temp = _memPool.allocateAny<int>( _triMax, true ); 
        compactIfNegative( _actTriVec, temp );
        break; 
    }

    int orgActNum = _actTriVec.size(); 

#pragma region Diagnostic
    if ( _input->isProfiling( ProfDiag ) )
    {
        _numActiveVec.push_back( orgActNum ); 

        if ( orgActNum == 0 || ( checkMode != CircleExactOrientSoS && 
            orgActNum < PredBlocksPerGrid * PredThreadsPerBlock ) ) 
        {
            _numFlipVec.push_back( 0 ); 
            _timeCheckVec.push_back( 0.0 ); 
            _timeFlipVec.push_back( 0.0 ); 
            _numCircleVec.push_back( 0 ); 
        }
    }
#pragma endregion

    restartTiming( ProfDetail, _diagLog->_t[ 0 ] ); 
/////////////////////////////////////////////////////////////////////
    ////
    // Check actNum, switch mode or quit if necessary
    ////

    // No more work
    if ( 0 == orgActNum )
        return false;

    // Little work, leave it for the Exact iterations
    if ( checkMode != CircleExactOrientSoS && 
        orgActNum < PredBlocksPerGrid * PredThreadsPerBlock ) 
        return false; 

    // See if there's little work enough to switch to collect mode. 
    // Safety check: make sure there's enough space to collect
    if ( orgActNum < BlocksPerGrid * ThreadsPerBlock &&
        orgActNum * 2 < _actTriVec.capacity() &&
        orgActNum * 2 < triNum )
    {
        _actTriMode = ActTriCollectCompact; 
        _diagLog    = &_diagLogCollect; 
    }
    else
    {
        _actTriMode = ActTriMarkCompact; 
        _diagLog    = &_diagLogCompact; 
    }

    ////
    // Vote for flips
    ////

#pragma region Diagnostic
    if ( _input->isProfiling( ProfDiag ) )
    {
        __circleCountVec.assign( triNum, 0 );
        __rejFlipVec.assign( triNum, 0 );
    }
#pragma endregion

    IntDVec triVoteVec = _memPool.allocateAny<int>( _triMax ); 
    triVoteVec.assign( triNum, INT_MAX );

    dispatchCheckDelaunay( checkMode, orgActNum, triVoteVec ); 

    double prevTime = _diagLog->_t[ 1 ]; 

    restartTiming( ProfDetail, _diagLog->_t[ 1 ] );
/////////////////////////////////////////////////////////////////////
    ////
    // Mark rejected flips
    ////

    IntDVec flipToTri = _memPool.allocateAny<int>( _triMax ); 

    flipToTri.resize( orgActNum );
    
    kerMarkRejectedFlips<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelPtr( _actTriVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( triVoteVec ),
        toKernelPtr( _triInfoVec ),
        toKernelPtr( flipToTri ),
        orgActNum,
        _input->isProfiling( ProfDiag ) ? toKernelPtr( __rejFlipVec ) : NULL );
    CudaCheckError();

    _memPool.release( triVoteVec ); 
    
    restartTiming( ProfDetail, _diagLog->_t[ 2 ] ); 
/////////////////////////////////////////////////////////////////////
    ////
    // Compact flips
    ////
    IntDVec temp = _memPool.allocateAny<int>( _triMax, true ); 
    const int flipNum = compactIfNegative( flipToTri, temp ); 

    if ( _input->isProfiling( ProfDiag ) )
    {
        _numFlipVec.push_back( flipNum ); 
        _timeCheckVec.push_back( _diagLog->_t[ 1 ] - prevTime ); 
    }

    restartTiming( ProfDetail, _diagLog->_t[ 3 ] );  
/////////////////////////////////////////////////////////////////////
    ////
    // Preparation for the actual flipping. Include several steps
    ////

#pragma region Diagnostic
    if ( _input->isProfiling( ProfDiag ) )
    {
        const int circleNum = thrust_sum( __circleCountVec );
        _diagLog->_circleCount += circleNum; 
        const int rejFlipNum = thrust_sum( __rejFlipVec );
        _diagLog->_rejFlipCount += rejFlipNum;

        _diagLog->_totFlipNum   += flipNum;

        std::cout << "Acts: " << orgActNum
			<< " Flips: " << flipNum << " ( " << rejFlipNum << " )" 
            << " circle: " << circleNum 
            << " Exact: " 
            << ( checkMode == CircleExactOrientSoS ? _counters[ CounterExact ] : -1 )
            << std::endl;

        _numCircleVec.push_back( circleNum ); 

        startTiming( ProfDetail ); 
    }
#pragma endregion

    if ( 0 == flipNum )
    {
        _numCircleVec.push_back( 0 ); 
        _timeFlipVec.push_back( 0 ); 
        _memPool.release( flipToTri ); 
        return false;
    }

    // Expand flip vector
    int orgFlipNum = _flipVec.size(); 
    int expFlipNum = orgFlipNum + flipNum; 

    if ( expFlipNum > _flipVec.capacity() ) 
    {
        stopTiming( ProfDetail, _diagLog->_t[ 4 ] ); 
        stopTiming( ProfDefault, _output->stats.flipTime ); 
           relocateAll(); 
        startTiming( ProfDefault ); 
        startTiming( ProfDetail ); 

        orgFlipNum = 0; 
        expFlipNum = flipNum; 
    }

    _flipVec.grow( expFlipNum ); 

    // _triMsgVec contains two components. 
    // - .x is the encoded new neighbor information
    // - .y is the flipIdx as in the flipVec (i.e. globIdx)
    // As such, we do not need to initialize it to -1 to 
    // know which tris are not flipped in the current rount. 
    // We can rely on the flipIdx being > or < than orgFlipIdx. 
    // Note that we have to initialize everything to -1 
    // when we clear the flipVec and reset the flip indexing. 
    //
    _triMsgVec.resize( _triVec.size() ); 

    ////
    // Expand active tri vector
    ////
    if ( _actTriMode == ActTriCollectCompact ) 
        _actTriVec.grow( orgActNum + flipNum );

    restartTiming( ProfDetail, _diagLog->_t[ 4 ] ); 
/////////////////////////////////////////////////////////////////////
    ////
    // Flipping
    ////

    // 32 ThreadsPerBlock is optimal
    kerFlip<<< BlocksPerGrid, 32 >>>( 
        toKernelArray( flipToTri ),
        toKernelPtr( _triVec ),
        toKernelPtr( _oppVec ),
        toKernelPtr( _triInfoVec ),
        toKernelPtr( _triMsgVec ),
        ( _actTriMode == ActTriCollectCompact ) ? toKernelPtr( _actTriVec ) : NULL,
        toKernelPtr( _flipVec ),
        NULL, NULL,
        orgFlipNum, orgActNum
        ); 
    CudaCheckError(); 

    _orgFlipNum.push_back( orgFlipNum ); 

    ////
    // Update oppTri
    ////

    kerUpdateOpp<<< BlocksPerGrid, 32 >>>(
        toKernelPtr( _flipVec ) + orgFlipNum,
        toKernelPtr( _oppVec ),
        toKernelPtr( _triMsgVec ),
        toKernelPtr( flipToTri ),
        orgFlipNum, flipNum
        ); 
    CudaCheckError();

    _memPool.release( flipToTri ); 

    prevTime = _diagLog->_t[ 5 ]; 

    stopTiming( ProfDetail, _diagLog->_t[ 5 ] ); 

    if ( _input->isProfiling( ProfDiag ) )
        _timeFlipVec.push_back( _diagLog->_t[ 5 ] - prevTime ); 
/////////////////////////////////////////////////////////////////////

    Visualizer::instance()->addFrame( _pointVec, SegmentDVec(), _triVec, _infIdx ); 

    return true;
}

void GpuDel::dispatchCheckDelaunay
( 
CheckDelaunayMode   checkMode, 
int                 orgActNum,
IntDVec&            triVoteVec
) 
{
    switch ( checkMode ) 
    {
    case CircleFastOrientFast: 
        kerCheckDelaunayFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelPtr( _actTriVec ),
            toKernelPtr( _triVec ),
            toKernelPtr( _oppVec ),
            toKernelPtr( _triInfoVec ),
            toKernelPtr( triVoteVec ),
            orgActNum,
            _input->isProfiling( ProfDiag ) ? toKernelPtr( __circleCountVec ) : NULL
            );
        CudaCheckError();
        break; 

    case CircleExactOrientSoS:
        // Reuse this array to save memory
        Int2DVec &exactCheckVi = _triMsgVec; 

        _counters.renew(); 

        kerCheckDelaunayExact_Fast<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelPtr( _actTriVec ),
            toKernelPtr( _triVec ),
            toKernelPtr( _oppVec ),
            toKernelPtr( _triInfoVec ),
            toKernelPtr( triVoteVec ),
            toKernelPtr( exactCheckVi ), 
            orgActNum,
            _counters.ptr(), 
            _input->isProfiling( ProfDiag ) ? toKernelPtr( __circleCountVec ) : NULL
            );

        kerCheckDelaunayExact_Exact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
            toKernelPtr( _triVec ),
            toKernelPtr( _oppVec ),
            toKernelPtr( triVoteVec ),
            toKernelPtr( exactCheckVi ), 
            _counters.ptr(), 
            _input->isProfiling( ProfDiag ) ? toKernelPtr( __circleCountVec ) : NULL
            );
        CudaCheckError();

        break; 
    }
}

template< typename T >
__global__ void 
kerShift
(
KerIntArray shiftVec, 
T*          src, 
T*          dest
) 
{
    for ( int idx = getCurThreadIdx(); idx < shiftVec._num; idx += getThreadNum() )
    {
        const int shift = shiftVec._arr[ idx ]; 

        dest[ idx + shift ] = src[ idx ]; 
    }
}

template< typename T > 
void GpuDel::shiftExpandVec( IntDVec &shiftVec, DevVector< T > &dataVec, int size )
{
    DevVector< T > tempVec = _memPool.allocateAny<T>( size ); 

    tempVec.resize( size );

    kerShift<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( shiftVec ), 
        toKernelPtr( dataVec ),
        toKernelPtr( tempVec )
        ); 
    CudaCheckError(); 

    dataVec.copyFrom( tempVec ); 

    _memPool.release( tempVec ); 
}

void GpuDel::shiftOppVec( IntDVec &shiftVec, TriOppDVec &dataVec, int size )
{
    TriOppDVec tempVec = _memPool.allocateAny< TriOpp >( size ); 

    tempVec.resize( size );

    kerShiftOpp<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( shiftVec ), 
        toKernelPtr( dataVec ),
        toKernelPtr( tempVec ),
        size
        ); 
    CudaCheckError(); 

    dataVec.copyFrom( tempVec ); 

    _memPool.release( tempVec ); 
}

void GpuDel::shiftTri( IntDVec &triToVert, IntDVec &splitTriVec )
{
    startTiming( ProfDefault ); 

    const int triNum = _triVec.size() + 2 * splitTriVec.size(); 

    IntDVec shiftVec = _memPool.allocateAny<int>( _triMax ); 

    thrust_scan_TriHasVert( triToVert, shiftVec ); 

    shiftExpandVec( shiftVec, _triVec, triNum ); 
    shiftExpandVec( shiftVec, _triInfoVec, triNum ); 
    shiftExpandVec( shiftVec, triToVert, triNum ); 
    shiftOppVec( shiftVec, _oppVec, triNum ); 

    kerShiftTriIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _vertTriVec ),
        toKernelPtr( shiftVec )
        ); 
    CudaCheckError(); 

    kerShiftTriIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( splitTriVec ),
        toKernelPtr( shiftVec )
        ); 
    CudaCheckError(); 

    _memPool.release( shiftVec ); 

    stopTiming( ProfDefault, _output->stats.sortTime ); 
}

void GpuDel::relocateAll()
{
    if ( _flipVec.size() == 0 ) 
        return ; 

    startTiming( ProfDefault ); 

    if ( _availPtNum > 0 ) 
    {
        const int triNum = _triVec.size(); 

        IntDVec triToFlip = _memPool.allocateAny<int>( _triMax ); 
        triToFlip.assign( triNum, -1 ); 

        // Rebuild the pointers from back to forth
        int nextFlipNum = _flipVec.size(); 

        for ( int i = _orgFlipNum.size() - 1; i >= 0; --i ) 
        {
            int prevFlipNum = _orgFlipNum[ i ]; 
            int flipNum     = nextFlipNum - prevFlipNum; 

            kerUpdateFlipTrace<<< BlocksPerGrid, ThreadsPerBlock >>>(
                toKernelPtr( _flipVec ), 
                toKernelPtr( triToFlip ),
                prevFlipNum, 
                flipNum 
                ); 

            nextFlipNum = prevFlipNum; 
        }
        CudaCheckError(); 
        
        // Relocate points
        IntDVec exactCheckVec = _memPool.allocateAny<int>( _pointNum ); 

        _counters.renew(); 

        kerRelocatePointsFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelArray( _vertTriVec ),
            toKernelPtr( triToFlip ),
            toKernelPtr( _flipVec ),
            toKernelPtr( exactCheckVec ), 
            _counters.ptr()
            );

        kerRelocatePointsExact<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelPtr( _vertTriVec ),
            toKernelPtr( triToFlip ),
            toKernelPtr( _flipVec ),
            toKernelPtr( exactCheckVec ), 
            _counters.ptr()
            );
        CudaCheckError();
    
        _memPool.release( exactCheckVec ); 
        _memPool.release( triToFlip ); 
    }

    // Just clean up the flips
    _flipVec.resize( 0 ); 
    _orgFlipNum.clear(); 

    // Reset the triMsgVec
    _triMsgVec.assign( _triMax, make_int2( -1, -1 ) ); 

    stopTiming( ProfDefault, _output->stats.relocateTime ); 
}

void GpuDel::compactTris()
{
    const int triNum = _triVec.size(); 

    IntDVec prefixVec = _memPool.allocateAny<int>( _triMax ); 
    
    prefixVec.resize( triNum ); 

    thrust_scan_TriAliveStencil( _triInfoVec, prefixVec ); 

    int newTriNum = prefixVec[ triNum - 1 ];
    int freeNum   = triNum - newTriNum; 

    IntDVec freeVec = _memPool.allocateAny<int>( _triMax ); 

    freeVec.resize( freeNum ); 

    kerCollectFreeSlots<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelPtr( _triInfoVec ), 
        toKernelPtr( prefixVec ),
        toKernelPtr( freeVec ),
        newTriNum
        ); 
    CudaCheckError(); 

    // Make map
    kerMakeCompactMap<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _triInfoVec ), 
        toKernelPtr( prefixVec ),
        toKernelPtr( freeVec ),
        newTriNum
        ); 
    CudaCheckError(); 

    // Reorder the tets
    kerCompactTris<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _triInfoVec ), 
        toKernelPtr( prefixVec ), 
        toKernelPtr( _triVec ), 
        toKernelPtr( _oppVec ),
        newTriNum
        ); 
    CudaCheckError(); 

    _triInfoVec.resize( newTriNum ); 
    _triVec.resize( newTriNum ); 
    _oppVec.resize( newTriNum ); 

    _memPool.release( freeVec ); 
    _memPool.release( prefixVec ); 
}

void GpuDel::outputToHost()
{
    startTiming( ProfDefault ); 

    kerMarkInfinityTri<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( _triVec ), 
        toKernelPtr( _triInfoVec ),
        toKernelPtr( _oppVec ),
        _infIdx
        ); 
    CudaCheckError(); 

    compactTris(); 

    if ( !_input->noSort ) 
    {
        // Change the indices back to the original order
        kerUpdateVertIdx<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelArray( _triVec ), 
            toKernelPtr( _triInfoVec ),
            toKernelPtr( _orgPointIdx )
            ); 
        CudaCheckError(); 
    }

    ////
    // Copy to host  
    _triVec.copyToHost( _output->triVec );
    _oppVec.copyToHost( _output->triOppVec );

    // Output Infty point
    _output->ptInfty = _ptInfty; 

    stopTiming( ProfDefault, _output->stats.outTime ); 
    
    ////
    std::cout << "# Triangles:     " << _triVec.size() << std::endl; 

    return;
}
