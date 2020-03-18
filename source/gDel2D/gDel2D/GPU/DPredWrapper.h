#pragma once

#include "../CommonTypes.h"

class DPredWrapper
{
private: 
    Point2      *_pointArr;
    int*        _orgPointIdx;
	int			_pointNum; 

    RealType*   _predConsts;

    __forceinline__ __device__ Orient doOrient2DFastExact( 
        const RealType* p0, const RealType* p1, const RealType* p2 ) const;

    __forceinline__ __device__ Orient doOrient2DSoSOnly(
        const RealType* p0, const RealType* p1, const RealType* p2,
        int v0, int v1, int v2 ) const; 

    __forceinline__ __device__ Side doInCircleFastExact( 
        const RealType* p0, const RealType* p1, const RealType* p2, const RealType* p3 ) const;

    __forceinline__ __device__ RealType doOrient1DExact_Lifted(
        const RealType* p0, const RealType* p1 ) const;

    __forceinline__ __device__ RealType doOrient2DExact_Lifted(
        const RealType* p0, const RealType* p1, const RealType* p2, bool lifted ) const;

    __forceinline__ __device__ Side doInCircleSoSOnly( 
        const RealType* p0, const RealType* p1, const RealType* p2, const RealType* p3,
        int v0, int v1, int v2, int v3 ) const; 

public: 
    int _infIdx;

    void init( 
        Point2* pointArr, 
        int     pointNum, 
        int*    orgPointIdx,
        int     infIdx
    ); 

    void cleanup(); 

	__forceinline__ __device__ __host__ int pointNum() const; 

    __forceinline__ __device__ const Point2& getPoint( int idx ) const; 

    __forceinline__ __device__ int getPointIdx( int idx ) const; 

    __forceinline__ __device__ Orient doOrient2DFast( int v0, int v1, int v2 ) const; 
  
    __forceinline__ __device__ Orient doOrient2DFastExactSoS( int v0, int v1, int v2 ) const; 

    __forceinline__ __device__ Side doInCircleFast( Tri tri, int vert ) const; 

    __forceinline__ __device__ Side doInCircleFastExactSoS( Tri tri, int vert ) const; 

    __forceinline__ __device__ float inCircleDet( Tri tri, int vert ) const; 
}; 

enum DPredicateBounds
{
    Splitter,       /* = 2^ceiling(p / 2) + 1.  Used to split floats in half. */
    Epsilon,        /* = 2^(-p).  Used to estimate roundoff errors. */

    /* A set of coefficients used to calculate maximum roundoff errors.          */
    Resulterrbound,
    CcwerrboundA,
    CcwerrboundB,
    CcwerrboundC,
    O3derrboundA,
    O3derrboundB,
    O3derrboundC,
    IccerrboundA,
    IccerrboundB,
    IccerrboundC,
    IsperrboundA,
    IsperrboundB,
    IsperrboundC,
    O3derrboundAlifted,
    O2derrboundAlifted,
    O1derrboundAlifted,

    DPredicateBoundNum  // Number of bounds in this enum
};

