#pragma once

#include "../CommonTypes.h"

// create a tag derived from device_system_tag for distinguishing
// our overloads of get_temporary_buffer and return_temporary_buffer
struct MyCrazyTag : thrust::device_system_tag {};

////
// Device iterators
////

typedef IntDVec::iterator                       IntDIter;
typedef thrust::tuple< int, int >               IntTuple2;
typedef thrust::tuple< IntDIter, IntDIter >     IntDIterTuple2;
typedef thrust::zip_iterator< IntDIterTuple2 >  IntZipDIter;

////
// Functions
////
void thrust_free_all();

void thrust_sort_by_key
(
DevVector<int>::iterator keyBeg, 
DevVector<int>::iterator keyEnd, 
thrust::zip_iterator< 
    thrust::tuple< 
        DevVector<int>::iterator,
        DevVector<Point2>::iterator > > valueBeg
)
;
void thrust_transform_GetMortonNumber
(
DevVector<Point2>::iterator inBeg, 
DevVector<Point2>::iterator inEnd, 
DevVector<int>::iterator    outBeg,
RealType                    minVal, 
RealType                    maxVal
)
;
int makeInPlaceMapAndSum
( 
IntDVec& inVec 
)
;
int makeInPlaceIncMapAndSum
( 
IntDVec& inVec 
)
;
int compactIfNegative
( 
DevVector<int>& inVec,
DevVector<int>& temp 
)
;
void compactBothIfNegative
( 
IntDVec& vec0, 
IntDVec& vec1 
)
;
int thrust_copyIf_IsActiveTri
(
const CharDVec& inVec,
IntDVec&        outVec
)
;
int thrust_copyIf_TriHasVert
(
const IntDVec& inVec,
IntDVec&       outVec
)
;
void thrust_scatterSequenceMap
(
const IntDVec& inVec,
IntDVec&       outVec
)
;
void thrust_scatterConstantMap
(
const IntDVec&  inVec,
CharDVec&       outVec,
char            value
)
;
void thrust_scan_TriHasVert
(
IntDVec& inVec, 
IntDVec& outVec
)
;
void thrust_scan_TriAliveStencil
(
const CharDVec& inVec, 
IntDVec& outVec
)
;
int thrust_sum
(
const IntDVec& inVec
)
;
int thrust_copyIf_IsNotNegative
(
const IntDVec& inVec,
IntDVec&       outVec
)
;
////
// Thrust helper functors
////

struct GetMortonNumber
{
    RealType _minVal, _range; 

    GetMortonNumber( RealType minVal, RealType maxVal ) 
        : _minVal( minVal ), _range( maxVal - minVal ) {}

    // Note: No performance benefit by changing by-reference to by-value here
    // Note: No benefit by making this __forceinline__
    __device__ int operator () ( const Point2& point ) const
    {
        const int Gap08 = 0x00FF00FF;   // Creates 16-bit gap between value bits
        const int Gap04 = 0x0F0F0F0F;   // ... and so on ...
        const int Gap02 = 0x33333333;   // ...
        const int Gap01 = 0x55555555;   // ...

        const int minInt = 0x0; 
        const int maxInt = 0x7FFF; 
        
        int mortonNum = 0; 

        // Iterate coordinates of point
        for ( int vi = 0; vi < 2; ++vi )
        {
            // Read
            int v = int( ( point._p[ vi ] - _minVal ) / _range * 32768.0 ); 

            if ( v < minInt ) 
                v = minInt; 

            if ( v > maxInt ) 
                v = maxInt; 

            // Create 1-bit gaps between the 10 value bits
            // Ex: 1010101010101010101
            v = ( v | ( v <<  8 ) ) & Gap08;
            v = ( v | ( v <<  4 ) ) & Gap04;
            v = ( v | ( v <<  2 ) ) & Gap02;
            v = ( v | ( v <<  1 ) ) & Gap01;

            // Interleave bits of x-y coordinates
            mortonNum |= ( v << vi ); 
        }

        return mortonNum;
    }
};

struct MakeKeyFromTriHasVert
{
    __device__ int operator() ( int v )
    {
        // getFlipType()
        return ( v < INT_MAX - 1 ) ? 2 : 0; 
    }
};

struct IsNotNegative
{
    __host__ __device__ bool operator() ( const int x )
    {
        return ( x >= 0 );
    }
};

// Check if first value in tuple2 is negative
struct IsIntTuple2Negative
{
    __host__ __device__ bool operator() ( const IntTuple2& tup )
    {
        const int x = thrust::get<0>( tup );
        return ( x < 0 );
    }
};

// Check if triangle is active
struct IsTriActive
{
    __host__ __device__ bool operator() ( char triInfo )
    {
        return ( isTriAlive( triInfo ) && ( Changed == getTriCheckState( triInfo ) ) );
    }
};

struct IsTriHasVert
{
    __host__ __device__ bool operator() ( int v )
    {
        return ( v < INT_MAX - 1 );
    }
};

// Check if triangle is alive
struct TriAliveStencil
{
    __host__ __device__ int operator() ( char triInfo )
    {
        return isTriAlive( triInfo ) ? 1 : 0;
    }
};
