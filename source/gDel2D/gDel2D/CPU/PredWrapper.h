#pragma once

#include "../CommonTypes.h"

// Shewchuk predicate declarations
void exactinit();

RealType orient2d
(
const RealType* pa,
const RealType* pb,
const RealType* pc
);
RealType incircle
(
const RealType *pa,
const RealType *pb,
const RealType *pc,
const RealType *pd
)
;

class PredWrapper2D
{
private:
	const Point2*	_pointArr; 
	Point2			_ptInfty; 
	int			    _pointNum; 

    Orient doOrient2DSoSOnly(
        const RealType* p0, const RealType* p1, const RealType* p2,
        int v0, int v1, int v2 ) const;

public: 
    int _infIdx; 

    void init( const Point2HVec& pointVec, Point2 ptInfty ); 

	const Point2& getPoint( int idx ) const; 
	int pointNum() const; 

    Orient doOrient2D( int v0, int v1, int v2 ) const;
    Orient doOrient2DFastExactSoS( int v0, int v1, int v2 ) const;
    Side doIncircle( Tri tri, int vert ) const;
};
