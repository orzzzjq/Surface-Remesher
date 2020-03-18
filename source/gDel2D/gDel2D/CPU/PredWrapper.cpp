#include "PredWrapper.h"

void PredWrapper2D::init( const Point2HVec& pointVec, Point2 ptInfty )
{
	_pointArr	= &pointVec[0]; 
	_pointNum	= pointVec.size(); 
	_infIdx		= _pointNum; 
	_ptInfty	= ptInfty; 

    exactinit(); 
}

const Point2& PredWrapper2D::getPoint( int idx ) const
{
	return ( idx == _infIdx ) ? _ptInfty : _pointArr[ idx ]; 
}

int PredWrapper2D::pointNum() const 
{
	return _pointNum + 1; 
}

Orient PredWrapper2D::doOrient2D( int v0, int v1, int v2 ) const
{
    assert(     ( v0 != v1 ) && ( v0 != v2 ) &&  ( v1 != v2 ) 
                &&  "Duplicate indices in orientation!" );

    const Point2 p[] = { getPoint( v0 ), getPoint( v1 ), getPoint( v2 ) };

    RealType det = orient2d( p[0]._p, p[1]._p, p[2]._p );

    if ( (v0 == _infIdx) || (v1 == _infIdx) || (v2 == _infIdx) ) 
        det = -det; 

    return ortToOrient( det );
}

Orient PredWrapper2D::doOrient2DSoSOnly
(
const RealType* p0,
const RealType* p1,
const RealType* p2,
int v0,
int v1,
int v2
) const
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////
    const RealType* p[DEG] = { p0, p1, p2 }; 
    int pn = 1;

    if ( v0 > v1 ) { std::swap( v0, v1 ); std::swap( p[0], p[1] ); pn = -pn; }
    if ( v0 > v2 ) { std::swap( v0, v2 ); std::swap( p[0], p[2] ); pn = -pn; }
    if ( v1 > v2 ) { std::swap( v1, v2 ); std::swap( p[1], p[2] ); pn = -pn; }

    RealType result = 0;
    int depth;

	for ( depth = 1; depth <= 4; ++depth )
	{
		switch ( depth )
		{
		case 1:
			result = p[2][0] - p[1][0];
			break;
		case 2:
			result = p[1][1] - p[2][1];
			break;
		case 3:
			result = p[0][0] - p[2][0];
			break;
        default:
			result = 1.0;
			break;
		}

		if ( result != 0 )
			break;
	}

	const RealType det = result * pn;

    return ortToOrient( det );
}

Orient PredWrapper2D::doOrient2DFastExactSoS( int v0, int v1, int v2 ) const
{
    const RealType* pt[] = { getPoint(v0)._p, getPoint(v1)._p, getPoint(v2)._p }; 

    // Fast-Exact
    Orient ord = doOrient2D( v0, v1, v2 );
    
    if ( OrientZero == ord ) 
        ord = doOrient2DSoSOnly( pt[0], pt[1], pt[2], v0, v1, v2 );

    if ( v0 == _infIdx | v1 == _infIdx ) 
        ord = flipOrient( ord ); 

    return ord; 
}

///////////////////////////////////////////////////////////////////// Circle //

Side PredWrapper2D::doIncircle( Tri tri, int vert ) const
{
    if ( vert == _infIdx ) 
        return SideOut; 

    const Point2 pt[]  = { 
        getPoint( tri._v[0] ), 
        getPoint( tri._v[1] ), 
        getPoint( tri._v[2] ), 
        getPoint( vert ) 
    };
    
    RealType det; 

    if ( tri.has( _infIdx ) ) 
    {
        const int infVi = tri.getIndexOf( _infIdx ); 
        
        det = orient2d( pt[ (infVi + 1) % 3 ]._p, pt[ (infVi + 2) % 3 ]._p, pt[3]._p  ); 
    }
    else
        det = incircle( pt[0]._p, pt[1]._p, pt[2]._p, pt[3]._p );

    return cicToSide( det );
}

