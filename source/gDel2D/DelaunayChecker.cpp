#include "DelaunayChecker.h"

DelaunayChecker::DelaunayChecker
( 
GDel2DInput&  input, 
GDel2DOutput& output
)
: _input( input ), _output( output ) 
{
    _predWrapper.init( _input.pointVec, output.ptInfty ); 
} 

void getTriSegments( const Tri& t, Segment* sArr )
{
    for ( int i = 0; i < TriSegNum; ++i )
    {
        Segment seg = { t._v[ TriSeg[i][0] ], t._v[ TriSeg[i][1] ] };

        seg.sort(); 

        sArr[i] = seg;
    }

    return;
}

int DelaunayChecker::getVertexCount()
{
    const TriHVec& triVec = _output.triVec; 

    std::set< int > vertSet; 

    // Add vertices
    for ( int ti = 0; ti < triVec.size(); ++ti )
    {
        const Tri& tri = triVec[ ti ];

        vertSet.insert( tri._v, tri._v + DEG ); 
    }

    return vertSet.size();
}

int DelaunayChecker::getSegmentCount()
{
    const TriHVec& triVec   = _output.triVec; 
    const int triNum        = ( int ) triVec.size();

    std::set< Segment > segSet; 

    // Read segments
    Segment segArr[ TriSegNum ];
    for ( int ti = 0; ti < triNum; ++ti )
    {
        const Tri& tri = triVec[ ti ];

        getTriSegments( tri, segArr );

        segSet.insert( segArr, segArr + TriSegNum ); 
    }

    return segSet.size();
}

int DelaunayChecker::getTriangleCount()
{
    return _output.triVec.size();
}

void DelaunayChecker::checkEuler()
{
    const int v = getVertexCount();
    std::cout << "V: " << v;

    const int e = getSegmentCount();
    std::cout << " E: " << e;

    const int f = getTriangleCount();
    std::cout << " F: " << f;

    const int euler = v - e + f;
    std::cout << " Euler: " << euler << std::endl;

    std::cout << "Euler check: " << ( ( 1 != euler ) ? " ***Fail***" : " Pass" ) << std::endl;

    return;
}

void printTriAndOpp( int ti, const Tri& tri, const TriOpp& opp )
{
    printf( "triIdx: %d [ %d %d %d ] ( %d:%d %d:%d %d:%d )\n",
        ti,
        tri._v[0], tri._v[1], tri._v[2],
        opp.getOppTri(0), opp.getOppVi(0),
        opp.getOppTri(1), opp.getOppVi(1),
        opp.getOppTri(2), opp.getOppVi(2) );
}

void DelaunayChecker::checkAdjacency()
{
    const TriHVec triVec        = _output.triVec; 
    const TriOppHVec oppVec     = _output.triOppVec; 

    for ( int ti0 = 0; ti0 < ( int ) triVec.size(); ++ti0 )
    {
        const Tri& tri0    = triVec[ ti0 ];
        const TriOpp& opp0 = oppVec[ ti0 ];

        for ( int vi = 0; vi < DEG; ++vi )
        {
            if ( -1 == opp0._t[ vi ] ) continue;

            const int ti1   = opp0.getOppTri( vi );
            const int vi0_1 = opp0.getOppVi( vi );

            const Tri& tri1    = triVec[ ti1 ];
            const TriOpp& opp1 = oppVec[ ti1 ];

            if ( -1 == opp1._t[ vi0_1 ] )
            {
                std::cout << "Fail4!" << std::endl;
                continue;
            }

            if ( ti0 != opp1.getOppTri( vi0_1 ) )
            {
                std::cout << "Not opp of each other! Tri0: " << ti0 << " Tri1: " << ti1 << std::endl;
                printTriAndOpp( ti0, tri0, opp0 );
                printTriAndOpp( ti1, tri1, opp1 );
                continue;
            }

            if ( vi != opp1.getOppVi( vi0_1 ) )
            {
                std::cout << "Vi mismatch! Tri0: " << ti0 << "Tri1: " << ti1 << std::endl;
                continue;
            }
        }
    }

    std::cout << "Adjacency check: Pass\n";

    return;
}

void DelaunayChecker::checkOrientation()
{
    const TriHVec triVec = _output.triVec; 

    int count = 0;

    for ( int i = 0; i < ( int ) triVec.size(); ++i )
    {
        const Tri& t     = triVec[i];
        const Orient ord = _predWrapper.doOrient2DFastExactSoS( t._v[0], t._v[1], t._v[2] );

        if ( OrientNeg == ord )
            ++count;
    }

    std::cout << "Orient check: ";
    if ( count )
        std::cout << "***Fail*** Wrong orient: " << count;
    else
        std::cout << "Pass";
    std::cout << "\n";

    return;
}

void DelaunayChecker::checkDelaunay()
{
    const TriHVec triVec    = _output.triVec; 
    const TriOppHVec oppVec = _output.triOppVec; 

    const int triNum = ( int ) triVec.size();
    int failNum      = 0;

    for ( int botTi = 0; botTi < triNum; ++botTi )
    {
        const Tri botTri    = triVec[ botTi ];
        const TriOpp botOpp = oppVec[ botTi ];

        for ( int botVi = 0; botVi < DEG; ++botVi ) // Face neighbours
        {
            // No face neighbour or facing constraint
            if ( -1 == botOpp._t[botVi] || botOpp.isOppConstraint( botVi ) )
                continue;

            const int topVi = botOpp.getOppVi( botVi );
            const int topTi = botOpp.getOppTri( botVi );

            if ( topTi < botTi ) continue; // Neighbour will check

            const Tri topTri  = triVec[ topTi ];
            const int topVert = topTri._v[ topVi ];
            const Side side   = _predWrapper.doIncircle( botTri, topVert );

            if ( SideIn != side ) continue;

            ++failNum;                 
        }
    }

    std::cout << "\nDelaunay check: ";

    if ( failNum == 0 )
        std::cout << "Pass" << std::endl;
    else    
        std::cout << "***Fail*** Failed faces: " << failNum << std::endl;

    return ;
}

void DelaunayChecker::checkConstraints()
{
    if ( _input.constraintVec.size() == 0 ) return ; 

    const TriHVec triVec      = _output.triVec; 
          TriOppHVec& oppVec  = _output.triOppVec; 
    const SegmentHVec consVec = _input.constraintVec; 

    const int triNum = ( int ) triVec.size();
    int failNum      = 0;

    // Clear any existing opp constraint info. 
    for ( int i = 0; i < triNum; ++i ) 
        for ( int j = 0; j < 3; ++j ) 
            if ( oppVec[i]._t[j] != -1 ) 
                oppVec[i].setOppConstraint( j, false ); 

    // Create a vertex to triangle map
    IntHVec vertTriMap( _predWrapper.pointNum(), -1 ); 

    for ( int i = 0; i < triNum; ++i ) 
        for ( int j = 0; j < 3; ++j ) 
            vertTriMap[ triVec[i]._v[j] ] = i; 

    // Check the constraints
    for ( int i = 0; i < consVec.size(); ++i ) 
    {
        Segment constraint = consVec[i]; 

        const int startIdx  = vertTriMap[ constraint._v[0] ]; 

        if ( startIdx < 0 ) { ++failNum;  continue; }

        int triIdx = startIdx; 
        int vi     = triVec[ triIdx ].getIndexOf( constraint._v[0] ); 

        // Walk around the starting vertex to find the constraint edge
        const int MaxWalking = 1000000; 
        int j                = 0; 

        for ( ; j < MaxWalking; ++j ) 
        {
            const Tri& tri      = triVec[ triIdx ]; 
            TriOpp& opp         = oppVec[ triIdx ]; 
            const int nextVert  = tri._v[ (vi + 2) % 3 ]; 

            // The constraint is already inserted
            if ( nextVert == constraint._v[ 1 ] ) 
            {
                vi = ( vi + 1 ) % DEG; 
                j  = INT_MAX;  
                break;  
            }

            // Rotate
            if ( opp._t[ ( vi + 1 ) % DEG ] == -1 ) break; 

            triIdx  = opp.getOppTri( ( vi + 1 ) % DEG ); 
            vi      = opp.getOppVi( ( vi + 1 ) % DEG ); 
            vi      = ( vi + 1 ) % DEG; 

            if ( triIdx == startIdx ) break; 
        } 

        // If not found, rotate the other direction
        if ( j < MaxWalking ) 
        {
            triIdx = startIdx; 
            vi     = triVec[ triIdx ].getIndexOf( constraint._v[0] ); 

            for ( ; j < MaxWalking; ++j ) 
            {
                const Tri& tri      = triVec[ triIdx ]; 
                const int nextVert  = tri._v[ (vi + 1) % 3 ]; 

                if ( nextVert == constraint._v[ 1 ] )
                {
                    vi = ( vi + 2 ) % DEG; 
                    j  = INT_MAX; 
                    break; 
                }

                // Rotate
                const TriOpp& opp = oppVec[ triIdx ]; 

                if ( opp._t[ ( vi + 2 ) % DEG ] == -1 ) break; 

                triIdx  = opp.getOppTri( ( vi + 2 ) % DEG ); 
                vi      = opp.getOppVi( ( vi + 2 ) % DEG ); 
                vi      = ( vi + 2 ) % DEG; 

                if ( triIdx == startIdx ) break; 
            } 
        }

        if ( j == INT_MAX )         // Found
        {
            TriOpp& opp = oppVec[ triIdx ]; 

            const int oppTri = opp.getOppTri( vi ); 
            const int oppVi  = opp.getOppVi( vi ); 

            opp.setOppConstraint( vi, true ); 
            oppVec[ oppTri ].setOppConstraint( oppVi, true ); 
        }
        else 
        {
            if ( j >= MaxWalking )
                std::cout << "Vertex degree too high; Skipping constraint " << i << std::endl; 
            //else 
            //    std::cout << "Missing constraint " << i << std::endl; 

            ++failNum; 
        }
    }

    std::cout << "\nConstraint check: ";

    if ( failNum == 0 )
        std::cout << "Pass" << std::endl;
    else    
        std::cout << "***Fail*** Missing constraints: " << failNum << std::endl;
    
    return ;
}
