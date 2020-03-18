#include "KerDivision.h"

#include "KerCommon.h"

template< typename T > 
__forceinline__ __device__ T min( T a, T b ) 
{
    if ( a < b ) 
        return a; 
    else
        return b; 
}

__global__ void
kerSplitTri
(
KerIntArray splitTriArr,
Tri*        triArr,
TriOpp*     oppArr,
char*       triInfoArr,
int*        insTriMap,
int*        triToVert,
int         triNum,
int         insTriNum
)
{
    // Iterate current triangles
    for ( int idx = getCurThreadIdx(); idx < splitTriArr._num; idx += getThreadNum() )
    {
        const int triIdx         = splitTriArr._arr[ idx ]; 
        const int newBeg         = ( triNum >= 0 ) ? ( triNum + 2 * insTriMap[ triIdx ] ) : ( triIdx + 1 );
        const int newTriIdx[DEG] = { triIdx, newBeg, newBeg + 1 };
        TriOpp newOpp[3]         = {
            { -1, -1, -1 },
            { -1, -1, -1 },
            { -1, -1, -1 }
        };

        // Set adjacency of 3 internal faces of 3 new triangles
        newOpp[ 0 ].setOpp( 0, newTriIdx[ 1 ], 1 );
        newOpp[ 0 ].setOpp( 1, newTriIdx[ 2 ], 0 );
        newOpp[ 1 ].setOpp( 0, newTriIdx[ 2 ], 1 );
        newOpp[ 1 ].setOpp( 1, newTriIdx[ 0 ], 0 );
        newOpp[ 2 ].setOpp( 0, newTriIdx[ 0 ], 1 );
        newOpp[ 2 ].setOpp( 1, newTriIdx[ 1 ], 0 );

        // Set adjacency of 4 external faces
        const TriOpp oldOpp       = oppArr[ triIdx ];

        // Iterate faces of old triangle
        for ( int ni = 0; ni < DEG; ++ni )
        {
            if ( -1 == oldOpp._t[ ni ] ) continue; // No neighbour at this face

            int neiTriIdx = oldOpp.getOppTri( ni );
            int neiTriVi  = oldOpp.getOppVi( ni );

            // Check if neighbour has split
            const int neiNewBeg = insTriMap[ neiTriIdx ];

            if ( -1 == neiNewBeg ) // Neighbour is un-split
            {
                oppArr[ neiTriIdx ].setOpp( neiTriVi, newTriIdx[ ni ], 2 ); // Point un-split neighbour back to this new triangle
            }
            else // Neighbour has split
            {
                // Get neighbour's new split triangle that has this face
                if ( triNum >= 0 ) 
                    neiTriIdx = (( 0 == neiTriVi ) ? neiTriIdx : (triNum + 2 * neiNewBeg + neiTriVi - 1));
                else
                    neiTriIdx += neiTriVi; 

                neiTriVi  = 2;
            }

            newOpp[ ni ].setOpp( 2, neiTriIdx, neiTriVi ); // Point this triangle to neighbour
        }

        // Write split triangle and opp
        const Tri tri           = triArr[ triIdx ]; // Note: This slot will be overwritten below
        const int splitVertex   = triToVert[ triIdx ];

        for ( int ti = 0; ti < DEG; ++ti )
        {
            const Tri newTri = {
                tri._v[ ( ti + 1 ) % DEG ],
                tri._v[ ( ti + 2 ) % DEG ],
                splitVertex
            };

            const int toTriIdx = newTriIdx[ ti ];
            triArr[ toTriIdx ] = newTri;
            oppArr[ toTriIdx ] = newOpp[ ti ];
            setTriAliveState( triInfoArr[ toTriIdx ], true );
            setTriCheckState( triInfoArr[ toTriIdx ], Changed );
        }
    }

    return;
}

// Note: triVoteArr should *not* be modified here
__global__ void
kerMarkRejectedFlips
(
int*        actTriArr,
TriOpp*     oppArr,
int*        triVoteArr,
char*       triInfoArr,
int*        flipToTri,
int         actTriNum,
int*        dbgRejFlipArr
)
{
    for ( int idx = getCurThreadIdx(); idx < actTriNum; idx += getThreadNum() )
    {
        int output = -1; 

        const int triIdx  = actTriArr[ idx ]; 
        const int voteVal = triVoteArr[ triIdx ];

        if ( INT_MAX == voteVal )
        {
            setTriCheckState( triInfoArr[ triIdx ], Checked );
            actTriArr[ idx ] = -1; 
        } 
        else 
        {
            int bossTriIdx, botVi; 
        
            decode( voteVal, &bossTriIdx, &botVi );

            if ( bossTriIdx == triIdx ) // Boss of myself
            {
                const TriOpp& opp    = oppArr[ triIdx ];
                const int topTriIdx  = opp.getOppTri( botVi );
                const int topVoteVal = triVoteArr[ topTriIdx ];

                if ( topVoteVal == voteVal ) 
                    output = voteVal;
            }

            if ( NULL != dbgRejFlipArr && output == -1 ) 
                dbgRejFlipArr[ triIdx ] = 1;
        }

        flipToTri[ idx ] = output; 
    }

    return;
}

__global__ void
kerFlip
(
KerIntArray flipToTri,
Tri*        triArr,
TriOpp*     oppArr,
char*       triInfoArr,
int2*       triMsgArr,
int*        actTriArr,
FlipItem*   flipArr,
int*        triConsArr, 
int*        vertTriArr,
int         orgFlipNum, 
int         actTriNum
)
{
    // Iterate flips
    for ( int flipIdx = getCurThreadIdx(); flipIdx < flipToTri._num; flipIdx += getThreadNum() )
    {
        int botIdx, botVi;  

        const int voteVal = flipToTri._arr[ flipIdx ];

        decode( voteVal, &botIdx, &botVi ); 

        // Bottom triangle
        Tri botTri            = triArr[ botIdx ];
        const TriOpp& botOpp  = oppArr[ botIdx ];

        // Top triangle
        const int topIdx = botOpp.getOppTri( botVi );
        const int topVi  = botOpp.getOppVi( botVi );
        Tri topTri       = triArr[ topIdx ];

        const int globFlipIdx = orgFlipNum + flipIdx; 

        const int botAVi = ( botVi + 1 ) % 3; 
        const int botBVi = ( botVi + 2 ) % 3; 
        const int topAVi = ( topVi + 2 ) % 3; 
        const int topBVi = ( topVi + 1 ) % 3; 

        // Create new triangle
        const int topVert = topTri._v[ topVi ];
        const int botVert = botTri._v[ botVi ];
        const int botA    = botTri._v[ botAVi ]; 
        const int botB    = botTri._v[ botBVi ];

        // Update the bottom and top triangle
        botTri = makeTri( botVert, botA, topVert ); 
        topTri = makeTri( topVert, botB, botVert ); 

        triArr[ botIdx ] = botTri; 
        triArr[ topIdx ] = topTri; 

        int newBotNei = 0xffff; 
        int newTopNei = 0xffff; 

        setTriIdxVi( newBotNei, botAVi, 1, 0 ); 
        setTriIdxVi( newBotNei, botBVi, 3, 2 ); 
        setTriIdxVi( newTopNei, topAVi, 3, 2 ); 
        setTriIdxVi( newTopNei, topBVi, 0, 0 ); 
    
        // Write down the new triangle idx
        triMsgArr[ botIdx ] = make_int2( newBotNei, globFlipIdx ); 
        triMsgArr[ topIdx ] = make_int2( newTopNei, globFlipIdx ); 

        // Record the flip
        FlipItem flipItem = { botVert, topVert, botIdx, topIdx };
        storeFlip( flipArr, globFlipIdx, flipItem ); 

        // Prepare for the next round
        if ( actTriArr != NULL ) 
            actTriArr[ actTriNum + flipIdx ] = 
                ( Checked == getTriCheckState( triInfoArr[ topIdx ] ) )
                ? topIdx : -1;

        if ( triConsArr == NULL )       // Standard flipping
            triInfoArr[ topIdx ] = 3;  // Alive + Changed
        else
        {
            vertTriArr[ botA ] = botIdx; 
            vertTriArr[ botB ] = topIdx; 

            // Update constraint intersection info
            int botLabel = triConsArr[ botIdx ]; 
            int topLabel = triConsArr[ topIdx ]; 

            const int consIdx   = decode_cIdx( botLabel ); 
            const int botSide   = decode_cSide( botLabel ); 
                  int topSide   = decode_cSide( topLabel );

            if ( topSide < 2 )  // Not the last triangle
                topSide = ( decode_cVi( topLabel ) == topAVi ? 0 : 1 ); 

            switch ( botSide )      // Cannto be 3
            {
            case 0: 
                switch ( topSide ) 
                {
                case 0: 
                    botLabel = -1; 
                    topLabel = encode_constraint( consIdx, 2, 0 ); 
                    break; 
                case 1: 
                    botLabel = encode_constraint( consIdx, 0, 0 ); 
                    topLabel = encode_constraint( consIdx, 1, 1 ); 
                    break; 
                case 3: 
                    botLabel = -1; 
                    topLabel = encode_constraint( consIdx, 0, 3 ); 
                    break; 
                }
                break; 
            case 1: 
                switch ( topSide ) 
                {
                case 0: 
                    botLabel = encode_constraint( consIdx, 1, 0 ); 
                    topLabel = encode_constraint( consIdx, 2, 1 ); 
                    break; 
                case 1: 
                    botLabel = encode_constraint( consIdx, 0, 1 ); 
                    topLabel = -1; 
                    break; 
                case 3: 
                    botLabel = encode_constraint( consIdx, 2, 3 ); 
                    topLabel = -1; 
                    break; 
                }
                break; 
            case 2: 
                botLabel = ( topSide == 1 ? encode_constraint( consIdx, 0, 2 ) : -1 ); 
                topLabel = ( topSide == 0 ? encode_constraint( consIdx, 2, 2 ) : -1 ); 
                break; 
            }

            triConsArr[ botIdx ] = botLabel; 
            triConsArr[ topIdx ] = topLabel; 
        }
    }

    return;
}

__global__ void
kerUpdateOpp
(
FlipItem*    flipVec,
TriOpp*      oppArr,
int2*        triMsgArr,
int*         flipToTri,
int          orgFlipNum,
int          flipNum
)
{
    // Iterate flips
    for ( int flipIdx = getCurThreadIdx(); flipIdx < flipNum; flipIdx += getThreadNum() )
    {
        int botIdx, botVi;  

        int voteVal = flipToTri[ flipIdx ];

        decode( voteVal, &botIdx, &botVi ); 

        int     extOpp[4]; 
        TriOpp  opp; 

        opp = oppArr[ botIdx ]; 

        extOpp[ 0 ] = opp.getOppTriVi( (botVi + 1) % 3 ); 
        extOpp[ 1 ] = opp.getOppTriVi( (botVi + 2) % 3 ); 

        int topIdx      = opp.getOppTri( botVi ); 
        const int topVi = opp.getOppVi( botVi ); 
        
        opp = oppArr[ topIdx ]; 

        extOpp[ 2 ] = opp.getOppTriVi( (topVi + 2) % 3 ); 
        extOpp[ 3 ] = opp.getOppTriVi( (topVi + 1) % 3 ); 

        // Ok, update with neighbors
        for ( int i = 0; i < 4; ++i ) 
        {
            int newTriIdx, vi; 
            int triOpp  = extOpp[ i ]; 
            bool isCons = isOppValConstraint( triOpp ); 

            // No neighbor
            if ( -1 == triOpp ) continue; 

            int oppIdx = getOppValTri( triOpp ); 
            int oppVi  = getOppValVi( triOpp ); 
        
            const int2 msg = triMsgArr[ oppIdx ]; 

            if ( msg.y < orgFlipNum )    // Neighbor not flipped
            {
                // Set my neighbor's opp
                newTriIdx = ( (i & 1) == 0 ? topIdx : botIdx ); 
                vi        = ( i == 0 || i == 3 ) ? 0 : 2; 

                oppArr[ oppIdx ].setOpp( oppVi, newTriIdx, vi, isCons ); 
            }
            else
            {
                const int oppFlipIdx = msg.y - orgFlipNum; 

                // Update my own opp
                const int newLocOppIdx = getTriIdx( msg.x, oppVi ); 
                    
                if ( newLocOppIdx != 3 ) 
                    oppIdx = flipVec[ oppFlipIdx ]._t[ newLocOppIdx ]; 

                oppVi = getTriVi( msg.x, oppVi ); 

                setOppValTriVi( extOpp[ i ], oppIdx, oppVi );
            }
        }

        // Now output
        opp._t[ 0 ] = extOpp[ 3 ]; 
        opp.setOpp( 1, topIdx, 1 ); 
        opp._t[ 2 ] = extOpp[ 1 ]; 

        oppArr[ botIdx ] = opp; 

        opp._t[ 0 ] = extOpp[ 0 ]; 
        opp.setOpp( 1, botIdx, 1 ); 
        opp._t[ 2 ] = extOpp[ 2 ]; 

        oppArr[ topIdx ] = opp; 
    }   

    return;
}

__global__ void 
kerUpdateFlipTrace
(
FlipItem*   flipArr, 
int*        triToFlip,
int         orgFlipNum, 
int         flipNum
)
{
    for ( int idx = getCurThreadIdx(); idx < flipNum; idx += getThreadNum() )
    {
        const int flipIdx   = orgFlipNum + idx; 
        FlipItem flipItem   = loadFlip( flipArr, flipIdx ); 

        int triIdx, nextFlip; 

        triIdx              = flipItem._t[ 0 ]; 
        nextFlip            = triToFlip[ triIdx ]; 
        flipItem._t[ 0 ]    = ( nextFlip == -1 ) ? ( triIdx << 1 ) | 0 : nextFlip; 
        triToFlip[ triIdx ] = ( flipIdx << 1 ) | 1; 

        triIdx              = flipItem._t[ 1 ]; 
        nextFlip            = triToFlip[ triIdx ]; 
        flipItem._t[ 1 ]    = ( nextFlip == -1 ) ? ( triIdx << 1 ) | 0 : nextFlip; 
        triToFlip[ triIdx ] = ( flipIdx << 1 ) | 1; 

        storeFlip( flipArr, flipIdx, flipItem ); 
    }
}

__global__ void 
kerUpdateVertIdx
(
KerTriArray triVec,
char*       triInfoArr,
int*        orgPointIdx
)
{
    for ( int idx = getCurThreadIdx(); idx < triVec._num; idx += getThreadNum() )
    {
        if ( !isTriAlive( triInfoArr[ idx ] ) ) continue; 

        Tri tri = triVec._arr[ idx ]; 

        for ( int i = 0; i < DEG; ++i ) 
            tri._v[ i ] = orgPointIdx[ tri._v[i] ]; 

        triVec._arr[ idx ] = tri; 
    }
}

__global__ void 
kerShiftTriIdx
(
KerIntArray idxVec, 
int*        shiftArr
) 
{
    for ( int idx = getCurThreadIdx(); idx < idxVec._num; idx += getThreadNum() )
    {
        const int oldIdx = idxVec._arr[ idx ]; 
        
        if ( oldIdx != -1 ) 
            idxVec._arr[ idx ] = oldIdx + shiftArr[ oldIdx ]; 
    }
}

__global__ void 
kerMarkSpecialTris
(
KerCharArray triInfoVec, 
TriOpp*      oppArr
)
{
    for ( int idx = getCurThreadIdx(); idx < triInfoVec._num; idx += getThreadNum() )
    {
        if ( !isTriAlive( triInfoVec._arr[ idx ] ) ) continue; 

        TriOpp opp = oppArr[ idx ]; 

        bool changed = false; 

        for ( int vi = 0; vi < DEG; ++vi ) 
        {
            if ( -1 == opp._t[ vi ] ) continue; 

            if ( opp.isOppSpecial( vi ) ) 
                changed = true; 
        }

        if ( changed ) 
            setTriCheckState( triInfoVec._arr[ idx ], Changed ); 
    }
}

__forceinline__ __device__ float hash( int k ) 
{
    k *= 357913941;
    k ^= k << 24;
    k += ~357913941;
    k ^= k >> 31;
    k ^= k << 31;

    return int_as_float( k ); 
}

__global__ void
kerPickWinnerPoint
(
KerIntArray  vertexTriVec,
int*         vertCircleArr,
int*         triCircleArr,
int*         triVertArr,
int          noSample
)
{
    const float rate = float(vertexTriVec._num) / noSample; 

    // Iterate uninserted points
    for ( int idx = getCurThreadIdx(); idx < noSample; idx += getThreadNum() )
    {
        const int vert   = int( idx * rate ); 
        const int triIdx = vertexTriVec._arr[ vert ];

        if ( triIdx == -1 ) continue; 

        const int vertSVal = vertCircleArr[ idx ];
        const int winSVal  = triCircleArr[ triIdx ];

        // Check if vertex is winner

        if ( winSVal == vertSVal )
            atomicMin( &triVertArr[ triIdx ], vert );
    }

    return;
}

__global__ void
kerMakeFirstTri
(
Tri*	triArr,
TriOpp*	oppArr,
char*	triInfoArr,
Tri     tri,
int     infIdx
)
{
	const Tri tris[] = {
		{ tri._v[0], tri._v[1], tri._v[2] }, 
		{ tri._v[2], tri._v[1], infIdx }, 
		{ tri._v[0], tri._v[2], infIdx }, 
		{ tri._v[1], tri._v[0], infIdx }
	};

	const int oppTri[][3] = {
		{ 1, 2, 3 }, 
		{ 3, 2, 0 }, 
		{ 1, 3, 0 }, 
		{ 2, 1, 0 }
	};

    const int oppVi[][4] = {
        { 2, 2, 2 }, 
        { 1, 0, 0 }, 
        { 1, 0, 1 }, 
        { 1, 0, 2 }
    }; 

	for ( int i = 0; i < 4; ++i ) 
	{
		triArr[ i ]     = tris[ i ]; 
		triInfoArr[ i ] = 1; 

        TriOpp opp = { -1, -1, -1 }; 

		for ( int j = 0; j < 3; ++j ) 
			opp.setOpp( j, oppTri[i][j], oppVi[i][j] ); 

		oppArr[ i ] = opp; 
	}
}

__global__ void 
kerShiftOpp
(
KerIntArray shiftVec, 
TriOpp*     src, 
TriOpp*     dest,
int         destSize
) 
{
    for ( int idx = getCurThreadIdx(); idx < shiftVec._num; idx += getThreadNum() )
    {
        const int shift = shiftVec._arr[ idx ]; 

        TriOpp opp = src[ idx ]; 

        for ( int vi = 0; vi < 3; ++vi ) 
        {
            const int oppTri = opp.getOppTri( vi ); 

            CudaAssert( oppTri >= 0 && oppTri < shiftVec._num ); 
            CudaAssert( oppTri + shiftVec._arr[ oppTri ] < destSize ); 

            opp.setOppTri( vi, oppTri + shiftVec._arr[ oppTri ] ); 
        }
        
        CudaAssert( idx + shift < destSize ); 

        dest[ idx + shift ] = opp; 
    }
}

__global__ void 
kerMarkInfinityTri
(
KerTriArray triVec, 
char*       triInfoArr,
TriOpp*     oppArr,
int         infIdx
)
{
    for ( int idx = getCurThreadIdx(); idx < triVec._num; idx += getThreadNum() )
    {
        if ( !triVec._arr[ idx ].has( infIdx ) ) continue; 

        // Mark as deleted
        setTriAliveState( triInfoArr[ idx ], false ); 

        TriOpp opp = oppArr[ idx ]; 

        for ( int vi = 0; vi < DEG; ++vi ) 
        {
            if ( opp._t[ vi ] < 0 ) continue; 

            const int oppIdx = opp.getOppTri( vi ); 
            const int oppVi  = opp.getOppVi( vi ); 

            oppArr[ oppIdx ]._t[ oppVi ] = -1; 
        }
    }
}

__global__ void 
kerCollectFreeSlots
(
char* triInfoArr, 
int*  prefixArr,
int*  freeArr,
int   newTriNum
)
{
    for ( int idx = getCurThreadIdx(); idx < newTriNum; idx += getThreadNum() )
    {
        if ( isTriAlive( triInfoArr[ idx ] ) ) continue; 

        int freeIdx = idx - prefixArr[ idx ]; 

        freeArr[ freeIdx ] = idx; 
    }
}

__global__ void
kerMakeCompactMap
(
KerCharArray triInfoVec, 
int*         prefixArr, 
int*         freeArr, 
int          newTriNum
)
{
    for ( int idx = newTriNum + getCurThreadIdx(); idx < triInfoVec._num; idx += getThreadNum() )
    {
        if ( !isTriAlive( triInfoVec._arr[ idx ] ) ) 
        {
            prefixArr[ idx ] = -1; 
            continue; 
        }

        int freeIdx     = newTriNum - prefixArr[ idx ]; 
        int newTriIdx   = freeArr[ freeIdx ]; 

        prefixArr[ idx ] = newTriIdx; 
    }
}

__global__ void
kerCompactTris
(
KerCharArray triInfoVec, 
int*         prefixArr, 
Tri*         triArr, 
TriOpp*      oppArr, 
int          newTriNum
)
{
    for ( int idx = newTriNum + getCurThreadIdx(); idx < triInfoVec._num; idx += getThreadNum() )
    {
        int newTriIdx = prefixArr[ idx ]; 

        if ( newTriIdx == -1 ) continue;

        triArr[ newTriIdx ]          = triArr[ idx ];    
        triInfoVec._arr[ newTriIdx ] = triInfoVec._arr[ idx ];    

        TriOpp opp = oppArr[ idx ]; 

        for ( int vi = 0; vi < DEG; ++vi ) 
        {
            if ( opp._t[ vi ] < 0 ) continue; 

            const int oppIdx = opp.getOppTri( vi ); 

            if ( oppIdx >= newTriNum ) 
            {
                const int oppNewIdx = prefixArr[ oppIdx ]; 

                opp.setOppTri( vi, oppNewIdx ); 
            }
            else
            {
                const int oppVi = opp.getOppVi( vi ); 

                oppArr[ oppIdx ].setOppTri( oppVi, newTriIdx ); 
            }
        }

        oppArr[ newTriIdx ] = opp; 
    }
}

__global__ void
kerMapTriToVert
(
KerTriArray triVec,
int*        vertTriArr
)
{
    for ( int idx = getCurThreadIdx(); idx < triVec._num; idx += getThreadNum() )
    {
        Tri tri = triVec._arr[ idx ]; 

        for ( int vi = 0; vi < DEG; ++vi ) 
            vertTriArr[ tri._v[ vi ] ] = idx; 
    }    
}

__global__ void
kerMarkRejectedConsFlips
(
KerIntArray actTriVec,
int*        triConsArr, 
int*        triVoteArr,
char*       triInfoArr,
TriOpp*     oppArr,
int*        flipToTri,
int*        dbgRejFlipArr
)
{
    for ( int idx = getCurThreadIdx(); idx < actTriVec._num; idx += getThreadNum() )
    {
        int output = -1; 

        const int midIdx  = actTriVec._arr[ idx ]; 
        const int voteVal = triVoteArr[ midIdx ];

        if ( INT_MAX != voteVal )
        {
            const int bossTriIdx = getConsFlipVoteIdx( voteVal ); 
            const int priority   = getConsFlipVotePriority( voteVal ); 

            if ( bossTriIdx == midIdx ) // Boss of myself
            {
                const int midLabel = triConsArr[ midIdx ]; 
                const int midVi    = decode_cVi( midLabel ); 
                const int midSide  = decode_cSide( midLabel ); 
                const int rightIdx = oppArr[ midIdx ].getOppTri( midVi ); 
                const int leftIdx  = oppArr[ midIdx ].getOppTri( ( midVi + midSide + 1 ) % 3 ); 

                if ( triVoteArr[ rightIdx ] == voteVal ) 
                {
                    if ( priority == PriorityCase1 ) 
                        output = encode( midIdx, midVi ); 
                    else
                        if ( triVoteArr[ leftIdx ] == voteVal ) 
                            output = encode( midIdx, midVi ); 

                    if ( NULL != dbgRejFlipArr && output == -1 ) 
                        dbgRejFlipArr[ midIdx ] = 1;

                    if ( output != -1 ) 
                    {
                        // Mark all triangles as changed
                        setTriCheckState( triInfoArr[  leftIdx ], Changed ); 
                        setTriCheckState( triInfoArr[   midIdx ], Changed ); 
                        setTriCheckState( triInfoArr[ rightIdx ], Changed ); 

                        const int rightLabel = triConsArr[ rightIdx ]; 

                        if ( decode_cSide( rightLabel ) != 3 )  // Not the last one
                        {
                            const int nextIdx = oppArr[ rightIdx ].getOppTri( decode_cVi( rightLabel ) ); 
                            setTriCheckState( triInfoArr[ nextIdx ], Changed ); 
                        }

                        // NOTE: Only marking the left and the right of the flip is 
                        // not enough, since when flipping we only check the front pair!
                        // This, however, does not affect the correctness since
                        // the remaining pairs will be processed in the next outer loop. 
                    }
                }
            }
        }

        flipToTri[ idx ] = output; 
    }

    return;
}
