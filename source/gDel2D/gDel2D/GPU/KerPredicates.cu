#include "KerPredicates.h"

#include "KerCommon.h"
#include "DPredWrapper.h"

#ifndef __CUDACC__
#define __launch_bounds__( x )
#endif

__constant__ DPredWrapper dPredWrapper; 

#include "KerPredWrapper.h"

void setPredWrapperConstant( const DPredWrapper &hostPredWrapper ) 
{
    CudaSafeCall( cudaMemcpyToSymbol( dPredWrapper, &hostPredWrapper, sizeof( hostPredWrapper ) ) ); 
}

///////////////////////////////////////////////////////////////////////////////

template< typename T>
__forceinline__ __device__ void storeIntoBuffer( T* s_buffer, int* s_num, const T& item )
{
    const int idx = atomicAdd( s_num, 1 ); 

    s_buffer[ idx ] = item; 
}

template< typename T >
__forceinline__ __device__ void flushBuffer
(
T*      s_buffer, 
int&    s_offset, 
int&    s_num, 
T*      g_output,
int&    g_counter,
bool    force
)
{
    __syncthreads(); 

    if ( (force && s_num > 0) || s_num >= BLOCK_DIM ) 
    {
        int writeNum = ( s_num >= BLOCK_DIM ) ? BLOCK_DIM : s_num; 

        if ( THREAD_IDX == 0 ) 
            s_offset = atomicAdd( &g_counter, writeNum ); 

        __syncthreads(); 

        if ( THREAD_IDX < writeNum ) 
            g_output[ s_offset + THREAD_IDX ] = s_buffer[ THREAD_IDX ]; 

        if ( THREAD_IDX < s_num - BLOCK_DIM )
            s_buffer[ THREAD_IDX ] = s_buffer[ BLOCK_DIM + THREAD_IDX ]; 

        __syncthreads(); 

        if ( THREAD_IDX == 0 ) 
            s_num -= writeNum; 

        __syncthreads(); 
    }
}

///////////////////////////////////////////////////////////////////////////////

__device__ const int SplitFaces[6][3] = {
    /*0*/ { 0, 3 },    
    /*1*/ { 2, 3 },                   /*2*/ { 1, 3 },
    /*3*/ { 1, 2 },  /*4*/ { 2, 0 },  /*5*/ { 0, 1 }
}; 

__device__ const int SplitNext[6][2] = {
    { 1, 2 }, 
    { 3, 4 },               { 5, 3 }, 
    { 1, 0 },   { 2, 0 },   { 3, 0 },
}; 

template < bool doFast >
__forceinline__ __device__ bool splitPoints
(
int     vertex,
int&    vertTriIdx,
int*    triToVert,
Tri*    triArr,
int*    insTriMap,
int     triNum,
int     insTriNum
)
{
    int triIdx = vertTriIdx;

    if ( triIdx == -1 ) return true;   // Point inserted

    const int splitVertex = triToVert[ triIdx ];

    if ( splitVertex >= INT_MAX - 1 ) return true; // Vertex's triangle will not be split in this round
        
    if ( doFast && vertex == splitVertex ) // Fast mode, *this* vertex will split its triangle
    {
        vertTriIdx = -1; 
        return true; 
    }

    const Tri tri         = triArr[ triIdx ];
    const int newBeg      = ( triNum >= 0 ) ? ( triNum + 2 * insTriMap[ triIdx ] ) : ( triIdx + 1 );
    const int triVert[4]  = { tri._v[0], tri._v[1], tri._v[2], splitVertex };

    int face = 0; 

    for ( int i = 0; i < 2; ++i ) 
    {
        const int *fv = SplitFaces[ face ]; 

        Orient ort = ( doFast ) 
            ? dPredWrapper.doOrient2DFast( triVert[ fv[0] ], triVert[ fv[1] ], vertex )
            : dPredWrapper.doOrient2DFastExactSoS( triVert[ fv[0] ], triVert[ fv[1] ], vertex );

        if ( doFast && (ort == OrientZero) ) return false; // Needs exact computation

        face = SplitNext[ face ][ ( ort == OrientPos ) ? 0 : 1 ]; 
    }

    vertTriIdx = ( ( face == 3 ) ? triIdx : (newBeg + face - 4) ); 

    return true; 
}

__global__ void
kerSplitPointsFast
(
KerIntArray vertTriVec,
int*        triToVert,
Tri*        triArr,
int*        insTriMap,
int*        exactCheckArr, 
int*        counter,
int         triNum,
int         insTriNum
)
{
    // Iterate points
    for ( int idx = getCurThreadIdx(); idx < vertTriVec._num; idx += getThreadNum() )
    {
        bool ret = splitPoints<true>( idx, vertTriVec._arr[ idx ], triToVert, triArr, 
            insTriMap, triNum, insTriNum ); 

        if ( !ret ) 
            storeIntoBuffer( exactCheckArr, &counter[ CounterExact ], idx ); 
    }
}

__global__ void
kerSplitPointsExactSoS
(
int*    vertTriArr,
int*    triToVert,
Tri*    triArr,
int*    insTriMap,
int*    exactCheckArr, 
int*    counter,
int     triNum,
int     insTriNum
)
{
    const int exactNum = counter[ CounterExact ]; 

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        const int vertIdx = exactCheckArr[ idx ]; 

        splitPoints< false >( vertIdx, vertTriArr[ vertIdx ], triToVert, 
            triArr, insTriMap, triNum, insTriNum ); 
    }
}

template < CheckDelaunayMode checkMode >
__forceinline__ __device__ void
checkDelaunayFast
(
int*    actTriArr,
Tri*    triArr,
TriOpp* oppArr,
char*   triInfoArr,
int*    triVoteArr,
int2*   exactCheckVi, 
int     actTriNum,
int*    counter,
int*    dbgCircleCountArr
)
{
    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < actTriNum; idx += getThreadNum() )
    {
        const int botTi = actTriArr[ idx ];

        ////
        // Check which side needs to be checked
        ////
        int checkVi         = 1; 
        const TriOpp botOpp = oppArr[ botTi ];

        for ( int botVi = 0; botVi < DEG; ++botVi ) 
            if ( -1 != botOpp._t[ botVi ]               // No neighbour at this face
                && !botOpp.isOppConstraint( botVi ) )   // or neighbor is a constraint
            {
                const int topTi = botOpp.getOppTri( botVi );
                const int topVi = botOpp.getOppVi( botVi );               

                if ( ( ( botTi < topTi ) || 
                        Checked == getTriCheckState( triInfoArr[ topTi ] ) ) )
                    checkVi = (checkVi << 2) | botVi; 
            }

        // Nothing to check?
        if ( checkVi != 1 ) 
        {
            ////
            // Do circle check
            ////
            const Tri botTri = triArr[ botTi ];

            int dbgCount = 0;
            bool hasFlip = false; 
            int exactVi  = 1; 

            // Check 2-3 flips
            for ( ; checkVi > 1; checkVi >>= 2 )
            {            
                const int botVi   = ( checkVi & 3 ); 
                const int topTi   = botOpp.getOppTri( botVi );
                const int topVi   = botOpp.getOppVi( botVi );
                    
                const int topVert = triArr[ topTi ]._v[ topVi ]; 

                Side side = dPredWrapper.doInCircleFast( botTri, topVert ); 

                ++dbgCount;

                if ( SideZero == side )
                    if ( checkMode == CircleFastOrientFast ) 
                        // Store for future exact mode
                        oppArr[ botTi ].setOppSpecial( botVi, true );      
                    else
                        // Pass to next kernel - exact kernel
                        exactVi = (exactVi << 2) | botVi;            
            
                if ( SideIn != side ) continue; // No incircle failure at this face

                // We have incircle failure, vote!
                voteForFlip( triVoteArr, botTi, topTi, botVi );
                hasFlip = true; 
                break; 
            }

            if ( ( checkMode == CircleExactOrientSoS ) && ( !hasFlip ) && ( exactVi != 1 ) )
                storeIntoBuffer( exactCheckVi, &counter[ CounterExact ], make_int2( botTi, exactVi ) ); 

            if ( NULL != dbgCircleCountArr )
                dbgCircleCountArr[ botTi ] = dbgCount;
        }
    }

    return;
}

__global__ void
kerCheckDelaunayFast
(
int*    actTriArr,
Tri*    triArr,
TriOpp* oppArr,
char*   triInfoArr,
int*    triVoteArr,
int     actTriNum,
int*    dbgCircleCountArr
)
{
    checkDelaunayFast< CircleFastOrientFast >(
        actTriArr,
        triArr,
        oppArr,
        triInfoArr,
        triVoteArr,
        NULL,
        actTriNum,
        NULL,
        dbgCircleCountArr );
    return;
}

__global__ void
kerCheckDelaunayExact_Fast
(
int*    actTriArr,
Tri*    triArr,
TriOpp* oppArr,
char*   triInfoArr,
int*    triVoteArr,
int2*   exactCheckVi, 
int     actTriNum,
int*    counter,
int*    dbgCircleCountArr
)
{
    checkDelaunayFast< CircleExactOrientSoS >(
        actTriArr,
        triArr,
        oppArr,
        triInfoArr,
        triVoteArr,
        exactCheckVi, 
        actTriNum,
        counter,
        dbgCircleCountArr );
    return;
}

__global__ void
kerCheckDelaunayExact_Exact
(
Tri*    triArr,
TriOpp* oppArr,
int*    triVoteArr,
int2*   exactCheckVi,
int*    counter,
int*    dbgCircleCountArr
)
{
    const int exactNum = counter[ CounterExact ]; 

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        int2 val    = exactCheckVi[ idx ]; 
        int botTi   = val.x; 
        int exactVi = val.y; 

        exactCheckVi[ idx ] = make_int2( -1, -1 ); 

        ////
        // Do circle check
        ////
        TriOpp botOpp    = oppArr[ botTi ];
        const Tri botTri = triArr[ botTi ];

        int dbgCount    = 0;

        if ( NULL != dbgCircleCountArr ) 
            dbgCount = dbgCircleCountArr[ botTi ]; 

        for ( ; exactVi > 1; exactVi >>= 2 )
        {            
            const int botVi = ( exactVi & 3 ); 

            const int topTi     = botOpp.getOppTri( botVi );
            const int topVi     = botOpp.getOppVi( botVi );               
            const int topVert   = triArr[ topTi ]._v[ topVi ];

            const Side side = dPredWrapper.doInCircleFastExactSoS( botTri, topVert );

            ++dbgCount; 

            if ( SideIn != side ) continue; // No incircle failure at this face

            voteForFlip( triVoteArr, botTi, topTi, botVi );
            break; 
        } // Check faces of triangle

        if ( NULL != dbgCircleCountArr )
            dbgCircleCountArr[ botTi ] = dbgCount;
    }

    return;
}

__device__ int setNeedExact( int val ) 
{
    return val | ( 1 << 31 ); 
}

__device__ int removeExactBit( int val ) 
{
    return ( val & ~(1 << 31) ); 
}

__device__ bool isNeedExact( int val ) 
{
    return ( val >> 31 ) & 1; 
}

template<bool doFast>
__forceinline__ __device__ bool
relocatePoints
(
int         vertex,
int&        location,
int*        triToFlip,
FlipItem*   flipArr
)
{
    const int triIdx = location;

    if ( triIdx == -1 ) return true; 

    int nextIdx = ( doFast ) ? triToFlip[ triIdx ] : triIdx; 

    if ( nextIdx == -1 ) return true;   // No flip 

    int flag              = nextIdx & 1; 
    int destIdx           = nextIdx >> 1; 

    while ( flag == 1 ) 
    {
        const FlipItem flipItem = loadFlip( flipArr, destIdx ); 
        
        const Orient ord = doFast 
            ? dPredWrapper.doOrient2DFast( flipItem._v[ 0 ], flipItem._v[ 1 ], vertex )
            : dPredWrapper.doOrient2DFastExactSoS( flipItem._v[ 0 ], flipItem._v[ 1 ], vertex );

        if ( doFast && ( OrientZero == ord ) )
        {
            location = nextIdx; 
            return false;  
        }

        nextIdx = flipItem._t[ ( OrientPos == ord ) ? 1 : 0 ]; 
        flag    = nextIdx & 1; 
        destIdx = nextIdx >> 1; 
    }

    location = destIdx; // Write back

    return true;
}

__global__ void
kerRelocatePointsFast
(
KerIntArray vertTriVec,
int*        triToFlip,
FlipItem*   flipArr,
int*        exactCheckArr, 
int*        counter
)
{
    // Iterate points
    for ( int idx = getCurThreadIdx(); idx < vertTriVec._num; idx += getThreadNum() )
    {
        bool ret = relocatePoints<true>( idx, vertTriVec._arr[ idx ], triToFlip, flipArr ); 

        if ( !ret ) 
            storeIntoBuffer( exactCheckArr, &counter[ CounterExact ], idx ); 
    }
}

__global__ void
kerRelocatePointsExact
(
int*        vertTriArr,
int*        triToFlip,
FlipItem*   flipArr,
int*        exactCheckArr,
int*        counter
)
{
    const int exactNum = counter[ CounterExact ]; 

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        const int vertIdx = exactCheckArr[ idx ]; 

        relocatePoints< false >( vertIdx, vertTriArr[ vertIdx ], triToFlip, flipArr ); 
    }
}

template<bool doFast> 
__forceinline__ __device__ int initPointLocation( int idx, Tri tri )
{
    if ( tri.has( idx ) || idx == dPredWrapper._infIdx ) return -1;   // Already inserted        

    const int triVert[5] = { tri._v[0], tri._v[1], tri._v[2], dPredWrapper._infIdx };
    int face = 0; 

    for ( int i = 0; i < 3; ++i ) 
    {
        const int *fv = SplitFaces[ face ]; 

        Orient ort = ( doFast ) 
            ? dPredWrapper.doOrient2DFast( triVert[ fv[0] ], triVert[ fv[1] ], idx )
            : dPredWrapper.doOrient2DFastExactSoS( triVert[ fv[0] ], triVert[ fv[1] ], idx );

        if ( doFast && (ort == OrientZero) ) return -2;    // Needs exact computation

        // Use the reverse direction 'cause the splitting point is Infty!
        face = SplitNext[ face ][ ( ort == OrientPos ) ? 1 : 0 ]; 
    }

    return face; 
}

__global__ void kerInitPointLocationFast
(
KerIntArray vertTriVec, 
int*        exactCheckArr, 
int*        counter,
Tri         tri
)
{
    // Iterate points
    for ( int idx = getCurThreadIdx(); idx < vertTriVec._num; idx += getThreadNum() )
    {
        const int loc = initPointLocation<true>( idx, tri ); 

        if ( loc != -2 ) 
            vertTriVec._arr[ idx ] = loc; 
        else 
            storeIntoBuffer( exactCheckArr, &counter[ CounterExact ], idx ); 
    }
}

__global__ void kerInitPointLocationExact
(
int*    vertTriArr, 
int*    exactCheckArr, 
int*    counter,
Tri     tri
)
{
    const int exactNum = counter[ CounterExact ]; 

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        const int vertIdx = exactCheckArr[ idx ]; 

        vertTriArr[ vertIdx ] = initPointLocation<false>( vertIdx, tri ); 
    }
}

__forceinline__ __device__ int hash( int x ) 
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
}

__global__ void
kerVoteForPoint
(
KerIntArray vertexTriVec,
Tri*        triArr,
int*        vertCircleArr,
int*        triCircleArr,
int         noSample
)
{
    const float rate = float(vertexTriVec._num) / noSample; 

    // Iterate uninserted points
    for ( int idx = getCurThreadIdx(); idx < noSample; idx += getThreadNum() )
    {
        const int vert = int( idx * rate ); 

        //*** Compute insphere value
        const int triIdx   = vertexTriVec._arr[ vert ];

        if ( -1 == triIdx ) continue; 

        const Tri tri = triArr[ triIdx ];
        float cval    = /*hash( idx );*/ dPredWrapper.inCircleDet( tri, vert ); 
        
        //*** Sanitize and store sphere value

        if ( cval <= 0 )
            cval = 0;

        int ival = __float_as_int(cval); 

        vertCircleArr[ idx ] =  ival;

        //*** Vote     
        if ( triCircleArr[ triIdx ] < ival ) 
            atomicMax( &triCircleArr[ triIdx ], ival );
    }

    return;
}

template< bool doFast > 
__forceinline__ __device__ void 
markTriConsIntersection
(
KerIntArray actConsVec, 
Segment*    constraintArr, 
Tri*        triArr, 
TriOpp*     oppArr, 
char*       triInfoArr,
int*        vertTriArr, 
int*        triConsArr, 
int*        counter
)
{
    // Iterate the constraints
    for ( int idx = getCurThreadIdx(); idx < actConsVec._num; idx += getThreadNum() )
    {
        int consIdx = actConsVec._arr[ idx ]; 

        if ( consIdx == -1 ) continue;  // Already inserted

        if ( !doFast ) 
            if ( consIdx >= 0 ) continue; 
            else actConsVec._arr[ idx ] = consIdx = makePositive( consIdx ); 

        Segment constraint = constraintArr[ consIdx ]; 

        int triIdx = vertTriArr[ constraint._v[0] ]; 
        int vi     = triArr[ triIdx ].getIndexOf( constraint._v[0] ); 

        // Walk around the starting vertex to find the intersected triangle
        // Should always find one, so the loop is not infinite!
        bool found = false;         // The constraint is found in triangulation? 
        Orient or1 = OrientZero; 
        Orient or2; 

        while ( true ) 
        {
            const Tri& tri = triArr[ triIdx ]; 

            // Check for intersections
            const int nextVert = tri._v[ (vi + 2) % 3 ]; 

            // Infinity
            //if ( nextVert == dPredWrapper._infIdx )         
            //    or1 = OrientZero; 

            // The constraint is already inserted
            if ( nextVert == constraint._v[ 1 ] ) { found = true; break; }

            else
            {                
                or2 = doFast
                    ? dPredWrapper.doOrient2DFast        ( constraint._v[0], constraint._v[1], nextVert )
                    : dPredWrapper.doOrient2DFastExactSoS( constraint._v[0], constraint._v[1], nextVert ); 

                if ( doFast && or2 == OrientZero ) break; 

                if ( or1 == OrientNeg && or2 == OrientPos ) 
                {
                    const int label = encode_constraint( consIdx, vi, 2 );  // side = 2 --> starting triangle
                    atomicMax( &triConsArr[ triIdx ], label ); 
                    setTriCheckState( triInfoArr[ triIdx ], Changed ); 
                    break; 
                }

                or1 = or2; 
            }

            // Rotate
            const TriOpp& opp = oppArr[ triIdx ]; 

            triIdx  = opp.getOppTri( ( vi + 1 ) % DEG ); 
            vi      = opp.getOppVi( ( vi + 1 ) % DEG ); 
            vi      = ( vi + 1 ) % DEG; 
        }

        if ( found )     // Already in the triangulation
        {
            // Mark constraint as found
            actConsVec._arr[ idx ] = -1;

            // Label the edge in the opp array
            vi          = ( vi + 1 ) % 3; 
            TriOpp &opp = oppArr[ triIdx ]; 

            opp.setOppConstraint( vi, true ); 

            triIdx = opp.getOppTri( vi ); 
            vi     = opp.getOppVi( vi ); 

            oppArr[ triIdx ].setOppConstraint( vi, true ); 

            continue;   
        }

        if ( or2 == OrientZero ) 
        { 
            actConsVec._arr[ idx ] = makeNegative( consIdx ); 
            continue; 
        }
        
        // Found the first triangle, walking to find the rest of the intersected ones
        counter[ CounterFlag ] = 1; 

        int side = 0; 

        //if ( consIdx == 1176931 ) 
        //    printf( "Cons %i (%i): %i (%i %i %i)\n ",  consIdx, doFast, triIdx, 
        //        triArr[triIdx]._v[0], triArr[triIdx]._v[1], triArr[triIdx]._v[2] ); 

        // Terminate inside when constraint ends
        while ( side != 3 )
        {
            const TriOpp& opp = oppArr[ triIdx ]; 

            triIdx = opp.getOppTri( vi ); 
            vi     = opp.getOppVi( vi ); 

            const Tri& tri      = triArr[ triIdx ]; 
            const int nextVert  = tri._v[ vi ]; 

            //if ( consIdx == 1176931 ) 
            //    printf("--> %i ", nextVert); 

            //CudaAssert( nextVert != dPredWrapper._infIdx ); 

            if ( nextVert == constraint._v[ 1 ] )   // Reach the end
            { vi = (vi + 2) % 3; side = 3; }
            else 
            {
                Orient ori = doFast
                    ? dPredWrapper.doOrient2DFast        ( constraint._v[0], constraint._v[1], nextVert )
                    : dPredWrapper.doOrient2DFastExactSoS( constraint._v[0], constraint._v[1], nextVert ); 

                if ( doFast && ori == OrientZero ) break;

                if ( ori == OrientNeg ) 
                { vi = (vi + 2) % 3; side = 0; }
                else
                { vi = (vi + 1) % 3; side = 1; }
            }

            const int label = encode_constraint( consIdx, vi, side );
            atomicMax( &triConsArr[ triIdx ], label ); 
            setTriCheckState( triInfoArr[ triIdx ], Changed ); 
        }

        //if ( consIdx == 1176931 ) 
        //    printf("\n"); 

        if ( doFast && side != 3 ) 
            actConsVec._arr[ idx ] = makeNegative( consIdx ); 
    }
}

__global__ void 
kerMarkTriConsIntersectionFast
(
KerIntArray actConsVec, 
Segment*    constraintArr, 
Tri*        triArr, 
TriOpp*     oppArr, 
char*       triInfoArr,
int*        vertTriArr, 
int*        triConsArr, 
int*        counter
)
{
    markTriConsIntersection<true>(
        actConsVec, 
        constraintArr, 
        triArr, 
        oppArr, 
        triInfoArr, 
        vertTriArr, 
        triConsArr, 
        counter
        ); 
}

__global__ void 
kerMarkTriConsIntersectionExact
(
KerIntArray actConsVec, 
Segment*    constraintArr, 
Tri*        triArr, 
TriOpp*     oppArr, 
char*       triInfoArr,
int*        vertTriArr, 
int*        triConsArr, 
int*        counter
)
{
    markTriConsIntersection<false>(
        actConsVec, 
        constraintArr, 
        triArr, 
        oppArr, 
        triInfoArr, 
        vertTriArr, 
        triConsArr, 
        counter
        ); 
}

template< bool doFast >
__forceinline__ __device__ bool 
updatePairStatus
(
int     triIdx, 
int*    triConsVec, 
Tri*    triArr, 
TriOpp* oppArr, 
char*   triInfoArr
)
{
    if ( doFast ) 
    {
        if ( getTriCheckState( triInfoArr[ triIdx ] ) != Changed ) return true;    // No changes

        setTriPairType( triInfoArr[ triIdx ], PairNone ); 
    }

    const int label = triConsVec[ triIdx ]; 

    if ( doFast && label < 0 ) return true;      // No longer intersect constraints, at least for this round

    const int consIdx = decode_cIdx( label );
    const int vi      = decode_cVi( label ); 
    const int side    = decode_cSide( label ); 

    if ( doFast && side == 3 ) return true;    // Last triangle

    TriOpp& opp = oppArr[ triIdx ]; 

    const int oppIdx     = opp.getOppTri( vi ); 
    const int oppVi      = opp.getOppVi( vi ); 
    const int oppLabel   = triConsVec[ oppIdx ]; 
    const int oppConsIdx = decode_cIdx( oppLabel ); 

    if ( doFast && consIdx != oppConsIdx ) return true;      // Different labels

    // Initialize as Single/Zero configuration
    PairType type = PairSingle; 

    const int oppVert = triArr[ oppIdx ]._v[ oppVi ]; 
    const Tri tri     = triArr[ triIdx ]; 

    // Check whether there's a concave configuration
    Orient or1 = doFast
        ? dPredWrapper.doOrient2DFast        ( tri._v[ vi ], oppVert, tri._v[ (vi + 1) % 3 ] )
        : dPredWrapper.doOrient2DFastExactSoS( tri._v[ vi ], oppVert, tri._v[ (vi + 1) % 3 ] ); 
    Orient or2 = doFast
        ? dPredWrapper.doOrient2DFast        ( tri._v[ vi ], oppVert, tri._v[ (vi + 2) % 3 ] )
        : dPredWrapper.doOrient2DFastExactSoS( tri._v[ vi ], oppVert, tri._v[ (vi + 2) % 3 ] ); 

    // Need exact computation?
    if ( doFast && ( or1 == OrientZero || or2 == OrientZero ) ) return false;   

    if ( or1 == or2 )   // Concave
        type = PairConcave; 
    else
    {
        // Check whether there's a double-intersection configuration
        const int oppVertSide = ( decode_cVi( oppLabel ) == ( oppVi + 2 ) % 3 ? 0 : 1 ); 

        if ( side < 2 && decode_cSide( oppLabel ) < 2 && side != oppVertSide ) 
            type = PairDouble; 
    }

    setTriPairType( triInfoArr[ triIdx ], type ); 

    return true; 
}

__global__ void 
kerUpdatePairStatusFast
(
KerIntArray actTriVec, 
int*        triConsVec, 
Tri*        triArr, 
TriOpp*     oppArr, 
char*       triInfoArr,
int*        exactArr, 
int*        counter
)
{
    // Iterate active triangles
    for ( int idx = getCurThreadIdx(); idx < actTriVec._num; idx += getThreadNum() )
    {
        const int triIdx = actTriVec._arr[ idx ]; 

        bool ret = updatePairStatus<true>( triIdx, triConsVec, triArr, oppArr, triInfoArr ); 

        if ( !ret ) 
            storeIntoBuffer( exactArr, &counter[ CounterExact ], triIdx ); 
    }
}

__global__ void 
kerUpdatePairStatusExact
(
KerIntArray actTriVec, 
int*        triConsVec, 
Tri*        triArr, 
TriOpp*     oppArr, 
char*       triInfoArr,
int*        exactArr, 
int*        counter
)
{
    const int exactNum = counter[ CounterExact ]; 

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        const int triIdx = exactArr[ idx ]; 

        updatePairStatus<false>( triIdx, triConsVec, triArr, oppArr, triInfoArr ); 
    }
}

template<bool doFast>
__forceinline__ __device__ bool
checkConsFlipping
(
int     midIdx,
int*    triConsArr, 
char*   triInfoArr, 
Tri*    triArr,
TriOpp* oppArr, 
int*    triVoteArr
)
{
    PairType midType; 

    if ( doFast ) 
    {
        const int triInfo = triInfoArr[ midIdx ]; 
                  midType = getTriPairType( triInfo ); 

        if ( getTriCheckState( triInfo) != Changed ) return true;      // Nothing changed. 

        setTriCheckState( triInfoArr[ midIdx ], Checked ); 

        if ( midType == PairNone || midType == PairConcave ) return true;     // Nothing to flip
    }

    const int midLabel = triConsArr[ midIdx ]; 

    CudaAssert( midLabel >= 0 && "Invalid label." ); 

    const int midVi   = decode_cVi( midLabel ); 
    const int midSide = decode_cSide( midLabel ); 

    const int rightIdx = oppArr[ midIdx ].getOppTri( midVi ); 
    const int rightVi  = oppArr[ midIdx ].getOppVi( midVi ); 
    const int leftIdx  = oppArr[ midIdx ].getOppTri( ( midVi + midSide + 1 ) % 3 ); 

    if ( doFast && midType == PairSingle )   // Case 1
    {
        const int vote = makeConsFlipVote( midIdx, PriorityCase1 ); 
        atomicMin( &triVoteArr[ midIdx ], vote ); 
        atomicMin( &triVoteArr[ rightIdx ], vote ); 
        return true; 
    }

    CudaAssert( midSide < 2 );

    // midType = PairDouble. Check for Case 2 and 3. Look back one step.
    const int leftLabel = triConsArr[ leftIdx ]; 

    // Check if two pairs are of the same constraint
    if ( doFast && ( leftLabel < 0 || decode_cIdx( leftLabel ) != decode_cIdx( midLabel ) ) ) return true; 

    const PairType leftType = getTriPairType( triInfoArr[ leftIdx ] ); 

    CudaAssert( leftType != PairNone && "Cannot be a PairNone." ); 

    if ( doFast && leftType == PairSingle ) return true; 

    const int leftVi    = decode_cVi( leftLabel ); 
    const Tri tri       = triArr[ leftIdx ]; 
    const int rightVert = triArr[ rightIdx ]._v[ rightVi ]; 

    // Check whether after flipping the left pair is a concave configuration
    Orient or1 = doFast
        ? dPredWrapper.doOrient2DFast        ( tri._v[ leftVi ], rightVert, tri._v[ (leftVi + 1) % 3 ] )
        : dPredWrapper.doOrient2DFastExactSoS( tri._v[ leftVi ], rightVert, tri._v[ (leftVi + 1) % 3 ] );
    Orient or2 = doFast 
        ? dPredWrapper.doOrient2DFast        ( tri._v[ leftVi ], rightVert, tri._v[ (leftVi + 2) % 3 ] )
        : dPredWrapper.doOrient2DFastExactSoS( tri._v[ leftVi ], rightVert, tri._v[ (leftVi + 2) % 3 ] ); 

    if ( doFast && ( or1 == OrientZero || or2 == OrientZero ) ) return false; // Need exact check

    if ( or1 == or2 ) return true;  // Concave, neither Case 2 nor 3

    int vote; 

    if ( leftType == PairDouble )       // Case 2
        vote = makeConsFlipVote( midIdx, PriorityCase2 ); 
    else  // leftType == PairConcave --> Case 3
        vote = makeConsFlipVote( midIdx, PriorityCase3 ); 

    atomicMin( &triVoteArr[ leftIdx ],  vote ); 
    atomicMin( &triVoteArr[ midIdx ],   vote ); 
    atomicMin( &triVoteArr[ rightIdx ], vote ); 

    return true; 
}

__global__ void
kerCheckConsFlippingFast
(
KerIntArray actTriVec, 
int*        triConsArr, 
char*       triInfoArr, 
Tri*        triArr,
TriOpp*     oppArr, 
int*        triVoteArr,
int*        exactArr, 
int*        counter
)
{
    // Iterate active triangles
    for ( int idx = getCurThreadIdx(); idx < actTriVec._num; idx += getThreadNum() )
    {
        const int triIdx = actTriVec._arr[ idx ]; 

        bool ret = checkConsFlipping<true>( triIdx, triConsArr, triInfoArr, triArr, oppArr, triVoteArr ); 

        if ( !ret ) 
            storeIntoBuffer( exactArr, &counter[ CounterExact ], triIdx ); 
    }
}

__global__ void
kerCheckConsFlippingExact
(
KerIntArray actTriVec, 
int*        triConsArr, 
char*       triInfoArr, 
Tri*        triArr,
TriOpp*     oppArr, 
int*        triVoteArr,
int*        exactArr, 
int*        counter
)
{
    const int exactNum = counter[ CounterExact ]; 

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        const int triIdx = exactArr[ idx ]; 

        checkConsFlipping<false>( triIdx, triConsArr, triInfoArr, triArr, oppArr, triVoteArr ); 
    }
}
