#pragma once

#include "../CommonTypes.h"

#include "DPredWrapper.h"
#include "HostToKernel.h"

void setPredWrapperConstant( const DPredWrapper &hostPredWrapper );

__global__ void
kerVoteForPoint
(
KerIntArray vertexTriVec,
Tri*        triArr,
int*        vertCircleArr,
int*        triCircleArr,
int         noSample
)
;
__global__ void
kerSplitPointsFast
(
KerIntArray vertTriArr,
int*        triToVert,
Tri*        triArr,
int*        insTriMap,
int*        exactCheckArr, 
int*        counter,
int         triNum,
int         insTriNum
)
;
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
;
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
;
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
;
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
;
__global__ void
kerRelocatePointsFast
(
KerIntArray vertTriVec,
int*        triToFlip,
FlipItem*   flipArr,
int*        exactCheckArr, 
int*        counter
)
;
__global__ void
kerRelocatePointsExact
(
int*        vertTriVec,
int*        triToFlip,
FlipItem*   flipArr,
int*        exactCheckArr, 
int*        counter
)
;
__global__ void 
kerInitPointLocationFast
(
KerIntArray vertTriVec, 
int*        exactCheckArr, 
int*        counter,
Tri         tri
)
;
__global__ void 
kerInitPointLocationExact
(
int*    vertTriArr, 
int*    exactCheckArr, 
int*    counter,
Tri     tri
)
;
__global__ void
kerCheckConsFlipping
(
KerIntArray actTriVec, 
int*        triConsArr, 
char*       triInfoArr, 
Tri*        triArr,
TriOpp*     oppArr, 
int*        triVoteArr
)
;
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
;
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
;
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
;
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
;
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
;
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
;