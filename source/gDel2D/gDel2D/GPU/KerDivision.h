#pragma once

#include "../CommonTypes.h"

#include "HostToKernel.h"

__global__ void
kerPickWinnerPoint
(
KerIntArray  vertexTriVec,
int*         vertCircleArr,
int*         triCircleArr,
int*         triVertArr,
int          noSample
)
;
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
;
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
;
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
;
__global__ void
kerUpdateOpp
(
FlipItem*    flipVec,
TriOpp*      oppArr,
int2*        triMsgArr,
int*         encodedFaceViArr,
int          orgFlipNum,
int          flipNum
)
; 
__global__ void 
kerUpdateFlipTrace
(
FlipItem*   flipArr, 
int*        triToFlip,
int         orgFlipNum, 
int         flipNum
)
;
__global__ void 
kerUpdateVertIdx
(
KerTriArray     triVec,
char*           triInfoArr,
int*            orgPointIdx
)
;
__global__ void 
kerMarkSpecialTris
(
KerCharArray triInfoVec, 
TriOpp*      oppArr
)
;
__global__ void
kerMakeFirstTri
(
Tri*	triArr,
TriOpp*	oppArr,
char*	tetInfoArr,
Tri     tri,
int     infIdx
)
;
__global__ void 
kerShiftTriIdx
(
KerIntArray idxVec, 
int*        shiftArr
) 
;
__global__ void 
kerShiftOpp
(
KerIntArray shiftVec, 
TriOpp*     src, 
TriOpp*     dest,
int         destSize
) 
;
__global__ void 
kerMarkInfinityTri
(
KerTriArray triVec, 
char*       triInfoArr,
TriOpp*     oppArr,
int         infIdx
)
;
__global__ void 
kerCollectFreeSlots
(
char* triInfoArr, 
int*  prefixArr,
int*  freeArr,
int   newTetNum
)
;
__global__ void
kerMakeCompactMap
(
KerCharArray triInfoVec, 
int*         prefixArr, 
int*         freeArr, 
int          newTriNum
)
;
__global__ void
kerCompactTris
(
KerCharArray triInfoVec, 
int*         prefixArr, 
Tri*         triArr, 
TriOpp*      oppArr, 
int          newTriNum
)
;
__global__ void
kerMapTriToVert
(
KerTriArray triVec,
int*        vertTriArr
)
;
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
;