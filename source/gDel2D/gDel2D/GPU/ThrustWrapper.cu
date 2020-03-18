#include "ThrustWrapper.h"

#include <map>

#include <thrust/system/cuda/execution_policy.h>

class CachedAllocator
{
private:
    const int BlockSize; 

    typedef std::multimap< std::ptrdiff_t, char * >     FreeBlocks;
    typedef std::map< char *, std::ptrdiff_t >          AllocBlocks;

    FreeBlocks freeBlocks;
    AllocBlocks allocBlocks;

public:
    // just allocate bytes
    typedef char value_type;

    CachedAllocator() 
        : BlockSize( 4096 ) {}

    void freeAll()
    {
        size_t totalSize = 0; 

        // deallocate all outstanding blocks in both lists
        for( FreeBlocks::iterator i = freeBlocks.begin();
             i != freeBlocks.end();
             ++i )
        {
            cudaFree( i->second );
            totalSize += i->first; 
        }

        for( AllocBlocks::iterator i = allocBlocks.begin();
             i != allocBlocks.end();
             ++i )
        {
            cudaFree( i->first );
            totalSize += i->second; 
        }

        freeBlocks.clear(); 
        allocBlocks.clear(); 

        //std::cout << "*** CacheAllocator size: " 
        //    << freeBlocks.size() + allocBlocks.size()
        //    << " Size in bytes: " << totalSize << std::endl;  
    }

    char *allocate( std::ptrdiff_t numBytes )
    {
        char *result    = 0;
        numBytes        = ( ( numBytes - 1 ) / BlockSize + 1 ) * BlockSize; 

        // search the cache for a free block
        FreeBlocks::iterator freeBlock = freeBlocks.find( numBytes );

        if( freeBlock != freeBlocks.end() )
        {
            //std::cout << "CachedAllocator: found a hit " << numBytes << std::endl;

            result = freeBlock->second;

            freeBlocks.erase( freeBlock );
        }
        else
        {
            // no allocation of the right size exists
            // create a new one with cuda::malloc
            // throw if cuda::malloc can't satisfy the request
            try
            {
                //std::cout << "CachedAllocator: no free block found; calling cudaMalloc " << numBytes << std::endl;

                // allocate memory and convert cuda::pointer to raw pointer
                result = thrust::device_malloc<char>( numBytes ).get();
            }
            catch( std::runtime_error &e )
            {
                // output an error message and exit
                std::cerr << "thrust::device_malloc failed to allocate " << numBytes << " bytes!" << std::endl;
                exit( -1 );
            }
        }

        // insert the allocated pointer into the allocated_blocks map
        allocBlocks.insert( std::make_pair( result, numBytes ) );

        return result;
    }

    void deallocate( char *ptr, size_t n )
    {
        // erase the allocated block from the allocated blocks map
        AllocBlocks::iterator iter  = allocBlocks.find( ptr );
        std::ptrdiff_t numBytes     = iter->second;
               
        allocBlocks.erase(iter);

        // insert the block into the free blocks map
        freeBlocks.insert( std::make_pair( numBytes, ptr ) );
    }
};

// the cache is simply a global variable
CachedAllocator thrustAllocator; 

void thrust_free_all()
{
    thrustAllocator.freeAll(); 
}

///////////////////////////////////////////////////////////////////////////////

void thrust_sort_by_key
(
DevVector<int>::iterator keyBeg, 
DevVector<int>::iterator keyEnd, 
thrust::zip_iterator< 
    thrust::tuple< 
        DevVector<int>::iterator,
        DevVector<Point2>::iterator > > valueBeg
)
{
    thrust::sort_by_key( 
        //thrust::cuda::par( thrustAllocator ),
        keyBeg, keyEnd, valueBeg ); 
}

void thrust_transform_GetMortonNumber
(
DevVector<Point2>::iterator inBeg, 
DevVector<Point2>::iterator inEnd, 
DevVector<int>::iterator    outBeg, 
RealType                    minVal, 
RealType                    maxVal
)
{
    thrust::transform( 
        thrust::cuda::par( thrustAllocator ),
        inBeg, inEnd, outBeg, GetMortonNumber( minVal, maxVal ) ); 
}

// Convert count vector with its map
// Also calculate the sum of input vector
// Input:  [ 4 2 0 5 ]
// Output: [ 0 4 6 6 ] Sum: 11
int makeInPlaceMapAndSum
( 
IntDVec& inVec 
)
{
    const int lastValue = inVec[ inVec.size() - 1 ]; 

    thrust::exclusive_scan( 
        thrust::cuda::par( thrustAllocator ),
        inVec.begin(), inVec.end(), inVec.begin() );

    const int sum = inVec[ inVec.size() - 1 ] + lastValue; 

    return sum;
}

// See: makeInPlaceMapAndSum
int makeInPlaceIncMapAndSum
( 
IntDVec& inVec 
)
{
    thrust::inclusive_scan( 
        thrust::cuda::par( thrustAllocator ),
        inVec.begin(), inVec.end(), inVec.begin() );

    const int sum = inVec[ inVec.size() - 1 ];

    return sum;
}

int compactIfNegative
( 
DevVector<int>& inVec,
DevVector<int>& tempVec
)
{
    tempVec.resize( inVec.size() ); 

    tempVec.erase( 
        thrust::copy_if( 
            thrust::cuda::par( thrustAllocator ),
            inVec.begin(), 
            inVec.end(), 
            tempVec.begin(), 
            IsNotNegative() ),
        tempVec.end() );

    inVec.copyFrom( tempVec ); 

    return (int) inVec.size();
}

void compactBothIfNegative
( 
IntDVec& vec0, 
IntDVec& vec1 
)
{
    assert( ( vec0.size() == vec1.size() ) && "Vectors should be equal size!" );

    const IntZipDIter newEnd = 
        thrust::remove_if(  
            //thrust::cuda::par( thrustAllocator ),
            thrust::make_zip_iterator( 
                thrust::make_tuple( vec0.begin(), vec1.begin() ) ),
                thrust::make_zip_iterator( thrust::make_tuple( vec0.end(), vec1.end() ) ),
            IsIntTuple2Negative() );

    const IntDIterTuple2 endTuple = newEnd.get_iterator_tuple();

    vec0.erase( thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( thrust::get<1>( endTuple ), vec1.end() );

    return;
}

void thrust_scan_TriHasVert
(
IntDVec& inVec, 
IntDVec& outVec
)
{
    outVec.resize( inVec.size() ); 

    thrust::transform_exclusive_scan( 
        thrust::cuda::par( thrustAllocator ),
        inVec.begin(), inVec.end(), 
        outVec.begin(),
        MakeKeyFromTriHasVert(), 
        0,
        thrust::plus<int>() );
}

int thrust_copyIf_IsActiveTri
(
const CharDVec& inVec,
IntDVec&        outVec
)
{
    thrust::counting_iterator<int> first( 0 ); 
    thrust::counting_iterator<int> last = first + inVec.size(); 

    outVec.resize( inVec.size() ); 

    outVec.erase( 
        thrust::copy_if( 
            thrust::cuda::par( thrustAllocator ),
            first, last, 
            inVec.begin(), 
            outVec.begin(), 
            IsTriActive() ),
        outVec.end()
        ); 

    return outVec.size(); 
}

int thrust_copyIf_IsNotNegative
(
const IntDVec& inVec,
IntDVec&       outVec
)
{
    thrust::counting_iterator<int> first( 0 ); 
    thrust::counting_iterator<int> last = first + inVec.size(); 

    outVec.resize( inVec.size() ); 

    outVec.erase( 
        thrust::copy_if( 
            thrust::cuda::par( thrustAllocator ),
            first, last, 
            inVec.begin(), 
            outVec.begin(), 
            IsNotNegative() ),
        outVec.end()
        ); 

    return outVec.size(); 
}

int thrust_copyIf_TriHasVert
(
const IntDVec& inVec,
IntDVec&       outVec
)
{
    thrust::counting_iterator<int> first( 0 ); 
    thrust::counting_iterator<int> last = first + inVec.size(); 

    outVec.erase( 
        thrust::copy_if( 
            thrust::cuda::par( thrustAllocator ),
            first, last, 
            inVec.begin(), 
            outVec.begin(), 
            IsTriHasVert() ),
        outVec.end()
        ); 

    return outVec.size(); 
}

void thrust_scatterSequenceMap
(
const IntDVec& inVec,
IntDVec&       outVec
)
{
    thrust::counting_iterator<int> first( 0 ); 
    thrust::counting_iterator<int> last = first + inVec.size(); 

    thrust::scatter( first, last, inVec.begin(), outVec.begin() ); 
}

void thrust_scatterConstantMap
(
const IntDVec&  inVec,
CharDVec&       outVec,
char            value
)
{
    thrust::constant_iterator<int> first( value, 0 ); 
    thrust::constant_iterator<int> last( value, inVec.size() ); 

    thrust::scatter( first, last, inVec.begin(), outVec.begin() ); 
}


void thrust_scan_TriAliveStencil
(
const CharDVec& inVec, 
IntDVec& outVec
)
{
    outVec.resize( inVec.size() ); 

    thrust::transform_inclusive_scan( 
        thrust::cuda::par( thrustAllocator ), 
        inVec.begin(), inVec.end(), 
        outVec.begin(), 
        TriAliveStencil(), 
        thrust::plus<int>() 
        ); 
}

int thrust_sum
(
const IntDVec& inVec
)
{
    return thrust::reduce( inVec.begin(), inVec.end(), 0 ); 
}