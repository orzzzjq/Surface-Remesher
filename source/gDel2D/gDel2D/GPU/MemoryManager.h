#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

////////////////////////////////////////////////////////////////// DevVector //

template< typename T > 
class DevVector
{
public: 
    // Types
    typedef typename thrust::device_ptr< T > iterator; 

    // Properties
    thrust::device_ptr< T > _ptr;
    size_t                  _size;
    size_t                  _capacity; 
    bool                    _owned; 
    
    DevVector( ) : _size( 0 ), _capacity( 0 ) {}
    
    DevVector( size_t n ) : _size( 0 ), _capacity( 0 )
    {
        resize( n ); 
        return;
    }

    DevVector( size_t n, T value ) : _size( 0 ), _capacity( 0 )
    {
        assign( n, value );
        return;
    }

    ~DevVector()
    {
        free();
        return;
    }

    void free() 
    {
        if ( _capacity > 0 && _owned )
            CudaSafeCall( cudaFree( _ptr.get() ) );

        _size       = 0; 
        _capacity   = 0; 

        return;
    }

    // Use only for cases where new size is within capacity
    // So, old data remains in-place
    void expand( size_t n )
    {
        assert( ( _capacity >= n ) && "New size not within current capacity! Use resize!" );
        _size = n;
    }

    // Resize with data remains
    void grow( size_t n ) 
    {
        assert( ( n >= _size ) && "New size not larger than old size." );

        if ( _capacity >= n )
        {
            _size = n; 
            return;
        }

        DevVector< T > tempVec( n ); 
        thrust::copy( begin(), end(), tempVec.begin() ); 
        swapAndFree( tempVec ); 
    }

    void resize( size_t n )
    {
        if ( _capacity >= n )
        {
            _size = n; 
            return;
        }

        if ( !_owned && _capacity > 0 ) 
        {
            std::cerr << "WARNING: Resizing a DevVector with borrowing pointer!" << std::endl; 
        }

        free(); 

        _size       = n; 
        _capacity   = ( n == 0 ) ? 1 : n; 
        _owned      = true; 

        try
        {
            _ptr = thrust::device_malloc< T >( _capacity );
        }
        catch( ... )
        {
            // output an error message and exit
            const int OneMB = ( 1 << 20 );
            std::cerr << "thrust::device_malloc failed to allocate " << ( sizeof( T ) * _capacity ) / OneMB << " MB!" << std::endl;
            std::cerr << "size = " << _size << " sizeof(T) = " << sizeof( T ) << std::endl; 
            exit( -1 );
        }

        return;
    }

    void assign( size_t n, const T& value )
    {
        resize( n ); 
        thrust::fill_n( begin(), n, value );
        return;
    }

    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }

    thrust::device_reference< T > operator[] ( const size_t index ) const
    {
        return _ptr[ index ]; 
    }

    const iterator begin() const { return _ptr; }

    const iterator end() const { return _ptr + _size; }

    void erase( const iterator& first, const iterator& last )
    {
        if ( last == end() )
        {
            _size -= (last - first);
        }
        else
        {
            assert( false && "Not supported right now!" );
        }

        return;
    }

    void swap( DevVector< T >& arr ) 
    {
        size_t tempSize = _size; 
        size_t tempCap  = _capacity; 
        bool tempOwned  = _owned; 
        T* tempPtr      = ( _capacity > 0 ) ? _ptr.get() : 0; 

        _size       = arr._size; 
        _capacity   = arr._capacity; 
        _owned      = arr._owned; 

        if ( _capacity > 0 )
        {
            _ptr = thrust::device_ptr< T >( arr._ptr.get() ); 
        }

        arr._size       = tempSize; 
        arr._capacity   = tempCap; 
        arr._owned      = tempOwned; 

        if ( tempCap > 0 )
        {
            arr._ptr = thrust::device_ptr< T >( tempPtr );
        }

        return;
    }
    
    // Input array is freed
    void swapAndFree( DevVector< T >& inArr )
    {
        swap( inArr );
        inArr.free();
        return;
    }

    void copyFrom( const DevVector< T >& inArr )
    {
        resize( inArr.size() );
        thrust::copy( inArr.begin(), inArr.end(), begin() );
        return;
    }

    void fill( const T& value )
    {
        thrust::fill_n( _ptr, _size, value );
        return;
    }

    void copyToHost( thrust::host_vector< T >& dest ) const
    {
        dest.insert( dest.begin(), begin(), end() );
        return;
    }

    // Do NOT remove! Useful for debugging.
    void copyFromHost( const thrust::host_vector< T >& inArr )
    {
        resize( inArr.size() );
        thrust::copy( inArr.begin(), inArr.end(), begin() );
        return;
    }
};

//////////////////////////////////////////////////////////// Memory pool //

struct Buffer
{
    void*   ptr; 
    size_t  sizeInBytes; 
    bool    avail; 
}; 

class MemoryPool
{
private:
    std::vector< Buffer > _memPool;      // Two items

public: 
    MemoryPool() {} 

    ~MemoryPool()
    {
        free(); 
    }

    void free( bool report = false ) 
    {
        for ( int i = 0; i < _memPool.size(); ++i ) 
        {
            if ( report ) 
                std::cout << "MemoryPool: [" << i << "]" << _memPool[i].sizeInBytes << std::endl; 

            if ( false == _memPool[ i ].avail ) 
                std::cerr << "WARNING: MemoryPool item not released!" << std::endl; 
            else 
                CudaSafeCall( cudaFree( _memPool[i].ptr ) );  
        }

        _memPool.clear(); 
    }

    template<typename T>
    int reserve( size_t size ) 
    {
        DevVector<T> vec( size ); 

        vec._owned = false;        

        Buffer buf = {
            ( void* ) vec._ptr.get(),
            size * sizeof(T), 
            true 
        }; 

        _memPool.push_back( buf ); 

        return _memPool.size() - 1; 
    }

    template<typename T> 
    DevVector<T> allocateAny( size_t size, bool tempOnly = false ) 
    {
        // Find best fit block
        size_t sizeInBytes  = size * sizeof(T); 
        int bufIdx          = -1; 

        for ( int i = 0; i < _memPool.size(); ++i ) 
            if ( _memPool[i].avail && _memPool[i].sizeInBytes >= sizeInBytes ) 
                if ( bufIdx == -1 || _memPool[i].sizeInBytes < _memPool[bufIdx].sizeInBytes ) 
                    bufIdx = i;       

        if ( bufIdx == -1 ) 
        {
            std::cout << "MemoryPool: Allocating " << sizeInBytes << std::endl; 

            bufIdx = reserve<T>( size ); 
        }

        DevVector<T> vec; 

        vec._ptr        = thrust::device_ptr<T>( (T*) _memPool[ bufIdx ].ptr );
        vec._capacity   = _memPool[ bufIdx ].sizeInBytes / sizeof(T); 
        vec._size       = 0; 
        vec._owned      = false; 

        //std::cout << "MemoryPool: Requesting " 
        //    << sizeInBytes << ", giving " << _memPool[ bufIdx ].sizeInBytes << std::endl; 

        // Disable the buffer in the pool
        if ( !tempOnly ) 
            _memPool[ bufIdx ].avail = false; 

        return vec; 
    }

    template<typename T> 
    void release( DevVector<T>& vec ) 
    {
        for ( int i = 0; i < _memPool.size(); ++i ) 
            if ( _memPool[i].ptr == (void*) vec._ptr.get() ) 
            {
                assert( !_memPool[i].avail ); 
                assert( !vec._owned ); 

                // Return the buffer to the pool
                _memPool[i].avail = true; 

                //std::cout << "MemoryPool: Returning " << _memPool[i].sizeInBytes << std::endl; 

                // Reset the vector to 0 size
                vec.free(); 

                return ; 
            }

        std::cerr << "WARNING: Releasing a DevVector not in the MemoryPool!" << std::endl; 

        // Release the vector
        vec._owned = true;  // Set this to true so it releases itself.
        vec.free(); 

        return ; 
    }
};
