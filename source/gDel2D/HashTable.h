#pragma once

#include<vector>

#include "gDel2D/CommonTypes.h"

template<typename T>
class HashFunctor
{
public: 
    virtual int operator()( T key ) const = 0; 
}; 

// Some assumptions: 
// - Known max size
// - Never insert duplicate item
template<typename KeyType, typename ValueType>
class HashTable
{
public: 
    typedef std::pair<KeyType, ValueType> PairType; 

    HashTable( const HashFunctor<KeyType>& hash ) : _hash( hash ), _size(0) 
    {
        _table.resize( 1 ); 
    }

    HashTable( int capacity, const HashFunctor<KeyType>& hash ) : _hash( hash ), _size(0) 
    {
        if ( capacity >= HashFactor ) 
            _table.resize( capacity / HashFactor ); 
        else
            _table.resize( 1 ); 
    }

    void insert( const KeyType& key, const ValueType& value )
    {
        growTable(); 

        const PairType item( key, value ); 

        int idx = _hash( key ) % _table.size(); 

        _table[ idx ].push_back( item ); 

        ++_size; 
    }

    bool get( const KeyType& key, ValueType* value ) const
    {
        int idx = _hash( key ) % _table.size(); 

        for ( int i = 0; i < _table[idx].size(); ++i ) 
        {
            PairType item = _table[idx][i]; 

            if ( key == item.first )
            {
                if ( value != NULL ) 
                    *value = item.second; 

                return true; 
            }
        }

        return false; 
    }

    void summary() const
    {
        int maxBin      = 0;
        int occupancy   = 0;

        for ( int i = 0; i < _table.size(); ++i ) 
        {
            if ( _table[i].size() > maxBin ) 
                maxBin = _table[i].size(); 

            if ( _table[i].size() > 0 ) 
                occupancy++; 
        }

        std::cout << "HashTable: Size = " << _size << ", MaxBin = " << maxBin 
            << ", Occupancy = " << float(occupancy) / _table.size() << std::endl; 
    }

private: 
    static const int HashFactor  = 5; 
    static const int Ratio       = 2; 

    const HashFunctor<KeyType>&             _hash; 
    std::vector< std::vector< PairType > >  _table; 
    int                                     _size; 


    void growTable()
    {
        if ( _table.size() * HashFactor * Ratio > _size ) return ; 

        // Grow the table 
        std::vector< std::vector< PairType > > oldTable( _table.size() * Ratio ); 

        oldTable.swap( _table ); 

        _size = 0; 

        // Rehash
        for ( int i = 0; i < oldTable.size(); ++i ) 
            for ( int j = 0; j < oldTable[i].size(); ++j ) 
                insert( oldTable[i][j].first, oldTable[i][j].second ); 
    }

}; 

class HashUInt: public HashFunctor<unsigned int>
{
public: 
    int operator()( unsigned int key ) const;
}; 

class HashPoint2: public HashFunctor<Point2>
{
private: 
    HashUInt hashUInt; 

public: 
    int operator()( Point2 p ) const; 
};
