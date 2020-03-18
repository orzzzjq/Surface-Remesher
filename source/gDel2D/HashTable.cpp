#include "HashTable.h"

int HashUInt::operator()( unsigned int x ) const
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
}

int HashPoint2::operator()( Point2 p ) const
{
    float x = p._p[0]; 
    float y = p._p[1]; 

    return hashUInt((unsigned int&) x) + hashUInt((unsigned int&) y);
}
