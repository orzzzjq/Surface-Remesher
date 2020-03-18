#pragma once

#include "../CommonTypes.h"

// Preallocate a collection of small integer counters, inilized to 0. 
class SmallCounters
{
private: 
    IntDVec _data; 
    int     _offset; 
    int     _size; 

public: 
    ~SmallCounters();
    
    void init( int size = 1, int capacity = 8192 ); 
    void free(); 
    void renew(); 
    int* ptr(); 
    int operator[]( int idx ) const; 
}; 