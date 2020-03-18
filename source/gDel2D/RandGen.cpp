// Random Number Generator

#include "RandGen.h"
#include <climits>
#include <cmath>

void RandGen::init( int seed, double minVal, double maxVal )
{
    _min    = minVal;
    _max    = maxVal;

    // Seeds
    _z      = seed;
    _w      = seed;
    _jsr    = seed;
    _jcong  = seed;

    return;
}

unsigned long RandGen::znew() 
{ return (_z = 36969 * (_z & 0xfffful) + (_z >> 16)); };

unsigned long RandGen::wnew() 
{ return (_w = 18000 * (_w & 0xfffful) + (_w >> 16)); };

unsigned long RandGen::MWC()  
{ return ((znew() << 16) + wnew()); };

unsigned long RandGen::SHR3()
{ _jsr ^= (_jsr << 17); _jsr ^= (_jsr >> 13); return (_jsr ^= (_jsr << 5)); };

unsigned long RandGen::CONG() 
{ return (_jcong = 69069 * _jcong + 1234567); };

unsigned long RandGen::rand_int()         // [0,2^32-1]
{ return ((MWC() ^ CONG()) + SHR3()); };

double RandGen::random()     // [0,1)
{ return ((double) rand_int() / (double(ULONG_MAX)+1)); };

double RandGen::getNext()
{
    const double val = _min + ( _max - _min) * random(); 
    return val; 
}

void RandGen::nextGaussian( double &x, double &y )
{
    double x1, x2, w;
    double tx, ty; 

    do {
        do {
            x1 = 2.0 * random() - 1.0;
            x2 = 2.0 * random() - 1.0;
            w = x1 * x1 + x2 * x2;
        } while ( w >= 1.0 );

        w = sqrt( (-2.0 * log( w ) ) / w );
        tx = x1 * w;
        ty = x2 * w;
    } while ( tx < -3 || tx >= 3 || ty < -3 || ty >= 3 );

    x = _min + (_max - _min) * ( (tx + 3.0) / 6.0 );
    y = _min + (_max - _min) * ( (ty + 3.0) / 6.0 );

    return;
}