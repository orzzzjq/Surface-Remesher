// Random Number Generator

#pragma once

class RandGen
{
public:
    void init( int, double, double );
    double getNext();
    void nextGaussian( double&, double& );
    unsigned long rand_int();

private:
    unsigned long _z, _w, _jsr, _jcong;
    double _min, _max; 

    unsigned long znew();
    unsigned long wnew();
    unsigned long MWC();
    unsigned long SHR3();
    unsigned long CONG();
    double random();
};

////////////////////////////////////////////////////////////////////////////////