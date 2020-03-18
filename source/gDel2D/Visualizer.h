#ifndef VISUALIZER
#define VISUALIZER

#include "GL/glew.h"

#include "gDel2D/GpuDelaunay.h"
#include "RandGen.h"
#include "HashTable.h"

#define min(a,b)  ((a) < (b) ? (a) : (b))
#define max(a,b)  ((a) > (b) ? (a) : (b))

typedef HashTable< unsigned int, GLuint > ColorTable; 

template<typename T>
struct DiffItem
{
    int         idx; 
    T           left; 
    T           right; 
}; 

template<typename T> 
struct DiffStorage
{
    int sizeLeft; 
    int sizeRight; 

    std::vector< DiffItem<T> > diffVec; 

    //////////////////////////

    int sizeInBytes()
    {
        return 8 + diffVec.capacity() * sizeof( DiffItem<T> ); 
    }

    void create
    (
    const thrust::host_vector< T >  &prevVec, 
    const thrust::host_vector< T >  &nextVec
    )
    {
        sizeLeft   = prevVec.size(); 
        sizeRight  = nextVec.size(); 

        for ( int i = 0; i < max( sizeLeft, sizeRight ); ++i ) 
        {
            if ( i >= sizeLeft ) 
            {
                DiffItem<T> item = { i, nextVec[i], nextVec[i] }; 
                diffVec.push_back( item ); 
            }
            else if ( i >= sizeRight ) 
            {
                DiffItem<T> item = { i, prevVec[i], prevVec[i] }; 
                diffVec.push_back( item ); 
            }
            else if ( prevVec[i] != nextVec[i] ) 
            {
                DiffItem<T> item = { i, prevVec[i], nextVec[i] }; 
                diffVec.push_back( item ); 
            }
        }

        //std::cout << "Left: " << sizeLeft 
        //    << " Right: " << sizeRight
        //    << " Diff: " << diffVec.size() << std::endl; 
    }

    void getRight( thrust::host_vector<T>& vec )
    {
        vec.resize( sizeRight ); 

        for ( int i = 0; i < diffVec.size(); ++i ) 
        {
            DiffItem<T> item = diffVec[i]; 

            if ( item.idx < sizeRight )
                vec[ item.idx ] = item.right; 
        }
    }

    void getLeft( thrust::host_vector<T>& vec )
    {
        vec.resize( sizeLeft ); 

        for ( int i = 0; i < diffVec.size(); ++i ) 
        {
            DiffItem<T> item = diffVec[i]; 

            if ( item.idx < sizeLeft ) 
                vec[ item.idx ] = item.left; 
        }
    }
}; 

struct Frame 
{
    int infIdx; 

    DiffStorage< Point2 >  pointStor; 
    DiffStorage< Segment > constraintStor; 
    DiffStorage< Tri >     triStor; 
    DiffStorage< int >     triColorStor; 
};

typedef struct { 
    float x, y; 
    unsigned int id; 
    unsigned int color; 
} PointAttribute; 

class Visualizer
{
private: 
    static Visualizer *_singleton; 

    static const int MaxTriangles = 10000000;
    static const int MaxSegments  = 5000000;

    bool    _enable; 
    bool    _paused; 

    int     _winSize; 
    int     _winWidth; 
    int     _winHeight; 
    float   _worldWidth;
    float   _worldHeight; 

    // Mouse Control
    int     _oldMouseX; 
    int     _oldMouseY;
    int     _selectX;
    int     _selectY; 

    bool    _isLeftMouseActive;
    bool    _isRightMouseActive;
    bool    _isSelectActive; 
    bool    _isClicked; 

    // Transformation
    float   _scale;
    float   _xTranslate; 
    float   _yTranslate;

    // Clicked
    unsigned int _clickedId; 

	// Data
    // Frames can be stored using diff from previous frame
    // Landmark should be manually specified to speedup  
	std::vector< Frame > _frameList; 

    // GPU buffers
    GLuint _idBuf; 
    GLuint _fbo; 
    GLuint _pointBuf; 
    GLuint _triBuf; 
    GLuint _constraintBuf; 

    // CPU buffers
    PointAttribute *glTriangles;
    PointAttribute *glConstraints;

    // Current state
	int  _curFrame; 
    int  _pointNum; 
    int  _triNum; 
    int  _constraintNum; 

    HashUInt   _hashUInt; 
    ColorTable _colorMap; 

    // Current frame
    Point2HVec  _curPointVec; 
    SegmentHVec _curConstraintVec; 
    TriHVec     _curTriVec; 
    IntHVec     _curTriColorVec; 
    int         _infIdx; 

    RandGen randGen; 

    // Settings
    bool _pointOn; 
    bool _constraintOn; 
    bool _triangleOn; 
    bool _centeredOn; 
    bool _clickableOn; 

    // Methods
    Visualizer(); 
	~Visualizer(); 

    void zoomSelect(); 
    void click(); 
    void zoomToFit(); 

    void drawPoints( bool drawId ); 
    void drawConstraints( bool drawId );
    void drawTriangles( bool drawId ); 
    void drawSelection(); 
    void drawClicked(); 

    template<typename T>
    void computeDiff(
        const thrust::host_vector< T >  &prevVec, 
        const thrust::host_vector< T >  &nextVec, 
        DiffStorage<T>                  &diffStor    
        ); 

    void nextFrame(); 
    void prevFrame(); 

    GLuint getColor( int id ); 

public: 
    static Visualizer* instance(); 

    void init( int argc, char *argv[], char* appName ); 
    void run();
    void printHelp(); 
    void refresh(); 
    void disable(); 
    bool isEnable(); 
    void pause(); 
    void resume(); 

    int getWidth()  { return _winWidth; } 
    int getHeight() { return _winHeight; }

	void addFrame( 
        const Point2HVec&   pointVec, 
        const SegmentHVec&  constraintVec, 
        const TriHVec&      triVec,
        const IntHVec&      triColorVec,
        int infIdx = -1
        );
	void addFrame( 
        const Point2HVec&   pointVec, 
        const SegmentHVec&  constraintVec, 
        const TriHVec&      triVec,
        int infIdx = -1
        );
	void addFrame( 
        const Point2DVec&   pointVec, 
        const TriDVec&      triVec,
        int infIdx = -1
        );
	void addFrame( 
        const Point2DVec&   pointVec, 
        const SegmentDVec&  constraintVec, 
        const TriDVec&      triVec,
        int infIdx = -1
        );
	void addFrame( 
        const Point2DVec&   pointVec, 
        const SegmentDVec&  constraintVec, 
        const TriDVec&      triVec,
        const IntHVec&      triColorVec,
        int infIdx = -1
        );

    // GLUT 
    void reshape(int width, int height);
    void display(); 
    void mouse( int button, int state, int x, int y ); 
    void mouseMotion( int x, int y );
    void keyboard( unsigned char key, int x, int y );
};

#endif