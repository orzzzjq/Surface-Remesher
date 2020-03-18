/*
Author: Cao Thanh Tung
Date: 07/02/2012

Desc: Visualization class - Rendering and interactions. 
*/

#include "Visualizer.h"

#include "GL/freeglut.h"
#include "gDel2D/GpuDelaunay.h"

const int PointMask = 0xc0000000; 
const int EdgeMask  = 0x80000000; 
const int TriMask   = 0x40000000; 
const int IdMask    = 0x3fffffff; 

const int ClickRadius = 3; 

namespace { 

    void glutReshape(int width, int height)
    {
        Visualizer *vis = Visualizer::instance(); 

        vis->reshape( width, height ); 
    }

    void glutDisplay()
    {
        Visualizer *vis = Visualizer::instance(); 

        vis->display(); 
    }
    
    void glutMouse( int button, int state, int x, int y )
    {
        Visualizer *vis = Visualizer::instance(); 

        vis->mouse( button, state, x, y ); 
    }

    void glutMouseMotion( int x, int y )
    {
        Visualizer *vis = Visualizer::instance(); 

        vis->mouseMotion( x, y ); 
    }

    void glutKeyboard( unsigned char key, int x, int y )
    {
        Visualizer *vis = Visualizer::instance(); 

        vis->keyboard( key, x, y ); 
    }
}

// Singleton initialization
Visualizer* Visualizer::_singleton = 0; 

Visualizer::Visualizer() : 
    _colorMap( _hashUInt )
{
    // Default window size
    _winSize     = 768; 
    _winWidth    = _winSize; 
    _winHeight   = _winSize; 
    _worldWidth  = 1.0; 
    _worldHeight = 1.0; 

    // Mouse state
    _oldMouseX          = 0; 
    _oldMouseY          = 0; 
    _isLeftMouseActive  = false; 
    _isRightMouseActive = false; 
    _isSelectActive     = false; 
    _isClicked          = false; 
    
    // Clicked
    _clickedId = 0; 

    // State
    _scale      = 1.0; 
    _xTranslate = 0.0; 
    _yTranslate = 0.0; 

    _enable     = true; 
    _paused     = false; 
    _curFrame   = -1; 

    // GPU buffers
    _pointBuf       = -1; 
    _triBuf         = -1; 
    _constraintBuf  = -1; 
    _fbo            = -1; 
    _idBuf          = -1; 

    // CPU buffers
    glTriangles     = NULL; 
    glConstraints   = NULL; 

    // Settings
    _pointOn        = true; 
    _constraintOn   = true; 
    _triangleOn     = true; 
    _clickableOn    = true; 
    _centeredOn     = true; 

    // Random generator
    randGen.init( time( NULL ), 0.0, 1.0 ); 
}

Visualizer* Visualizer::instance()  
{
    if ( _singleton == 0 ) 
        _singleton = new Visualizer(); 

    return _singleton; 
}

void Visualizer::init( int argc, char *argv[], char* appName )
{
    if ( !_enable ) return ; 

    glutInit( &argc, argv );
    glutInitWindowPosition( 0, 0 );
    glutInitWindowSize( _winWidth, _winHeight );
    glutSetOption( GLUT_MULTISAMPLE, 8 ); 
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE );
    glutCreateWindow( appName );
    
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        printf( "GLEW Error: %s\n", glewGetErrorString(err) );
        return ; 
    }

    // Create framebuffer if needed
    if ( _clickableOn ) 
        glGenFramebuffersEXT( 1, &_fbo ); 

    glutDisplayFunc( glutDisplay );
    glutReshapeFunc( glutReshape );
    glutMouseFunc( glutMouse );
    glutMotionFunc( glutMouseMotion );
    glutKeyboardFunc( glutKeyboard ); 
}

void Visualizer::disable()
{
    _enable = false; 
}

bool Visualizer::isEnable()
{
    return _enable; 
}

void Visualizer::pause()
{
    _paused = true; 
}

void Visualizer::resume()
{
    _paused = false; 
}

void Visualizer::run()
{
    if ( !_enable ) return ; 

    _curPointVec.clear(); 
    _curConstraintVec.clear(); 
    _curTriVec.clear(); 

    _curFrame = -1; 

    nextFrame(); 

    zoomToFit(); 

    glutMainLoop(); 
}

void Visualizer::printHelp()
{
    printf( "\n" ); 
    printf( "Visualization mouse controls: \n" ); 
    printf( "    Left            Pan\n" ); 
    printf( "    Right           Scale\n" ); 
    printf( "    Ctrl + Left     Rectangle Zoom\n" ); 
    printf( "    Shift + Left    Select\n" ); 
    printf( "\n" ); 
    printf( "Visualization keyboard shortcuts: \n" ); 
    printf( "    0       Reset view\n" ); 
    printf( "    p       Show/Hide points\n" );       
    printf( "    t       Show/Hide triangles\n" );       
    printf( "\n"); 
}

GLuint Visualizer::getColor( int id ) 
{
    GLuint color = 0; 

    if ( id >= 0 ) 
    {
        if ( !_colorMap.get( id, &color ) ) 
        {
            GLubyte newColor[4] = { 
                int(randGen.getNext() * 255), 
                int(randGen.getNext() * 255), 
                int(randGen.getNext() * 255), 
                64 
            }; 

            color = *((GLuint *) newColor); 

            _colorMap.insert( id, color ); 
        }
    }
        
    return color; 
}

void Visualizer::refresh()
{
    if ( _curFrame == -1 ) return ; 

    // Transfer points data to GPU
    if ( _pointBuf != -1 ) 
        glDeleteBuffers( 1, &_pointBuf ); 

    PointAttribute *glPoints = new PointAttribute[ _pointNum ]; 

    for ( unsigned int i = 0; i < _pointNum; i++ ) 
    {
        glPoints[i].x = (float) _curPointVec[i]._p[0]; 
        glPoints[i].y = (float) _curPointVec[i]._p[1]; 
        glPoints[i].id = i | PointMask; 
    }

    if ( _infIdx >= 0 && _infIdx < _pointNum ) 
        if ( _infIdx == 0 ) 
            glPoints[ _infIdx ] = glPoints[ 1 ]; 
        else
            glPoints[ _infIdx ] = glPoints[ 0 ]; 

    glGenBuffers( 1, &_pointBuf );
    glBindBuffer( GL_ARRAY_BUFFER, _pointBuf );
    glBufferData( GL_ARRAY_BUFFER, _pointNum * sizeof( PointAttribute ), 
        glPoints, GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    delete [] glPoints; 

    // Transfer triangles data to GPU
    if ( _triBuf!= -1 ) 
        glDeleteBuffers( 1, &_triBuf ); 

    if ( glTriangles != NULL ) 
        delete [] glTriangles; 

    glTriangles = new PointAttribute[ _triNum * 3 ]; 

    for ( unsigned int i = 0; i < _triNum; i++ ) 
    {
        Tri tri = _curTriVec[i]; 

        for ( int j = 0; j < 3; j++ ) 
            if ( !tri.has( _infIdx ) ) 
            {
                glTriangles[i * 3 + j].x  = (float) _curPointVec[ _curTriVec[i]._v[j] ]._p[0]; 
                glTriangles[i * 3 + j].y  = (float) _curPointVec[ _curTriVec[i]._v[j] ]._p[1]; 
                glTriangles[i * 3 + j].id = i | TriMask; 

                if ( _curTriColorVec.size() > 0 ) 
                    glTriangles[i * 3 + j].color = getColor( _curTriColorVec[ i ] ); 
                else
                    glTriangles[i * 3 + j].color = 0; 
            }
            else
            {
                glTriangles[i * 3 + j].x     = 0;
                glTriangles[i * 3 + j].y     = 0;
                glTriangles[i * 3 + j].id    = 0; 
                glTriangles[i * 3 + j].color = 0; 
            }
    }

    int size = min( _triNum, MaxTriangles ) * 3 * sizeof(PointAttribute);

    glGenBuffers( 1, &_triBuf );
    glBindBuffer( GL_ARRAY_BUFFER, _triBuf );
    glBufferData( GL_ARRAY_BUFFER, size, glTriangles, GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    // Transfer constraints data to GPU
    if ( _constraintNum == 0 ) return ;

    if ( _constraintBuf!= -1 ) 
        glDeleteBuffers( 1, &_constraintBuf ); 

    if ( glConstraints != NULL ) 
        delete [] glConstraints; 
   
    glConstraints = new PointAttribute[ _constraintNum * 2 ]; 

    GLubyte col[2][4] = { 
        {  81, 31,  82, 255 },
        { 243, 110, 244, 255 } }; 

    for ( unsigned int i = 0; i < _constraintNum; i++ ) 
        for ( int j = 0; j < 2; j++ ) 
        {
            glConstraints[i * 2 + j].x  = (float) _curPointVec[ _curConstraintVec[i]._v[j] ]._p[0]; 
            glConstraints[i * 2 + j].y  = (float) _curPointVec[ _curConstraintVec[i]._v[j] ]._p[1]; 
            glConstraints[i * 2 + j].id = i | EdgeMask; 
            glConstraints[i * 2 + j].color = (( GLuint* ) col)[ j ]; 
        }

    size = min( _constraintNum, MaxSegments ) * 2 * sizeof(PointAttribute);

    glGenBuffers( 1, &_constraintBuf );
    glBindBuffer( GL_ARRAY_BUFFER, _constraintBuf );
    glBufferData( GL_ARRAY_BUFFER, size, glConstraints, GL_STATIC_DRAW );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    //_colorMap.summary(); 
}

void Visualizer::drawConstraints( bool drawId = false )
{
    if ( _constraintNum == 0 ) return ; 

    if ( drawId == false ) 
        glLineWidth( 2.0f ); 
        //glColor3ub( 163, 73, 164 );

    glBindBuffer( GL_ARRAY_BUFFER, _constraintBuf );
    glVertexPointer( 2, GL_FLOAT, sizeof( PointAttribute ), 0 ); 
    glEnableClientState( GL_VERTEX_ARRAY ); 

    if ( drawId ) 
        glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof( PointAttribute ), (void *) 8 ); 
    else 
        glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof( PointAttribute ), (void *) 12 ); 

    glEnableClientState( GL_COLOR_ARRAY ); 

    unsigned int size = min( MaxSegments, _constraintNum ); 
    glDrawArrays( GL_LINES, 0, size * 2 ); 

    glDisableClientState( GL_VERTEX_ARRAY ); 
    glDisableClientState( GL_COLOR_ARRAY ); 
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    // Draw unbuffered triangles
    glBegin( GL_LINES ); 
        for ( unsigned int i = size; i < _constraintNum; i++ )
            for ( int j = 0; j < 2; ++j ) 
            {
                if ( drawId )           
                    glColor4ubv( (GLubyte *) &glConstraints[i * 2 + j].id ); 
                else 
                    glColor4ubv( (GLubyte *) &glConstraints[i * 2 + j].color ); 

                glVertex2f( glConstraints[i * 2 + j].x, glConstraints[i * 2 + j].y ); 
            }
    glEnd(); 

    glLineWidth( 1.0f ); 
}

void Visualizer::drawPoints( bool drawId = false )
{
    if ( drawId == false ) 
    {
        glPointSize( 4.0 );  
        glColor3ub( 237, 28, 36 );
    }

    glBindBuffer( GL_ARRAY_BUFFER, _pointBuf );
    glVertexPointer( 2, GL_FLOAT, sizeof( PointAttribute ), 0 ); 
    glEnableClientState( GL_VERTEX_ARRAY ); 

    if ( drawId ) 
    {
        glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof( PointAttribute ), (void *) 8 ); 
        glEnableClientState( GL_COLOR_ARRAY ); 
    }

    glDrawArrays( GL_POINTS, 0, _pointNum ); 

    glDisableClientState( GL_VERTEX_ARRAY ); 
    glDisableClientState( GL_COLOR_ARRAY ); 
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    glPointSize(1.0);   
}

void Visualizer::drawTriangles( bool drawId = false ) 
{
    glBindBuffer( GL_ARRAY_BUFFER, _triBuf );
    glVertexPointer( 2, GL_FLOAT, sizeof( PointAttribute ), 0 ); 
    glEnableClientState( GL_VERTEX_ARRAY ); 

    if ( !drawId && _curTriColorVec.size() > 0 ) 
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof( PointAttribute ), (void *) 12 ); 
        glEnableClientState( GL_COLOR_ARRAY ); 

        unsigned int size = min( MaxTriangles, _triNum ); 
        glDrawArrays( GL_TRIANGLES, 0, size * 3 ); 

        glDisableClientState( GL_COLOR_ARRAY ); 

        // Draw unbuffered triangles
        glBegin( GL_TRIANGLES ); 
            for ( unsigned int i = size; i < _triNum; i++ )
            {
                glColor4ubv( (GLubyte *) &glTriangles[i * 3 + 0].color ); 
                glVertex2f( glTriangles[i * 3 + 0].x, glTriangles[i * 3 + 0].y ); 
                glVertex2f( glTriangles[i * 3 + 1].x, glTriangles[i * 3 + 1].y ); 
                glVertex2f( glTriangles[i * 3 + 2].x, glTriangles[i * 3 + 2].y ); 
            }
        glEnd(); 
    }

    if ( drawId == false ) 
    {
        //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        //glPolygonMode(GL_BACK, GL_FILL);

        //glColor3ub( 153, 217, 234 );
        //glColor3ub( 72, 118, 121 );
        glColor3ub( 94, 157, 136 );
    }
    else 
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if ( drawId ) 
    {
        glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof( PointAttribute ), (void *) 8 ); 
        glEnableClientState( GL_COLOR_ARRAY ); 
    }

    unsigned int size = min( MaxTriangles, _triNum ); 
    glDrawArrays( GL_TRIANGLES, 0, size * 3 ); 

    glDisableClientState( GL_VERTEX_ARRAY ); 
    glDisableClientState( GL_COLOR_ARRAY ); 
    glBindBuffer( GL_ARRAY_BUFFER, 0 );

    // Draw unbuffered triangles
    glBegin( GL_TRIANGLES ); 
        for ( unsigned int i = size; i < _triNum; i++ )
        {
            if ( drawId )           
                glColor4ubv( (GLubyte *) &glTriangles[i * 3 + 0].id ); 

            glVertex2f( glTriangles[i * 3 + 0].x, glTriangles[i * 3 + 0].y ); 
            glVertex2f( glTriangles[i * 3 + 1].x, glTriangles[i * 3 + 1].y ); 
            glVertex2f( glTriangles[i * 3 + 2].x, glTriangles[i * 3 + 2].y ); 
        }
    glEnd(); 
}

void Visualizer::drawSelection()
{
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluOrtho2D( 0.0, _winWidth, _winHeight, 0.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    int x0 = min( _selectX, _oldMouseX ); 
    int x1 = max( _selectX, _oldMouseX ); 
    int y0 = min( _selectY, _oldMouseY ); 
    int y1 = max( _selectY, _oldMouseY ); 

    if ( x0 >= x1 || y0 >= y1 ) 
        return ; 

    glColor3f( 0.2f, 0.6f, 1.0f );  
    glBegin( GL_LINE_STRIP ); 
        glVertex2i( x0, y0 ); 
        glVertex2i( x0, y1 ); 
        glVertex2i( x1, y1 ); 
        glVertex2i( x1, y0 ); 
        glVertex2i( x0, y0 ); 
    glEnd(); 

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glColor4f( 0.0f, 0.35f, 0.7f, 0.31f );  
    glBegin( GL_QUADS ); 
        glVertex2i( x0, y0 + 1 ); 
        glVertex2i( x0, y1 ); 
        glVertex2i( x1 - 1, y1 ); 
        glVertex2i( x1 - 1, y0 + 1 ); 
    glEnd(); 
}

void Visualizer::drawClicked() 
{
    unsigned int id = _clickedId & IdMask; 

    int tip = -1; 

    if ( _clickedId >= PointMask ) 
    {
        glPointSize( 10 );  
        glColor4ub( 34, 177, 76, 128 ); 

        glBegin( GL_POINTS ); 
            glVertex2f( (float) _curPointVec[id]._p[0], (float) _curPointVec[id]._p[1] ); 
        glEnd(); 

        tip = id; 

        glPointSize( 1.0 ); 
    } 
    else if ( _clickedId >= EdgeMask ) 
    {
        glColor4ub( 0, 162, 232, 128 ); 
        glLineWidth( 8.0 ); 

        glBegin( GL_LINES ); 
            for ( int i = 0; i < 2; i++ ) 
            {
                int idx = _curConstraintVec[id]._v[i]; 
                glVertex2f( (float) _curPointVec[idx]._p[0], (float) _curPointVec[idx]._p[1] ); 
                tip = idx; 
            }
        glEnd(); 

        glLineWidth( 1.0 ); 
    }
    else if ( _clickedId >= TriMask ) 
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glColor4ub( 34, 177, 76, 128 ); 

        glBegin( GL_TRIANGLES ); 
            for ( int i = 0; i < 3; i++ ) 
            {
                int idx = _curTriVec[id]._v[i]; 
                glVertex2f( (float) _curPointVec[idx]._p[0], (float) _curPointVec[idx]._p[1] ); 
                tip = idx; 
            }
        glEnd(); 
    }

    // Show a pointer
    if ( tip != -1 ) 
    {
        glColor3ub( 63, 72, 204 ); 
        glBegin( GL_LINES ); 
            glVertex2f( -100, -100 ); 
            glVertex2f( (float) _curPointVec[tip]._p[0], (float) _curPointVec[tip]._p[1] ); 
        glEnd(); 
    }

    // Show the clicked position
    //float x = _oldMouseX; 
    //float y = _winHeight - 1 - _oldMouseY; 

    //x = (-_xTranslate + x / _winSize) / _scale;
    //y = (-_yTranslate + y / _winSize) / _scale;

    //glPointSize( 4 );  
    //glColor3ub( 3, 7, 46 ); 

    //glBegin( GL_POINTS ); 
    //    glVertex2f( x, y ); 
    //glEnd(); 

    //glPointSize( 1.0 ); 
}

void Visualizer::zoomSelect() 
{
    int x0 = min( _selectX, _oldMouseX ); 
    int x1 = max( _selectX, _oldMouseX ); 
    int y0 = _winHeight - 1 - max( _selectY, _oldMouseY ); 
    int y1 = _winHeight - 1 - min( _selectY, _oldMouseY ); 

    if ( x0 >= x1 || y0 >= y1 ) 
        return ; 

    float wx0 = (-_xTranslate + (float) x0 / _winSize) / _scale;
    float wx1 = (-_xTranslate + (float) x1 / _winSize) / _scale;
    float wy0 = (-_yTranslate + (float) y0 / _winSize) / _scale;
    float wy1 = (-_yTranslate + (float) y1 / _winSize) / _scale;

    // Set new zoom setting
    _scale      = min( _winWidth / (wx1 - wx0), _winHeight / (wy1 - wy0) ) / _winSize; 
    _xTranslate = -wx0 * _scale; 
    _yTranslate = -wy0 * _scale; 
}

void Visualizer::click()
{
    int x = _oldMouseX; 
    int y = _winHeight - 1 - _oldMouseY; 

    int x0 = max( x - ClickRadius, 0 ); 
    int y0 = max( y - ClickRadius, 0 ); 
    int x1 = min( x + ClickRadius, _winWidth - 1 ); 
    int y1 = min( y + ClickRadius, _winHeight - 1 ); 
    int w  = x1 - x0 + 1; 
    int h  = y1 - y0 + 1; 

    unsigned int *buf = new unsigned int[w * h]; 

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, _fbo ); 
    glReadPixels( x0, y0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf ); 
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 ); 

    _clickedId = 0; 

    int px = 0, py = 0; 

    for ( int iy = y0; iy <= y1; ++iy ) 
        for ( int ix = x0; ix <= x1; ++ix ) 
        {
            unsigned int val = buf[ (iy - y0) * w + (ix - x0) ]; 

            if ( (val | IdMask) > (_clickedId | IdMask) ) 
            { _clickedId = val; px = ix; py = iy; }
            else if ( (val | IdMask) == (_clickedId | IdMask) ) 
                if ( (ix-x)*(ix-x) + (iy-y)*(iy-y) < (px-x)*(px-x) + (py-y)*(py-y) ) 
                { _clickedId = val; px = ix; py = iy; }
        }

    if ( _clickedId > 0 ) 
    {
        int id      = _clickedId & IdMask; 
        int type    = _clickedId - id; 

        printf( "Clicked: 0x%08X --> ", _clickedId ); 

        switch ( type ) 
        {
        case PointMask: 
            printf( "Vertex %i (%.10f %.10f)\n", id, 
                _curPointVec[ id ]._p[0], _curPointVec[ id ]._p[1] ); 
            break; 
        case EdgeMask: 
            printf( "Constraint %i (%i %i)\n", id, 
                _curConstraintVec[ id ]._v[0], _curConstraintVec[ id ]._v[1] ); 
            break; 
        case TriMask: 
            printf( "Triangle %i (%i %i %i)\n", id, 
                _curTriVec[ id ]._v[0], _curTriVec[ id ]._v[1], _curTriVec[ id ]._v[2] ); 
            break; 
        }
    }
    else
        printf( "Clicked: nothing!\n"); 

    delete [] buf; 
}

void Visualizer::reshape(int width, int height)
{
    if ( width <= 0 || height <= 0 ) 
        return ; 

    Visualizer *vis = Visualizer::instance(); 

    // Update translation to make sure the center of the viewport is stable
    float centerX = vis->_worldWidth / 2.0f; 
    float centerY = vis->_worldHeight / 2.0f; 

    vis->_winWidth    = width;
    vis->_winHeight   = height;
    vis->_winSize     = min( width, height ); 
    vis->_worldWidth  = (float) width  / vis->_winSize; 
    vis->_worldHeight = (float) height / vis->_winSize; 

    if ( _centeredOn ) 
    {
        vis->_xTranslate += vis->_worldWidth  / 2.0f - centerX; 
        vis->_yTranslate += vis->_worldHeight / 2.0f - centerY;
    }

    // Allocate framebuffer for mouse click processing
    if ( _clickableOn ) 
    {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, vis->_fbo ); 

        if ( vis->_idBuf != -1 ) 
        {
            glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
                GL_RENDERBUFFER_EXT, 0 ); 
            glDeleteBuffers( 1, &vis->_idBuf ); 
        }

        glGenBuffers( 1, &vis->_idBuf ); 
        glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, vis->_idBuf ); 
        glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, GL_RGBA, width, height ); 
        glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, 0 ); 

        glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
            GL_RENDERBUFFER_EXT, vis->_idBuf ); 
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 ); 
    }

    display();
}

void Visualizer::display()
{
    Visualizer *vis = Visualizer::instance(); 

    if ( vis->_pointBuf == -1 ) 
        vis->refresh(); 

    glClearColor( 1.0, 1.0, 1.0, 0.0 );
    glClear( GL_COLOR_BUFFER_BIT );

    // Setup Matrices
    glViewport( 0, 0, vis->_winWidth, vis->_winHeight);
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluOrtho2D( 0.0, vis->_worldWidth, 0.0, vis->_worldHeight );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( vis->_xTranslate, vis->_yTranslate, 0.0 );
    glScalef( vis->_scale, vis->_scale, 1.0 );

    glEnable( GL_MULTISAMPLE );

    glEnable( GL_BLEND ); 
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );   
    //glEnable( GL_LINE_SMOOTH ); 
    //glEnable( GL_POINT_SMOOTH ); 
    //glHint( GL_POINT_SMOOTH_HINT, GL_NICEST ); 
    //glHint( GL_LINE_SMOOTH_HINT, GL_NICEST ); 

    if ( _triangleOn ) 
        vis->drawTriangles();

    if ( _pointOn ) 
        vis->drawPoints();

    if ( _constraintOn ) 
        vis->drawConstraints();

    vis->drawClicked(); 

    // Select mode
    if ( vis->_isSelectActive ) 
        vis->drawSelection(); 

    //glDisable( GL_LINE_SMOOTH ); 
    //glDisable( GL_POINT_SMOOTH ); 
    glDisable( GL_BLEND ); 
    glDisable( GL_MULTISAMPLE );

    if ( _clickableOn ) 
    {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, vis->_fbo ); 

        glClearColor( 0.0, 0.0, 0.0, 0.0 );
        glClear(GL_COLOR_BUFFER_BIT );

        if ( _triangleOn ) 
            vis->drawTriangles( true ); 

        if ( _constraintOn ) 
            vis->drawConstraints( true );

        if ( _pointOn ) 
                vis->drawPoints( true ); 

        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 ); 
    }

	// Display states
	const int MaxLen = 20; 
	char buffer[ MaxLen ]; 
	
	glColor3ub( 0, 39, 74 ); 

	glMatrixMode( GL_PROJECTION ); 
	glLoadIdentity(); 
    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 ); 
	glMatrixMode( GL_MODELVIEW ); 
	glLoadIdentity();

	glRasterPos2d( 0.01, 0.01 ); 
	snprintf( buffer, MaxLen, "%i / %i", _curFrame + 1, _frameList.size() ); 
	glutBitmapString( GLUT_BITMAP_9_BY_15, ( unsigned char * ) buffer ); 

    glutSwapBuffers();
}

void Visualizer::mouse( int button, int state, int x, int y )
{
    Visualizer *vis = Visualizer::instance(); 

    int modifiers = glutGetModifiers(); 
    
    vis->_oldMouseX = x;
    vis->_oldMouseY = y;

    if ( state == GLUT_UP )
    {
        switch (button)
        {
            case GLUT_LEFT_BUTTON:
                if ( vis->_isClicked ) 
                    vis->click();  
                else if ( vis->_isSelectActive ) 
                    vis->zoomSelect();                                         

                vis->_isClicked         = false; 
                vis->_isSelectActive    = false; 
                vis->_isLeftMouseActive = false;
                break;
            case GLUT_RIGHT_BUTTON:
                vis->_isRightMouseActive = false;
                break;
        }
        glutPostRedisplay(); 
    }

    if ( state == GLUT_DOWN )
    {
        switch (button)
        {
        case GLUT_LEFT_BUTTON:
            if (( modifiers & GLUT_ACTIVE_SHIFT ) > 0 ) 
            {
                vis->_isClicked = true; 
            }   
            else if (( modifiers & GLUT_ACTIVE_CTRL ) > 0 )
            {
                vis->_isSelectActive = true; 
                vis->_selectX = x; 
                vis->_selectY = y; 
            }
            else
                vis->_isLeftMouseActive = true;
                        
            break;
        case GLUT_RIGHT_BUTTON:
            vis->_isRightMouseActive = true;
            break;
        }

        glutPostRedisplay(); 
    }
}

void Visualizer::mouseMotion( int x, int y )
{
    Visualizer *vis = Visualizer::instance(); 

    int currentMouseX = x;
    int currentMouseY = y;

    if ( vis->_isLeftMouseActive )
    {
        float transX = (float) (currentMouseX - vis->_oldMouseX) / vis->_winSize; 
        float transY = (float) (vis->_oldMouseY - currentMouseY) / vis->_winSize; 

        vis->_xTranslate += transX; 
        vis->_yTranslate += transY;
        glutPostRedisplay();
    }
    else if ( vis->_isRightMouseActive )
    {
        // Update translation to make sure the center of the viewport is stable
        float centerX = vis->_worldWidth / 2.0f; 
        float centerY = vis->_worldHeight / 2.0f; 
        float wX = ( -vis->_xTranslate + centerX ) / vis->_scale; 
        float wY = ( -vis->_yTranslate + centerY ) / vis->_scale; 

        vis->_scale -= (currentMouseY - vis->_oldMouseY) * vis->_scale / 400.0f;

        if ( _centeredOn ) 
        {
            vis->_xTranslate = centerX - wX * vis->_scale; 
            vis->_yTranslate = centerY - wY * vis->_scale; 
        }

        glutPostRedisplay();
    }
    else if ( vis->_isSelectActive ) 
        glutPostRedisplay(); 

    vis->_oldMouseX = currentMouseX;
    vis->_oldMouseY = currentMouseY;
}

void Visualizer::nextFrame()
{
    if ( _curFrame + 1 >= _frameList.size() ) return ; 

    _curFrame++; 

    _frameList[ _curFrame ].pointStor.     getRight( _curPointVec ); 
    _frameList[ _curFrame ].constraintStor.getRight( _curConstraintVec ); 
    _frameList[ _curFrame ].triStor.       getRight( _curTriVec ); 
    _frameList[ _curFrame ].triColorStor.  getRight( _curTriColorVec ); 

    _infIdx         = _frameList[ _curFrame ].infIdx; 
    _pointNum       = _curPointVec.size(); 
    _constraintNum  = _curConstraintVec.size(); 
    _triNum         = _curTriVec.size(); 
}

void Visualizer::prevFrame()
{
    if ( _curFrame - 1 < 0 ) return ; 

    _frameList[ _curFrame ].pointStor.     getLeft( _curPointVec ); 
    _frameList[ _curFrame ].constraintStor.getLeft( _curConstraintVec ); 
    _frameList[ _curFrame ].triStor.       getLeft( _curTriVec ); 
    _frameList[ _curFrame ].triColorStor.  getLeft( _curTriColorVec ); 

    _curFrame--; 

    _infIdx         = _frameList[ _curFrame ].infIdx; 
    _pointNum       = _curPointVec.size(); 
    _constraintNum  = _curConstraintVec.size(); 
    _triNum         = _curTriVec.size(); 
}

void Visualizer::keyboard( unsigned char key, int x, int y )
{
    Visualizer *vis = Visualizer::instance(); 

    switch ( key ) 
    {
    case 'p': 
        _pointOn = !_pointOn;
        break; 
    case 't': 
        _triangleOn = !_triangleOn;
        break; 
    case 'c': 
        _constraintOn = !_constraintOn;
        break; 
    case '>': 
    case '.':
        nextFrame(); 
        refresh(); 
        break; 
    case '<': 
    case ',':
        prevFrame(); 
        refresh(); 
        break;       
    case '\x1B': // ESC
        vis->_clickedId = 0; 
        break;
    case '0': 
        vis->zoomToFit(); 
        break; 
    case 'q': 
        exit(0); 
    case 'C': 
        int cIdx; 

        std::cin.clear(); 
        std::cout << "Enter constraint ID: "; 
        std::cin >> cIdx; 

        vis->_clickedId = cIdx | EdgeMask; 
        break; 
    case 'T': 
        int tIdx; 

        std::cin.clear(); 
        std::cout << "Enter triangle ID: "; 
        std::cin >> tIdx; 

        vis->_clickedId = tIdx | TriMask; 
        break; 
    case 'P': 
        int pIdx; 

        std::cin.clear(); 
        std::cout << "Enter point ID: "; 
        std::cin >> pIdx; 

        vis->_clickedId = pIdx | PointMask; 
        break; 
    }

    glutPostRedisplay(); 
}

void Visualizer::addFrame
( 
const Point2HVec&   pointVec, 
const SegmentHVec&  constraintVec,
const TriHVec&      triVec,
const IntHVec&      triColorVec,
int                 infIdx
)
{
    if ( !_enable || _paused ) return ; 

    _frameList.push_back( Frame() ); 

    Frame& frame = _frameList.back(); 

    frame.infIdx        = infIdx; 

    frame.pointStor     .create( _curPointVec,      pointVec ); 
    frame.constraintStor.create( _curConstraintVec, constraintVec ); 
    frame.triStor       .create( _curTriVec,        triVec ); 
    frame.triColorStor  .create( _curTriColorVec,   triColorVec ); 

    //int sizeInBytes = frame.pointStor.sizeInBytes()
    //    + frame.constraintStor.sizeInBytes() 
    //    + frame.triStor.sizeInBytes()
    //    + frame.triColorStor.sizeInBytes(); 

    //std::cout << "Frame " << _frameList.size() 
    //    << ": " << (int) (sizeInBytes / 1024.0 / 1024.0 * 100 ) / 100.0 << "MB" << std::endl; 

    _curPointVec        = pointVec; 
    _curConstraintVec   = constraintVec; 
    _curTriVec          = triVec; 
    _curTriColorVec     = triColorVec; 
}

void Visualizer::addFrame
( 
const Point2HVec&   pointVec, 
const SegmentHVec&  constraintVec,
const TriHVec&      triVec,
int                 infIdx
)
{
    addFrame( pointVec, constraintVec, triVec, IntHVec(), infIdx ); 
}

void Visualizer::addFrame
( 
const Point2DVec&   pointVec, 
const TriDVec&      triVec,
int                 infIdx 
)
{
    if ( !_enable || _paused ) return ; 

    Point2HVec  hPointVec; 
    TriHVec     hTriVec; 

    pointVec.copyToHost( hPointVec ); 
    triVec.copyToHost( hTriVec ); 

    addFrame( hPointVec, SegmentHVec(), hTriVec, infIdx ); 
}

void Visualizer::addFrame
( 
const Point2DVec&   pointVec, 
const SegmentDVec&  constraintVec,
const TriDVec&      triVec,
int                 infIdx 
)
{
    if ( !_enable || _paused ) return ; 

    Point2HVec  hPointVec; 
    SegmentHVec hConstraintVec; 
    TriHVec     hTriVec; 

    pointVec.copyToHost( hPointVec ); 
    constraintVec.copyToHost( hConstraintVec ); 
    triVec.copyToHost( hTriVec ); 

    addFrame( hPointVec, hConstraintVec, hTriVec, infIdx ); 
}

void Visualizer::addFrame
( 
const Point2DVec&   pointVec, 
const SegmentDVec&  constraintVec,
const TriDVec&      triVec,
const IntHVec&      triColorVec,
int                 infIdx 
)
{
    if ( !_enable || _paused ) return ; 

    Point2HVec  hPointVec; 
    SegmentHVec hConstraintVec; 
    TriHVec     hTriVec; 

    pointVec.copyToHost( hPointVec ); 
    constraintVec.copyToHost( hConstraintVec ); 
    triVec.copyToHost( hTriVec ); 

    addFrame( hPointVec, hConstraintVec, hTriVec, triColorVec, infIdx ); 
}

void Visualizer::zoomToFit() 
{
    // Find coordinate range
    RealType minX = _curPointVec[0]._p[0]; 
    RealType maxX = _curPointVec[0]._p[0]; 
    RealType minY = _curPointVec[0]._p[1]; 
    RealType maxY = _curPointVec[0]._p[1]; 

    for ( int i = 0; i < _curPointVec.size(); ++i ) 
    {
        if ( _curPointVec[i]._p[0] < minX ) minX = _curPointVec[i]._p[0]; 
        if ( _curPointVec[i]._p[0] > maxX ) maxX = _curPointVec[i]._p[0]; 
        if ( _curPointVec[i]._p[1] < minY ) minY = _curPointVec[i]._p[1]; 
        if ( _curPointVec[i]._p[1] > maxY ) maxY = _curPointVec[i]._p[1]; 
    }

    RealType rangeX = RealType( maxX - minX ); 
    RealType rangeY = RealType( maxY - minY ); 
    RealType scaleX = _worldWidth  / rangeX; 
    RealType scaleY = _worldHeight / rangeY; 

    _scale = min( scaleX, scaleY ) * 0.95; 

    _xTranslate = -(minX + ( rangeX - _worldWidth  / _scale ) / 2.0) * _scale; 
    _yTranslate = -(minY + ( rangeY - _worldHeight / _scale ) / 2.0) * _scale; 
}
