#pragma once


#include <stdio.h>
#include <tchar.h>

//#include <emmintrin.h> // SSE2:<e*.h>, SSE3:<p*.h>, SSE4:<s*.h>
//#include <xmmintrin.h>

//SSE 主要用于图像处理的运算加速

#define PI 3.1415926535897931f



// compute x and y gradients for just one column (uses sse)
void grad1( float *I, float *Gx, float *Gy, int h, int w, int x ) ;

// compute x and y gradients at each location (uses sse)
void grad2( float *I, float *Gx, float *Gy, int h, int w, int d ) ;

// build lookup table a[] s.t. a[dx/2.02*n]~=acos(dx)
float* acosTable() ;

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag( float *I, float *M, float *O, int h, int w, int d ) ;

// normalize gradient magnitude at each location (uses sse)
void gradMagNorm( float *M, float *S, int h, int w, float norm );

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
				  int nOrients, int nb, int n, float norm );

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist( float *M, float *O, float *H, int h, int w,
			  int bin, int nOrients, bool softBin );

// compute HOG features given gradient histograms
void hog( float *H, float *G, int h, int w, int bin, int nOrients, float clip );

// TODO: 在此处引用程序需要的其他头文件

// convolve two columns of I by ones filter
void convBoxY( float *I, float *O, int h, int r, int s ) ;

// convolve I by a 2r+1 x 2r+1 ones filter (uses SSE)
void convBox( float *I, float *O, int h, int w, int d, int r, int s ) ;

// convolve single column of I by [1; 1] filter (uses SSE)
void conv11Y( float *I, float *O, int h, int side, int s ) ;

// convolve I by [1 1; 1 1] filter (uses SSE)
void conv11( float *I, float *O, int h, int w, int d, int side, int s ) ;

// convolve one column of I by a 2rx1 triangle filter
void convTriY( float *I, float *O, int h, int r, int s ) ;

// convolve I by a 2rx1 triangle filter (uses SSE)
void convTri( float *I, float *O, int h, int w, int d, int r, int s ) ;

// convolve one column of I by [1 p 1] filter (uses SSE)
void convTri1Y( float *I, float *O, int h, float p, int s ) ;

// convolve I by [1 p 1] filter (uses SSE)
void convTri1( float *I, float *O, int h, int w, int d, float p, int s ) ;

// convolve one column of I by a 2rx1 max filter
void convMaxY( float *I, float *O, float *T, int h, int r ) ;

// convolve every column of I by a 2rx1 max filter
void convMax( float *I, float *O, int h, int w, int d, int r ) ;

// Constants for rgb2luv conversion and lookup table for y-> l conversion
template<class oT> oT* rgb2luv_setup( oT z, oT *mr, oT *mg, oT *mb,
									 oT &minu, oT &minv, oT &un, oT &vn );

// Convert from rgb to luv
template<class iT, class oT> void rgb2luv( iT *I, oT *J, int n, oT nrm ) ;

// Convert from rgb to luv using sse
template<class iT> void rgb2luv_sse( iT *I, float *J, int n, float nrm ) ;

// Convert from rgb to hsv
template<class iT, class oT> void rgb2hsv( iT *I, oT *J, int n, oT nrm ) ;

// Convert from rgb to gray
template<class iT, class oT> void rgb2gray( iT *I, oT *J, int n, oT nrm ) ;

// Convert from rgb (double) to gray (float)
template<> void rgb2gray( double *I, float *J, int n, float nrm ) ;

// Copy and normalize only
template<class iT, class oT> void normalize( iT *I, oT *J, int n, oT nrm ) ;

// Convert rgb to various colorspaces
float* rgbConvert( unsigned char *I, int n, int d, int flag, float nrm );

// compute interpolation values for single column for resapling
template<class T> void resampleCoef( int ha, int hb, int &n, int *&yas,
									int *&ybs, T *&wts, int bd[2], int pad=0 );

// resample A using bilinear interpolation and and store result in B
template<class T>
void resample( T *A, T *B, int ha, int hb, int wa, int wb, int d, T r );

void float_resample( float *A, float *B, int ha, int hb, int wa, int wb, int d, float r );
/*void Uchar_resample(unsigned char *A, unsigned char *B, int ha, int hb, int wa, int wb, int d, float r);*/