#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>  
#include <time.h> 
#include "image.h"                                          
using namespace std;

//2D kernel convolution with 1D arrays
int* convolution (int* image, int kernel[], int imWidth,
	int imHeight, int kernWidth)
{
	int* result = (int*) malloc(imHeight*imWidth*sizeof(int));

	for (int i = 1; i < imHeight-1;i++)
	{
		for (int j = 1; j < imWidth-1;j++)
		{
			int kernSum = 0.0;
			for (int k = 0; k < kernWidth; k++)
				for (int l=0;l<kernWidth; l++)
				{
					int index = (i+k-1)*imWidth + (j+l-1);
					kernSum += kernel[k*kernWidth+l]*image[index];
				}
			result[i*imWidth+j] = kernSum;
		}
	}
	return result;
}

//Matrix multiplication with 1D arrays
int* matmult(int* matA, int* matB,int rowsa, int colsa,int rowsb,int colsb)
{
	int* result = new int[rowsa*colsb];

	//Initialize resultant matrix values to 0
	for (int i = 0; i < rowsa*colsb; i++)
	{
			result[i] = 0;
	}

	//Perform matrix multiplication
	for (int i = 0; i < rowsa; i++)
	{
		for (int j = 0; j < colsb; j++)
		{
			for (int k = 0; k < rowsb; k++)
			{
				result[colsb*i + j] += matA[colsa*i+k] * matB[colsb*k+j];
			}
		}
	}

	return result;
}

//Compute the sum of squares for the gradients with a sliding window
int* sumofSquares(int* matrix, int rows, int cols)
{
	int* result = (int*) malloc(rows*cols*sizeof(int));

	for (int i = 1; i < rows-1;i++)
	{
		for (int j = 1; j < cols-1;j++)
		{
			int windowSum = 0.0;
			for (int k = 0; k < 3; k++)
				for (int l=0;l<3; l++)
					windowSum += matrix[(i+k-1)*cols + (j+l-1)];

			result[i*cols+j] = windowSum;
		}
	}
	return result;
}

//Calculate HarrisResponse at each pixel using the sum of derivative squares
//k value can be chosen to change relative sensitivity to corners
int* HarrisResponse(int* Sxx, int* Syy, int* Sxy, int matSize,int k)
{
	int* result = (int*) malloc(matSize*sizeof(int));

	for (int i = 0; i < matSize;i++)
	{
		int determinant = (Sxx[i]*Syy[i] - pow(Sxy[i],2));
		int trace = Sxx[i]+Syy[i];
		result[i] = (int)(determinant - k*pow(trace,2));
	}

	return result;
}

int main(int argc, char **argv)
{
	int imWidth = 512;
	int imHeight = 512;
	int* imArray = (int*) calloc(imWidth*imHeight*sizeof(int),1);
	imreadgray("rock512.pgm",imArray);
	clock_t time = clock();

	//Calculate horizontal and vertical derivatives using Sobel filter
	int horizKernel[9] = {-1,0,1,-2,0,2,-1,0,1};
	int vertKernel[9] = {1,2,1,0,0,0,-1,-2,-1};
	int* horizGradient = convolution(imArray,horizKernel,imWidth,imHeight,3);
	int* vertGradient = convolution(imArray,vertKernel,imWidth,imHeight,3);

	//Construct structure tensor using derivatives
	int* Ixx = matmult(horizGradient,horizGradient,imHeight,imWidth,imHeight,imWidth);
	int* Iyy = matmult(vertGradient,vertGradient,imHeight,imWidth,imHeight,imWidth);
	int* Ixy = matmult(horizGradient,vertGradient,imHeight,imWidth,imHeight,imWidth);
	free(horizGradient);
	free(vertGradient);

	//Calculate sum of squares of structure tensor with sliding window
	int* Sxx = sumofSquares(Ixx,imHeight,imWidth);
	int* Syy = sumofSquares(Iyy,imHeight,imWidth);
	int* Sxy = sumofSquares(Ixy,imHeight,imWidth);
	free(Ixx);
	free(Iyy);
	free(Ixy);

	//Calculate Harris response at each pixel
	int* R = HarrisResponse(Sxx,Syy,Sxy,imHeight*imWidth,0.01);
	FILE * f = fopen("ExTimesSerial.txt","a+");
	double extime = (double)(clock()-time)/CLOCKS_PER_SEC;
	fprintf(f,"%f\n",extime);
	fclose(f);

	free(Sxx);
	free(Syy);
	free(Sxy);
	//fprintf(stderr,"Runtime: %lf\n",(double)(clock()-time)/CLOCKS_PER_SEC);
	//Indicate corner in original image
	for (int i =0; i<imHeight;i++){
		for (int j=0; j<imWidth;j++){
			if(R[i*imWidth+j] > 1000){
				//fprintf(stderr,"R value is %u\n",imArray[i*imWidth+j]);
				imArray[i*imWidth+j] = 255;
			}
		}
	}

	free(R);
	imwritegray("rockcorners.pgm",imArray,512,512);
	free(imArray);
	return (0);
}
