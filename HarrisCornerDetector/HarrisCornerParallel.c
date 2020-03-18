#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>   
#include "image.h"
#include <mpi.h>  
#include <omp.h>                                           
using namespace std;
#define ROOT_NODE	0

//2D kernel convolution with 1D arrays
int* convolution (int* image, int kernel[], int imWidth,
	int imHeight, int kernWidth)
{
	int* result = (int*) malloc(imHeight*imWidth*sizeof(int));
	#pragma omp parallel for
	for (int i = 1; i < imHeight-1;i++){
		for (int j = 1; j < imWidth-1;j++){
			int kernSum = 0.0;
			for (int k = 0; k < kernWidth; k++)
				for (int l=0;l<kernWidth; l++){
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
	for (int i = 0; i < rowsa*colsb; i++){
			result[i] = 0;
	}

	//Perform matrix multiplication
	#pragma omp parallel for
	for (int i = 0; i < rowsa; i++){
		for (int j = 0; j < colsb; j++){
			for (int k = 0; k < rowsb; k++){
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

	#pragma omp parallel for
	for (int i = 1; i < rows-1;i++){
		for (int j = 1; j < cols-1;j++){
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

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < matSize;i++){
		int determinant = (Sxx[i]*Syy[i] - pow(Sxy[i],2));
		int trace = Sxx[i]+Syy[i];
		result[i] = (int)(determinant - k*pow(trace,2));
	}

	return result;
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	//Rank of this node and number of nodes
	int curr_rank = 0, num_nodes=0;
	//Get rank of node/size of communicator
	MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &num_nodes); 
	int horizKernel[9] = {-1,0,1,-2,0,2,-1,0,1};
	int vertKernel[9] = {1,2,1,0,0,0,-1,-2,-1};

	//Arbitrate sizes of matrix to be sent to each node
	int imWidth = 512;
	int imHeight = 512;
	int imageSize = imWidth*imHeight;
	int seg_size = imHeight/num_nodes;
	int leftovers = imHeight % num_nodes;
	int* sendcounts = (int*) malloc(num_nodes*sizeof(int));
	int* displs = (int*) malloc(num_nodes*sizeof(int));
	int sum = 0;
	float time;
	for (int i=0; i < num_nodes; i++){
		if (leftovers>0){
			sendcounts[i] = (seg_size+1)*imWidth;
			leftovers--;
		}
		else
			sendcounts[i] = (seg_size)*imWidth;

		displs[i] = sum;
		sum += sendcounts[i];
		//fprintf(stderr,"For node %d, sendcounts = %d and displs = %d\n",i,sendcounts[i],displs[i]);
	}

	int *imArray,*horizontalGradient,*verticalGradient,*Ixx,*Iyy,*Ixy,*Sxx,*Sxy,*Syy,*R;
	if (curr_rank == ROOT_NODE){
		//Convert image to 1D array
		imArray = (int*) calloc(imageSize*sizeof(int),1);
		imreadgray("rock512.pgm",imArray);
		horizontalGradient = (int*) malloc(imageSize*sizeof(int));
		verticalGradient = (int*) malloc(imageSize*sizeof(int));
		Ixx = (int*) malloc(imageSize*sizeof(int));
		Ixy = (int*) malloc(imageSize*sizeof(int));
		Iyy = (int*) malloc(imageSize*sizeof(int));
		Sxx = (int*) malloc(imageSize*sizeof(int));
		Sxy = (int*) malloc(imageSize*sizeof(int));
		Syy = (int*) malloc(imageSize*sizeof(int));
		time = MPI_Wtime();
	}
	else{
		imArray = (int*) malloc(sendcounts[curr_rank]*sizeof(int));
		horizontalGradient = (int*) malloc(imageSize*sizeof(int));
		verticalGradient = (int*) malloc(imageSize*sizeof(int));
	}
	MPI_Scatterv(imArray,sendcounts,displs,MPI_INT,imArray,imageSize,MPI_INT,ROOT_NODE,MPI_COMM_WORLD);
	int* horizGradSeg = convolution(imArray,horizKernel,imWidth,sendcounts[curr_rank]/imWidth,3);
	int* vertGradSeg = convolution(imArray,vertKernel,imWidth,sendcounts[curr_rank]/imWidth,3);
	MPI_Gatherv(horizGradSeg, sendcounts[curr_rank], MPI_INT, horizontalGradient, sendcounts,displs, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);
	MPI_Gatherv(vertGradSeg, sendcounts[curr_rank], MPI_INT, verticalGradient, sendcounts,displs, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);
	MPI_Bcast(horizontalGradient,imageSize,MPI_INT,ROOT_NODE,MPI_COMM_WORLD);
	MPI_Bcast(verticalGradient,imageSize,MPI_INT,ROOT_NODE,MPI_COMM_WORLD);
	int* Ixxseg = matmult(horizGradSeg,horizontalGradient,sendcounts[curr_rank]/imWidth,imWidth,imHeight,imWidth);
	int* Iyyseg = matmult(vertGradSeg,verticalGradient,sendcounts[curr_rank]/imWidth,imWidth,imHeight,imWidth);
	int* Ixyseg = matmult(horizGradSeg,verticalGradient,sendcounts[curr_rank]/imWidth,imWidth,imHeight,imWidth);
	MPI_Gatherv(Ixxseg, sendcounts[curr_rank], MPI_INT, Ixx, sendcounts,displs, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);
	MPI_Gatherv(Iyyseg, sendcounts[curr_rank], MPI_INT, Iyy, sendcounts,displs, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);
	MPI_Gatherv(Ixyseg, sendcounts[curr_rank], MPI_INT, Ixy, sendcounts,displs, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);
	
	if(curr_rank == ROOT_NODE){
		Sxx = sumofSquares(Ixx,imHeight,imWidth);
		Syy = sumofSquares(Iyy,imHeight,imWidth);
		Sxy = sumofSquares(Ixy,imHeight,imWidth);
		int* R = HarrisResponse(Sxx,Syy,Sxy,imHeight*imWidth,0.01);
		double extime = MPI_Wtime()-time;
		//fprintf(stderr,"Runtime: %f\n",MPI_Wtime()-time);
		//Indicate corner in original image
		#pragma omp parallel for
		for (int i =0; i<imHeight;i++){
			for (int j=0; j<imWidth;j++){
				//fprintf(stderr,"R value is %f\n", R[i*imWidth+j]);
				if(R[i*imWidth+j] > -100){
					imArray[i*imWidth+j] = 255;
					//circle(image,Point(j,i),5,Scalar(0),2,8,0);
				}
			}
		}
		imwritegray("rockcorners.pgm",imArray,512,512);

		FILE * f = fopen("ExTimesParallel.txt","a+");
		fprintf(f,"%f\n",extime);
		fclose(f);

		free(Ixx);
		free(Ixy);
		free(Iyy);
		free(Sxx);
		free(Syy);
		free(Sxy);
		free(R);
	}

	free(Ixxseg);
	free(Ixyseg);
	free(Iyyseg);
	free(horizGradSeg);
	free(vertGradSeg);
	free(horizontalGradient);
	free(verticalGradient);
	free(imArray);
	MPI_Finalize();
	return (0);
}
