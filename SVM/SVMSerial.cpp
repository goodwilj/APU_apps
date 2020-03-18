#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <time.h>                       
using namespace std;

#define numsupvecs 1000
#define supveclen 784
#define numimages 1000
#define imglen 784
#define numcoeffs 1000

long int* matmult(long int* matA, long int* matB,int rowsa, int colsa,int rowsb,int colsb)
{
	long int* result = new long int[rowsa*colsb];
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

long int* readData(const char* filename, int size)
{
	long int* result = new long int[size];
	FILE* file = fopen(filename,"rb");
	fread(result,sizeof(char),size,file);
	fclose(file);
	return result;
}

int main(int argc, char **argv)
{
	//Initialize Support Vector Matrix
	long int* Z = readData("Zmatrix.bin",numsupvecs*supveclen);

	//Initialize MNIST image matrix
	long int* images = readData("Xmatrix.bin",numimages*imglen);

	//Initialize dual coeff matrix
	long int* dualcoeffs = readData("DualCoeffs.bin",numcoeffs);

	//Hold actual test targets
	long int* testtargets = readData("TestTargets.bin",numimages);

	clock_t time = clock();

	//Extract linear kernel
	long int* klin = matmult(images,Z,numimages,imglen,supveclen,numsupvecs);
	
	//Form decision function
	long int* predictions = new long int[numimages];
	for (int i = 0; i < numimages; i++)
	{
		predictions[i] = 0;
	}
	for (int i = 0; i < numimages; i++)
	{
		for (int j = 0; j < numsupvecs; j++)
		{
			predictions[i] += (dualcoeffs[j]-1) * klin[numsupvecs*i+j];
		}
	}
	FILE * f = fopen("ExTimesSerial.txt","a+");
    double extime = (double)(clock()-time)/CLOCKS_PER_SEC;
    fprintf(f,"%f\n",extime);
    fclose(f);
 
	delete[] Z;
	delete[] images;
	delete[] dualcoeffs;

	//Make Prediction and check accuracy
	int num_correct = 0;
	for (int i = 0; i < numimages; i++)
	{
		if (predictions[i] > 0)
			predictions[i] = 1;
		else
			predictions[i] = 0;

		if (predictions[i] == testtargets[i])
			num_correct++;
	}
	float accuracy = ((float)num_correct/numimages)*100;
	//cout << "Prediction Accuracy: "<< accuracy << "%" << endl;
	//fprintf(stderr,"Runtime: %lf\n",(double)(clock()-time)/CLOCKS_PER_SEC);

	delete[] klin;
	delete[] testtargets;
	delete[] predictions;
    return 0;
}

