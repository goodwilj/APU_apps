#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <mpi.h>  
#include <omp.h>                      
using namespace std;

#define ROOT_NODE	0
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
	#pragma omp parallel for
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
	MPI_Init(&argc, &argv);
	//Rank of this node and number of nodes
	int curr_rank = 0, num_nodes=0;
	MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &num_nodes); 

	long int* klinTotal;
	long int* Z;
	long int* images;
	long int* dualcoeffs;
	long int* testtargets;
	int seg_images = numimages/num_nodes;
	int leftovers = (numimages % seg_images);
	int* sendcounts = (int*) malloc(num_nodes*sizeof(int));
	int* displs = (int*) malloc(num_nodes*sizeof(int));
	int sum = 0;
	float time;
	for (int i=0; i < num_nodes; i++){
		if (leftovers>0){
			sendcounts[i] = (seg_images+1)*imglen;
			leftovers--;
		}
		else
			sendcounts[i] = (seg_images)*imglen;

		displs[i] = sum;
		sum += sendcounts[i];
	}
	if(curr_rank == ROOT_NODE){
		//Initialize Support Vector Matrix
		Z = readData("Zmatrix.bin",numsupvecs*supveclen);
		//Initialize MNIST image matrix
		images = readData("Xmatrix.bin",numimages*imglen);
		//Initialize dual coeff matrix
		dualcoeffs = readData("DualCoeffs.bin",numcoeffs);
		//Hold actual test targets
		testtargets = readData("TestTargets.bin",numimages);
		time = MPI_Wtime();
	}
	else{
		Z = (long int*) malloc(numsupvecs*supveclen*sizeof(long int));
		images = (long int*) malloc(sendcounts[curr_rank]*imglen*sizeof(long int));
	}
	MPI_Scatterv(images,sendcounts,displs,MPI_LONG,images,numimages*imglen,MPI_LONG,ROOT_NODE,MPI_COMM_WORLD);
	MPI_Bcast(Z, numsupvecs*supveclen, MPI_LONG, ROOT_NODE, MPI_COMM_WORLD);
	if (curr_rank == ROOT_NODE)
		klinTotal = (long int*) malloc(numimages*numsupvecs*sizeof(long int));
	long int* klin = matmult(images,Z,sendcounts[curr_rank]/imglen,imglen,supveclen,numsupvecs);
	sum =0;
	for (int i = 0;i< num_nodes;i++){
		sendcounts[i] = (sendcounts[i]/imglen)*numsupvecs;
		displs[i] = sum;
		sum+= sendcounts[i];
	}
	MPI_Gatherv(klin, sendcounts[curr_rank], MPI_LONG, klinTotal, sendcounts,displs, MPI_LONG, ROOT_NODE, MPI_COMM_WORLD);

	if (curr_rank == ROOT_NODE){
		//Form decision function 
		//(intercept rounds down to 0 for fixed point representation)
		long int* predictions = new long int[numimages];
		for (int i = 0; i < numimages; i++){
			predictions[i] = 0;
		}

		#pragma omp parallel for
		for (int i = 0; i < numimages; i++){
			for (int j = 0; j < numsupvecs; j++){
				predictions[i] += dualcoeffs[j] * klinTotal[numsupvecs*i+j];
			}
		}
		double end = MPI_Wtime();
		FILE * f = fopen("ExTimesParallel.txt","a+");
    		fprintf(f,"%f\n",end-time);
		fclose(f);
		//Make Prediction and check accuracy
		int num_correct = 0;
		for (int i = 0; i < numimages; i++){
			if (predictions[i] > 0)
				predictions[i] = 1;
			else
				predictions[i] = 0;

			if (predictions[i] == testtargets[i])
				num_correct++;
		}
		float accuracy = ((float)num_correct/numimages)*100;
		//cout << "Prediction Accuracy: "<< accuracy << "%" << endl;
		//fprintf(stderr,"Runtime: %f\n",MPI_Wtime()-time);

		delete[] dualcoeffs;
		delete[] testtargets;
		delete[] predictions;
		delete[] klinTotal;
	}
	free(images);
	free(Z);
	free(sendcounts);
	free(displs);
	delete[] klin;

	MPI_Finalize();
    return 0;
}

