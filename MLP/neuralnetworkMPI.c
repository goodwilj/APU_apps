#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define MAIN_NODE 	0

#define imgbatchH 500
#define imgbatchW 785

#define fc1weightH 785
#define fc1weightW 512

#define layer1H 500
#define layer1W 513

#define fc2weightH 513
#define fc2weightW 512

#define layer2H 500
#define layer2W 513

#define fc3weightH 513
#define fc3weightW 10

#define layer3H 500
#define layer3W 10

void readbin(FILE* f, float* m, int h, int w);
inline void readbin(FILE* f, float* m, int h, int w) {
    fseek(f,0,SEEK_SET); //fread(mat.chan,sizeof(float),w*h,f);
    fread(m,sizeof(float),w*h,f);
}

void initappendedmatrix(float* mat, int h, int w){
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			if(j==0)
				mat[i*w + j]=1;
			else
				mat[i*w + j]=0;
		}
	}
}

void dividerows(int nprocs, int nrows, int extrarows,int *sendcounts, int *displs, int matrixwidth){
	int sum=0;
	for(int i=0;i<nprocs;i++){
		if(extrarows>0){
			sendcounts[i]=matrixwidth*nrows+matrixwidth;
			extrarows--;
		}else{
			sendcounts[i]=matrixwidth*nrows;
		}
		displs[i]=sum;
		sum+=sendcounts[i];
	}
}

void forwardprop(float* imgbatch, int myprocrows, float* fc1weight, float* layer1, float* fc2weight, float* layer2, float* fc3weight, float* layer3){
	//printf("%d",imgbatch.height);
	
    //printf("INFO#!#!: multiplying imgbatch with fc1weightmatrix...\n");
	
	for(unsigned int i=0;i<myprocrows;i++){
		for(unsigned int j=0;j<fc1weightW;j++){
			layer1[i*layer1W+j+1]=0;
			for(unsigned int k=0;k<imgbatchW;k++){
				layer1[i*layer1W+j+1] += imgbatch[i*imgbatchW+k] * fc1weight[k*fc1weightW+j];
			}
            //printf("%f\n",layer1[i*layer1W+j+1]);
			if(layer1[i*layer1W+j+1] < 0) {
				layer1[i*layer1W+j+1] = 0;
            }
		}
	}

    //printf("INFO#!#!: multipying layer1output with fc2weightmatrix...\n");
	//float sum2=0;
	for(unsigned int i=0;i<myprocrows;i++){
		for(unsigned int j=0;j<fc2weightW;j++){
			layer2[i*layer2W+j+1]=0;
			for(unsigned int k=0;k<layer1W;k++){
				layer2[i*layer2W+j+1] += (layer1[i*layer1W+k] * fc2weight[k*fc2weightW+j]);
			}
			if(layer2[i*layer2W+j+1] < 0) {
				layer2[i*layer2W+j+1] = 0;
            }
		}
	}

	for(unsigned int i=0;i<myprocrows;i++){
		for(unsigned int j=0;j<fc3weightW;j++){
			layer3[i*layer3W+j]=0;
			for(unsigned int k=0;k<layer2W;k++){
				layer3[i*layer3W+j] += (layer2[i*layer2W+k] * fc3weight[k*fc3weightW+j]);
			}
		}
	}
    //printf("INFO#!#!: Done with forward propagation!\n");

}

void broadcast(float* matrix,int nelements){
	MPI_Bcast(matrix,nelements,MPI_FLOAT,MAIN_NODE,MPI_COMM_WORLD);
}

void scatterv(float* matrix, int* sendcounts, int* displs, 
	float* dest, int destnelements){
	MPI_Scatterv(matrix, sendcounts,displs,MPI_FLOAT, dest,
		destnelements,MPI_FLOAT,MAIN_NODE,MPI_COMM_WORLD);
}

void gatherv(float* partial_result, int division_size,
	float* dest, int* sendcounts, int* displs){
	
	MPI_Gatherv(partial_result,division_size,MPI_FLOAT,dest,sendcounts,
		displs,MPI_FLOAT,MAIN_NODE,MPI_COMM_WORLD);
}

int main(int argc, char **argv){
	MPI_Init(&argc,&argv);
	int myproc = 0, nprocs =0;
	MPI_Comm_rank(MPI_COMM_WORLD, &myproc); // get the rank of the process
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // get the size of the communicator
	double start=0,end=0;

	float *final_result = malloc(sizeof(float) * layer3H * layer3W);
	float *fc1weight = malloc(sizeof(float) * fc1weightH * fc1weightW);
	float *fc2weight = malloc(sizeof(float) * fc2weightH * fc2weightW);
	float *fc3weight = malloc(sizeof(float) * fc3weightH * fc3weightW);

	int nrows=0,extrarows=0;
	nrows = imgbatchH/nprocs;
	extrarows = imgbatchH%nprocs;

	int *sendcountsinit = malloc(nprocs*sizeof(int));
	int *displsinit = malloc(nprocs*sizeof(int));
	dividerows(nprocs, nrows, extrarows,sendcountsinit,displsinit,imgbatchW);
	int division_sizeinit = sendcountsinit[myproc];
	int myprocrows = division_sizeinit/imgbatchW;

	float *imgbatch = malloc(sizeof(float) * imgbatchH * imgbatchW);
	float *imgbatch_div = malloc(division_sizeinit*sizeof(float));

	float *layer1 = malloc(myprocrows * layer1W * sizeof(float));
	initappendedmatrix(layer1,myprocrows,layer1W);
	float *layer2 = malloc(myprocrows * layer2W * sizeof(float));
	initappendedmatrix(layer2,myprocrows,layer2W);
	float *layer3 = malloc(myprocrows * layer3W * sizeof(float));

	int *sendcountsfinal = malloc(nprocs*sizeof(int));
	int *displsfinal = malloc(nprocs*sizeof(int));
	dividerows(nprocs, nrows, extrarows,sendcountsfinal,displsfinal,layer3W);
	int division_sizefinal = sendcountsfinal[myproc];

	if(myproc == MAIN_NODE){

		FILE* imgbatchsrc = fopen("batchtest.bin","rb");
	    //float *imgbatch = malloc(sizeof(float) * imgbatchH * imgbatchW);
		readbin(imgbatchsrc, imgbatch, imgbatchH, imgbatchW);
	    fclose(imgbatchsrc);

	    FILE *fc1w_src;
	    fc1w_src = fopen("fc1_wb.bin","rb");
		readbin(fc1w_src, fc1weight,fc1weightH,fc1weightW);
	    fclose(fc1w_src);

	    FILE *fc2w_src;
	    fc2w_src = fopen("fc2_wb.bin","rb");
		readbin(fc2w_src, fc2weight,fc2weightH,fc2weightW);
	    fclose(fc2w_src);

	    FILE *fc3w_src;
	    fc3w_src = fopen("fc3_wb.bin","rb");
		readbin(fc3w_src, fc3weight,fc3weightH,fc3weightW);
	    fclose(fc3w_src);

		start = MPI_Wtime();

	}

	broadcast(fc1weight,fc1weightH*fc1weightW);
	broadcast(fc2weight,fc2weightH*fc2weightW);
	broadcast(fc3weight,fc3weightH*fc3weightW);

	scatterv(imgbatch,sendcountsinit,displsinit,imgbatch_div,division_sizeinit);

	forwardprop(imgbatch_div, myprocrows, fc1weight, layer1, fc2weight, layer2, fc3weight, layer3);

	gatherv(layer3,division_sizefinal,final_result,sendcountsfinal,displsfinal);

	if(myproc == MAIN_NODE){
		end = MPI_Wtime();
		double extime = end - start;
		FILE * f = fopen("ExTimesParallel.txt","a+");
		fprintf(f,"%f\n",end-start);
		fclose(f);
		//printf("Execution Time: %f\n",extime);
		/*
		for(int i=0;i<10;i++){
			fprintf(stderr,"\n");
			for(int j=0;j<10;j++){
				fprintf(stderr,"%f ",final_result[i*10+j]);
			}
		}
		fprintf(stderr,"\n");
		*/
		free(imgbatch);
	}


	free(final_result);
	free(fc1weight);
	free(fc2weight);
	free(fc3weight);
	free(imgbatch_div);
	free(layer1);
	free(layer2);
	free(layer3);
	free(sendcountsinit);
	free(displsinit);
	free(sendcountsfinal);
	free(displsfinal);
	MPI_Finalize();
	return 0;

}