#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "image.h"
#include <math.h>
#include <omp.h>

#define MAIN_NODE 0
#define imgheight 512
#define imgwidth 512

void convolution(short int *input, int height, int width, short int *output, const float *kernel,const int offset, const int ksize){
		int k_offset = ksize/2;
		for(int i = k_offset; i < height-k_offset; i++){
			for(int j = k_offset; j < width-k_offset; j++){
				float result=0;
				for(int y=0;y<ksize;y++){
					for(int x=0;x<ksize;x++){

						result += input[(i+y-k_offset)*width+(j+x-k_offset)] * kernel[y*ksize+x];
					}
				}

				if(result>255){
					output[(i-offset)*width+j]= 255;
				}else if(result<0){
					output[(i-offset)*width+j]= 0;
				} else{
					output[(i-offset)*width+j]= (short int) result;
				}
			}
		}

}

void compute_gradient(short int *input, int height, int width, short int *Gx, short int *Gy, short int *G,
						const float *kernelx, const float *kernely, const int ksize, const int offset){
	int k_offset = ksize/2;
	int pix=0;
	for(int i = k_offset; i < height-k_offset; i++){
		for(int j = k_offset; j < width-k_offset; j++){
			pix = (i-offset)*width+j;
			Gx[pix]=0;
			Gy[pix]=0;
			G[pix]=0;
			for(int y=0;y<ksize;y++){
				for(int x=0;x<ksize;x++){
					Gx[pix]+= input[(i+y-k_offset)*width+(j+x-k_offset)] * kernelx[y*ksize+x];
					Gy[pix]+= input[(i+y-k_offset)*width+(j+x-k_offset)] * kernely[y*ksize+x];
				}
			}
			G[pix]=hypot(Gx[pix],Gy[pix]);
		}
	}
}

void nonmaxsupress(short int* Gx, short int* Gy, short int * G, int height, int width, short int* nms, const int offset){
	int pix=0;
	int n=0;
	int s=0;
	int e=0;
	int w=0;

	int nw=0;
	int ne=0;
	int sw=0;
	int se=0;

	float direction=0;
	float bin=0;
	//#pragma omp parallel for shared(input,nms,G) private(pix,n,s,e,w,nw,ne,sw,se,direction,bin)
	for(int i=1;i<height-1;i++){
		for(int j=1;j<width-1;j++){
			pix = i*width+j;
			n=pix-width;
			s=pix+width;
			w=pix-1;
			e=pix+1;
			nw=n-1;
			ne=n+1;
			sw=s-1;
			se=s+1;

			direction = atan2(Gy[pix],Gx[pix]);
			bin = fmod(direction+M_PI,M_PI)/M_PI * 8;
			if(((bin <=1 || bin>7) && G[pix]>G[e] && G[pix]>G[w]) || //0 degrees
			   ((bin >1 || bin <=3) && G[pix]>G[nw] && G[pix]>G[se]) || //45 degrees
			   ((bin >3 || bin <=5) && G[pix]> G[n] && G[pix]>G[s])  || //90 degrees
			   ((bin >5 || bin <=7) && G[pix] >G[ne] && G[pix]>G[sw]))  // 135 degrees
				nms[(i-offset)*width+j]=G[pix];
			else
				nms[(i-offset)*width+j]=0;
		}
	}

}

void sendshort(short int *data, int nelements, int dst){
	MPI_Send(data,nelements,MPI_SHORT,dst,0,MPI_COMM_WORLD);
}

void recvshort(short int *data, int nelements, int src){
	MPI_Recv(data,nelements,MPI_SHORT,src,0,MPI_COMM_WORLD, NULL);
}

void dividerows(int nprocs, int nrows, int extrarows, int *sendcounts, int *displs, int matrixwidth){
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

int main(int argc, char **argv){
	
	MPI_Init(&argc,&argv);
	int myproc=0, nprocs=0;
	MPI_Comm_rank(MPI_COMM_WORLD, &myproc); 
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	float gaussiankernel[25] = {2/159., 4/159., 5/159., 4/159., 2/159.,
								4/159., 9/159., 12/159., 9/159., 4/159.,
								5/159., 12/159., 15/159., 12/159., 5/159.,
								4/159., 9/159., 12/159., 9/159., 4/159.,
								2/159., 4/159., 5/159., 4/159., 2/159.};

	int gaussiankernelsize = 5;
	int k_offsetgauss = gaussiankernelsize/2;

	float kernelgx[9] = {-1,0,1,
					    -2,0,2,
					    -1,0,1};
 	int kernelgxsize=3;

 	float kernelgy[9] = {1, 2, 1,
					     0, 0, 0,
					    -1,-2,-1};
	int kernelgysize=3;


	short int *smoothed = calloc(imgheight * imgwidth * sizeof(short int),1);
	int nrows = imgheight/nprocs;
	int extrarows = imgheight%nprocs;
	int* sendcounts = malloc(nprocs*sizeof(int));
	int* displs = malloc(nprocs*sizeof(int));
	dividerows(nprocs,nrows,extrarows,sendcounts,displs,imgwidth);
	short int *partial_input;
	short int *conv_output;
	short int *Gx;
	short int *Gy;
	short int *G;
	short int *nms;

	if(myproc>1 || myproc<nprocs-1){
		partial_input=calloc((sendcounts[myproc]+(2*k_offsetgauss+4)*imgwidth) *sizeof(short int),1);
		conv_output=calloc((sendcounts[myproc]+4*imgwidth)*sizeof(short int),1);
		Gx=calloc((sendcounts[myproc]+2*imgwidth)*sizeof(short int),1);
		Gy=calloc((sendcounts[myproc]+2*imgwidth)*sizeof(short int),1);
		G=calloc((sendcounts[myproc]+2*imgwidth)*sizeof(short int),1);
		nms=calloc(sendcounts[myproc]*sizeof(short int),1);
	}else{
		partial_input=calloc((sendcounts[myproc]+(k_offsetgauss+2)*imgwidth) *sizeof(short int),1);
		conv_output=calloc((sendcounts[myproc]+2*imgwidth)*sizeof(short int),1);
		Gx=calloc((sendcounts[myproc]+imgwidth)*sizeof(short int),1);
		Gy=calloc((sendcounts[myproc]+imgwidth)*sizeof(short int),1);
		G=calloc((sendcounts[myproc]+imgwidth)*sizeof(short int),1);
		nms=calloc(sendcounts[myproc]*sizeof(short int),1);
	}

	if(myproc == MAIN_NODE){
		fprintf(stderr,"numprocs: %d\n",nprocs);
		for(int i=0;i<nprocs;i++){
			fprintf(stderr,"sendcounts %d displs %d\n",sendcounts[i],displs[i]);
		}
			
		short int *input = calloc(imgheight * imgwidth * sizeof(short int),1);
		short int *smoothed = calloc((sendcounts[myproc]+2*imgwidth) * sizeof(short int),1);
		nms=calloc(imgheight*imgwidth*sizeof(short int),1);
		short int *output = calloc(imgheight * imgwidth * sizeof(short int),1);
		int *edge = calloc(imgheight*imgwidth*sizeof(int),1);
		char *src = "rock512.pgm";
		char *dst = "rock512_canny.pgm";
		imreadgray(src,input);

		int pix=0;
		int n=0;
		int s=0;
		int e=0;
		int w=0;

		int nw=0;
		int ne=0;
		int sw=0;
		int se=0;
		int nedges=0;
		int ed=0;

		const int threshmax = 50;
		const int threshmin = 40;
		
		double start=0,end=0;

		start = omp_get_wtime();
		for(int i=1;i<nprocs;i++){
			if(i<nprocs-1){
				sendshort(input+displs[i]-(k_offsetgauss+2)*imgwidth,sendcounts[i]+(2*k_offsetgauss+4)*imgwidth,i);
			}else{
				sendshort(input+displs[i]-(k_offsetgauss+2)*imgwidth,sendcounts[i]+(k_offsetgauss+2)*imgwidth,i);
			}
		}

		convolution(input,sendcounts[myproc]/imgwidth+k_offsetgauss+2,imgwidth,smoothed,gaussiankernel,0,gaussiankernelsize);
		compute_gradient(smoothed,sendcounts[myproc]/imgwidth+2,imgwidth,Gx,Gy,G,kernelgx,kernelgy,kernelgxsize,0);
		nonmaxsupress(Gx, Gy, G,sendcounts[myproc]/imgwidth+1,imgwidth, nms,0);
		for(int i=1;i<nprocs;i++){
			recvshort(nms+displs[i],sendcounts[i],i);
		}

		//#pragma omp parallel for shared(input,nms,output) private(pix,n,s,e,w,nw,ne,sw,se,edge,nedges)
		for(int i=1;i<imgheight-1;i++){
			for(int j=1;j<imgwidth-1;j++){
				pix = i*imgwidth+j;
				//fprintf(stderr,"%d\n",pix);
				if(nms[pix]>=threshmax && output[pix]==0){
					output[pix]=255;
					nedges=1;
					edge[0]=pix;
					
					do{
						nedges--;
						ed = edge[nedges];
						n=ed-imgwidth;
						s=ed+imgwidth;
						w=ed-1;
						e=ed+1;
						nw=n-1;
						ne=n+1;
						sw=s-1;
						se=s+1;
						if (nms[n] >= threshmin && output[n] == 0) {
							
							output[n]=255;
							edge[nedges]=n;
							nedges++;
						}
						if (nms[s] >= threshmin && output[s] == 0) {
							
							output[s]=255;
							edge[nedges]=s;
							nedges++;
						}
						if (nms[w] >= threshmin && output[w] == 0) {
							
							output[w]=255;
							edge[nedges]=w;
							nedges++;
						}
						if (nms[e] >= threshmin && output[e] == 0) {
							
							output[e]=255;
							edge[nedges]=e;
							nedges++;
						}
						if (nms[nw] >= threshmin && output[nw] == 0) {
							
							output[nw]=255;
							edge[nedges]=nw;
							nedges++;
						}
						if (nms[ne] >= threshmin && output[ne] == 0) {
							
							output[ne]=255;
							edge[nedges]=ne;
							nedges++;
						}
						if (nms[sw] >= threshmin && output[sw] == 0) {
							
							output[sw]=255;
							edge[nedges]=sw;
							nedges++;
						}
						if (nms[se] >= threshmin && output[se] == 0) {
							
							output[se]=255;
							edge[nedges]=se;
							nedges++;
						}
						
					}while (nedges>0);

				}
			}
		}
		/*
		for(int i=0;i<512;i++){
			for(int j=0;j<512;j++){
				fprintf(stderr,"%d ",smoothed[i*imgwidth+j]);
			}
			fprintf(stderr,"\n");
		}


*/
		end = omp_get_wtime();
		//printf("ex time: %f\n",end-start);
		FILE * f = fopen("ExTimesParallel.txt","a+");
		fprintf(f,"%f\n",end-start);
		fclose(f);
		imwritegray(dst,output,imgheight,imgwidth);

	}else if(myproc==nprocs-1){
		recvshort(partial_input,sendcounts[myproc]+(k_offsetgauss+2)*imgwidth,MAIN_NODE);
		//fprintf(stderr,"here\n");
		convolution(partial_input,sendcounts[myproc]/imgwidth+k_offsetgauss+2,imgwidth,conv_output,gaussiankernel,k_offsetgauss,gaussiankernelsize);
		//fprintf(stderr,"here2\n");
		compute_gradient(conv_output,sendcounts[myproc]/imgwidth+2,imgwidth,Gx,Gy,G,kernelgx,kernelgy,kernelgxsize,1);
		//fprintf(stderr,"here3\n");
		nonmaxsupress(Gx, Gy, G, sendcounts[myproc]/imgwidth+1, imgwidth, nms,1);
		//fprintf(stderr,"here4\n");

		sendshort(nms,sendcounts[myproc],MAIN_NODE);
	}else{
		recvshort(partial_input,sendcounts[myproc]+(2*k_offsetgauss+4)*imgwidth,MAIN_NODE);
		convolution(partial_input,sendcounts[myproc]/imgwidth+2*k_offsetgauss+4,imgwidth,conv_output,gaussiankernel,k_offsetgauss,gaussiankernelsize);
		compute_gradient(conv_output,sendcounts[myproc]/imgwidth+4,imgwidth,Gx,Gy,G,kernelgx,kernelgy,kernelgxsize,1);
		nonmaxsupress(Gx, Gy, G, sendcounts[myproc]/imgwidth+2, imgwidth, nms,1);
		sendshort(nms,sendcounts[myproc],MAIN_NODE);
	}

	MPI_Finalize();
	return 0;
}