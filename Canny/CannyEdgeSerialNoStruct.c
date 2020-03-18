#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define M_PI 3.14159265358979323846264338327

void compute_gradient(short int *input, int height, int width, short int *Gx, short int *Gy, short int *G,
						const float *kernelx, const float *kernely, const int ksize){
	int k_offset = ksize/2;
	int pix=0;
	for(int i = k_offset; i < height-k_offset; i++){
		for(int j = k_offset; j < width-k_offset; j++){
			pix = i*width+j;
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

void convolution(short int *input, int height, int width, short int *output, const float *kernel,const int ksize){
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
					output[i*width+j]= 255;
				}else if(result<0){
					output[i*width+j]= 0;
				} else{
					output[i*width+j]= (short int) result;
				}
			}
		}
}

int main(void){
	
	//initialization
	const int threshmax = 50;
	const int threshmin = 40;

	float gaussiankernel[25] = {2/159., 4/159., 5/159., 4/159., 2/159.,
								4/159., 9/159., 12/159., 9/159., 4/159.,
								5/159., 12/159., 15/159., 12/159., 5/159.,
								4/159., 9/159., 12/159., 9/159., 4/159.,
								2/159., 4/159., 5/159., 4/159., 2/159.};

	int gaussiankernelsize = 5;

	float kernelgx[9] = {-1,0,1,
					    -2,0,2,
					    -1,0,1};
 	int kernelgxsize=3;

 	float kernelgy[9] = {1, 2, 1,
					     0, 0, 0,
					    -1,-2,-1};
	int kernelgysize=3;
	int height=512, width=512;

	short int *input = calloc(height * width * sizeof(short int),1);
	short int *smoothed = calloc(height * width * sizeof(short int),1);
	short int *Gx = calloc(height * width * sizeof(short int),1);
	short int *Gy = calloc(height * width * sizeof(short int),1);
	short int *G = calloc(height * width * sizeof(short int),1);
	short int *nms = calloc(height * width * sizeof(short int),1);
	short int *output = calloc(height * width * sizeof(short int),1);
	char *src = "rock512.pgm";
	char *dst = "rock512_canny.pgm";
	
	imreadgray(src,input);
	fprintf(stderr,"%dx%d\n",height,width);
/*
	for(int i=0;i<512;i++){
		for(int j=0;j<512;j++){
			printf("%d ",input[i*width+j]);
		}
		printf("\n");
	}
*/ 


	int *edge = calloc(height*width*sizeof(int),1);

	double start=0,end=0;

	start = omp_get_wtime();
	//smooth the input image

	convolution(input,height,width,smoothed,gaussiankernel,gaussiankernelsize);
	/*
	for(int i=0;i<512;i++){
		for(int j=0;j<512;j++){
			printf("%d ",smoothed[i*width+j]);
		}
		printf("\n");
	}
	*/
	//compute x and y derivatives and gradients
	compute_gradient(smoothed,height,width,Gx,Gy,G,kernelgx,kernelgy,kernelgxsize);
 

	//non-maxsuppression
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
				nms[pix]=G[pix];
			else
				nms[pix]=0;
		}
	}

	pix=0;
	int nedges=0;
	int ed=0;
	//#pragma omp parallel for shared(input,nms,output) private(pix,n,s,e,w,nw,ne,sw,se,edge,nedges)
	for(int i=1;i<height-1;i++){
		for(int j=1;j<width-1;j++){
			pix = i*width+j;
			//fprintf(stderr,"%d\n",pix);
			if(nms[pix]>=threshmax && output[pix]==0){
				output[pix]=255;
				nedges=1;
				edge[0]=pix;
				
				do{
					nedges--;
					ed = edge[nedges];
					n=ed-width;
					s=ed+width;
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

	end = omp_get_wtime();
	//printf("ex time: %f\n",end-start);

	FILE * f = fopen("ExTimesSerial.txt","a+");
	fprintf(f,"%f\n",end-start);
	fclose(f);
	printf("%f\n",end-start);
	imwritegray(dst,nms,height,width);
	free(input);
	free(smoothed);
	free(Gx);
	free(Gy);
	free(nms);
	free(output);
	free(edge);
	
	return(0);

}