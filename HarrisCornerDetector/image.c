#include "image.h"
#include <stdio.h>
#include <stdlib.h>

int imreadcolor(char *src, int **img){
	FILE *file = fopen(src,"r");
	if (!file) {
		fprintf(stderr, "Could not open file %s\n", src);
		return 1;
	}

	unsigned int ppm_type, ppm_width, ppm_height;
	fscanf(file, "P%u\n%u %u\n255\n", &ppm_type, &ppm_width, &ppm_height);
	if(ppm_type != 6){
		fprintf(stderr,"Unrecognized ppm type %u\n",ppm_type);
		fclose(file);
		return 2;
	}

	for(int i=0;i<ppm_height;i++){
		for(int j =0;j<ppm_width;j++){
			for(int c=0;c<3;c++){
				fscanf(file,"%c",&(img[c][i*ppm_width+j]));
			}
		}
	}
	fclose(file);
	return 0;
}

int imreadgray(char *src, int *img){
	FILE *file = fopen(src,"r");
	if (!file) {
		fprintf(stderr, "Could not open file %s\n", src);
		return 1;
	}
	unsigned int ppm_type, ppm_width, ppm_height;
	fscanf(file, "P%u\n%u %u\n255\n", &ppm_type, &ppm_width, &ppm_height);
	if(ppm_type != 5){
		fprintf(stderr,"Unrecognized ppm type %u\n",ppm_type);
		fclose(file);
		return 2;
	}

	for(int i=0;i<ppm_height;i++){
		for(int j =0;j<ppm_width;j++){
				fscanf(file,"%c",&(img[i*ppm_width+j]));
		}
	}
	fclose(file);
	return 0;
}

void rgb2gray(int **imgcolor, int *imggray, int h, int w){
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			imggray[i*w+j] = (int) 0.2989 * imgcolor[0][i*w+j]  + 
													0.5870 * imgcolor[1][i*w+j] + 
													0.1140 * imgcolor[2][i*w+j];
		}
	}
}

int imwritegray(char *dst, int *img, int h, int w){
	FILE *file = fopen(dst,"wb");
	if (!file) {
		fprintf(stderr, "Could not open file %s\n", dst);
		return 1;
	}
	int ppm_height = h;
	int ppm_width = w;
	int ppm_type = 5;

	fprintf(file, "P%u\n%u %u\n255\n", ppm_type, ppm_width, ppm_height);

	for (int i = 0; i < ppm_height; i++)
		for (int j = 0; j < ppm_width; j++)
				fwrite(&(img[i*ppm_width + j]), 1, 1, file);
	
	fclose(file);
}