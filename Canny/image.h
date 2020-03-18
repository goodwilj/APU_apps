#include <stdint.h>

int imreadcolor(char *src, short int **img);
int imreadgray(char *src, short int *img);
void rgb2gray(short int **imgcolor, short int *imggray, int h, int w);
int imwritegray(char *dst, short int *img, int h, int w);