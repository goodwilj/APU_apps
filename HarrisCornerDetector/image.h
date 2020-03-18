#include <stdint.h>

int imreadcolor(char *src, int **img);
int imreadgray(char *src, int *img);
void rgb2gray(int **imgcolor, int *imggray, int h, int w);
int imwritegray(char *dst, int *img, int h, int w);