#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

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


double timerval();
double timerval() {
    struct timeval st;
    gettimeofday(&st,NULL);
    return st.tv_sec + st.tv_usec * 1e-6;
}

void readbin(FILE* f, float* m, int h, int w);
inline void readbin(FILE* f, float* m, int h, int w) {
    fseek(f,0,SEEK_SET); //fread(mat.chan,sizeof(float),w*h,f);
    fread(m,sizeof(float),w*h,f);
}
/*
inline void readcsv(string &src, Mat &mat, int h, int w){
	FILE *f = fopen(src.c_str(), "rb");
	float temp;
	int count=0;
	mat.height = h;
	mat.width = w;
	mat.chan = new float[w*h];
	while(fscanf(f, "%f,", &temp)==1){
		mat.chan[count]=temp;
		count++;
	}
}
*/
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
/*
inline void initMat(float* mat, int h, int w){
	mat = new float[mat.height * mat.width];
}
*/
void forwardprop(float* imgbatch, float* fc1weight, float* layer1, float* fc2weight, float* layer2, float* fc3weight, float* layer3){
	//printf("%d",imgbatch.height);
	
   //printf("INFO#!#!: multiplying imgbatch with fc1weightmatrix...\n");
	
	for(unsigned int i=0;i<layer1H;i++){
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
	for(unsigned int i=0;i<layer2H;i++){
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

	for(unsigned int i=0;i<layer3H;i++){
		for(unsigned int j=0;j<fc3weightW;j++){
			layer3[i*layer3W+j]=0;
			for(unsigned int k=0;k<layer2W;k++){
				layer3[i*layer3W+j] += (layer2[i*layer2W+k] * fc3weight[k*fc3weightW+j]);
			}
		}
	}
    //printf("INFO#!#!: Done with forward propagation!\n");

}

int main(){

	//Must append csv with ones in first column (images are row wise vectors)

    FILE* imgbatchsrc = fopen("batchtest.bin","rb");
    float * imgbatch;
    imgbatch = malloc(sizeof(float) * imgbatchH * imgbatchW);
	readbin(imgbatchsrc, imgbatch, imgbatchH, imgbatchW);
    fclose(imgbatchsrc);

    FILE *fc1w_src;
    fc1w_src = fopen("fc1_wb.bin","rb");
    float* fc1weight;
	fc1weight = malloc(sizeof(float) * fc1weightH * fc1weightW);
	readbin(fc1w_src, fc1weight,fc1weightH,fc1weightW);
    fclose(fc1w_src);

    FILE *fc2w_src;
    fc2w_src = fopen("fc2_wb.bin","rb");
    float* fc2weight;
	fc2weight = malloc(sizeof(float) * fc2weightH * fc2weightW);
	readbin(fc2w_src, fc2weight,fc2weightH,fc2weightW);
    fclose(fc2w_src);

    FILE *fc3w_src;
    fc3w_src = fopen("fc3_wb.bin","rb");
    float* fc3weight;
	fc3weight = malloc(sizeof(float) * fc3weightH * fc3weightW);
	readbin(fc3w_src, fc3weight,fc3weightH,fc3weightW);
    fclose(fc3w_src);


	float* layer1;
    layer1 = malloc(sizeof(float) * layer1H * layer1W);
	initappendedmatrix(layer1, layer1H,layer1W);

	float* layer2;
    layer2 = malloc(sizeof(float) * layer2H * layer2W);
    initappendedmatrix(layer2, layer2H,layer2W);

	float* layer3;
    layer3 = malloc(sizeof(float) * layer3H * layer3W);
	
    double start=0,end=0;

	start = omp_get_wtime();

	forwardprop(imgbatch, fc1weight, layer1, fc2weight, layer2, fc3weight, layer3);

	end = omp_get_wtime();
	FILE * f = fopen("ExTimesSerial.txt","a+");
	fprintf(f,"%f\n",end-start);
	fclose(f);
/*
	printf("Layer 1 ouput:\n");
	for(int i=0;i<2;i++){
		printf("\n");
		for(int j=0;j<513;j++){
			printf("%f ",layer1[i*513+j]);
		}
	}
	printf("\n");

    for(int i=0;i<10;i++){
		printf("\n");
		for(int j=0;j<10;j++){
			printf("%f ",layer3[i*10+j]);
		}
	}
*/
    free(imgbatch);
    free(fc1weight);
    free(fc2weight);
    free(fc3weight);
    free(layer1);
    free(layer2);
    free(layer3);

	return 0;
}
