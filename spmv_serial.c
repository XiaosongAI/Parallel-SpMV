#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <immintrin.h>
#include "mmio_highlevel.h"

#define NTIMES 200

int main(int argc, char ** argv)
{
	struct timeval t1, t2;
	char *filename = argv[1];
	printf ("filename = %s\n", filename);

	//read matrix
	int m, n, nnzR, isSymmetric;

	mmio_info(&m, &n, &nnzR, &isSymmetric, filename);
	int *RowPtr = (int *)malloc((m+1) * sizeof(int));
	int *ColIdx = (int *)malloc(nnzR * sizeof(int));
	float *Val    = (float *)malloc(nnzR * sizeof(float));
	mmio_data(RowPtr, ColIdx, Val, filename);
	for (int i = 0; i < nnzR; i++)
	    Val[i] = 1;
	printf("The order of the rating matrix R is %i by %i, #nonzeros = %i\n",m, n, nnzR);

	//create X, Y,Y_golden
	float *X = (float *)malloc(sizeof(float) * n);
	float *Y = (float *)malloc(sizeof(float) * m);

	memset (X, 0, sizeof(float) * n);
	memset (Y, 0, sizeof(float) * m);
	
	for (int i = 0; i < n; i++)
		X[i] = 1;

	gettimeofday(&t1, NULL);
	for (int r = 0; r < NTIMES; r++)
	{
		for (int i = 0; i < m; i++)
		{
			float sum = 0;
			for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++) 
			{
				sum += Val[j] * X[ColIdx[j]];
			}
			Y[i] = sum;
		}
	}
	gettimeofday(&t2, NULL);
	float time_overall_serial = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
	float GFlops_serial = 2 * nnzR / time_overall_serial / pow(10,6);

	printf("GFlops_serial = %f\n", GFlops_serial);

	free(X);
	free(Y);
}