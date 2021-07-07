#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>
#include "mmio_highlevel.h"

#define NTIMES 200

int binary_search_right_boundary_kernel(const int *row_pointer, const int  key_input, const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;
    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = row_pointer[median];
        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }
    return start;
}

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
	float *Y_golden = (float *)malloc(sizeof(float) * m);

	memset (X, 0, sizeof(float) * n);
	memset (Y, 0, sizeof(float) * m);
	memset (Y_golden, 0, sizeof(float) * m);
	
	for (int i = 0; i < n; i++)
		X[i] = 1;

	for (int i = 0; i < m; i++)
	    for(int j = RowPtr[i]; j < RowPtr[i+1]; j++)
		    Y_golden[i] += Val[j] * X[ColIdx[j]];

	int nthreads = atoi(argv[2]);
  	omp_set_num_threads(nthreads);
  	printf("#threads is %i \n", nthreads);

	int *csrSplitter = (int *)malloc((nthreads+1) * sizeof(int));
	int stridennz = ceil((double)nnzR/(double)nthreads);

	#pragma omp parallel for
	for (int tid = 0; tid <= nthreads; tid++)
	{
		int boundary = tid * stridennz;
		boundary = boundary > nnzR ? nnzR : boundary;
		csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
	}

	gettimeofday(&t1, NULL);
	for (int r = 0; r < NTIMES; r++)
	{
		#pragma omp parallel for
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
	float time_overall_parallel = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
	float GFlops_parallel = 2 * nnzR / time_overall_parallel / pow(10,6);
    int errorcount_parallel = 0;
	for (int i = 0; i < m; i++)
	    if (Y[i] != Y_golden[i])
		    errorcount_parallel++;

	printf("errorcount_parallel = %i\n", errorcount_parallel);
	printf("GFlops_parallel = %f\n", GFlops_parallel);

//------------------------------------------------------------------------

//-----------------------------------parallel_omp_balanced-------------------------------------
	gettimeofday(&t1, NULL);
	for (int r = 0; r < NTIMES; r++)
	{
		#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++)
		{
			for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
			{
				float sum = 0;
				for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) 
				{
					sum += Val[j] * X[ColIdx[j]];
				}
				Y[u] = sum;
			}
		}
	}
	gettimeofday(&t2, NULL);
	float time_overall_parallel_balanced = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
	float GFlops_parallel_balanced = 2 * nnzR / time_overall_parallel_balanced / pow(10,6);
    int errorcount_parallel_balanced = 0;
	for (int i = 0; i < m; i++)
	    if (Y[i] != Y_golden[i])
		    errorcount_parallel_balanced++;

	printf("errorcount_parallel_balanced = %i\n", errorcount_parallel_balanced);
	printf("GFlops_parallel_balanced = %f\n", GFlops_parallel_balanced);
	
    free(X);
    free(Y);
    free(Y_golden);
    free(csrSplitter);
}