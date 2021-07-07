#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>
#include "mmio_highlevel.h"

#define NTIMES 200

float hsum_avx(__m256 in256)
{
    float sum;

    __m256 hsum = _mm256_hadd_ps(in256, in256);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sum, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );

    return sum;
}

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

	// find balanced points
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
		for (int tid = 0; tid < nthreads; tid++)
		{
			for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
			{
        		__m256 res = _mm256_setzero_ps();
				float sum = 0;
				int dif = RowPtr[u+1] - RowPtr[u];
				int nloop = dif / 8;
				int remainder = dif % 8;
				for (int li = 0; li < nloop; li++)
				{
					int j = RowPtr[u] + li * 8;
					__m256 vecv = _mm256_loadu_ps(&Val[j]);
					__m256i veci =  _mm256_loadu_si256((__m256i *)(&ColIdx[j]));
					__m256 vecx = _mm256_i32gather_ps(X, veci, 4);
					res = _mm256_fmadd_ps(vecv, vecx, res);
				}
				//Y[u] += _mm256_reduce_add_ps(res);
				sum += hsum_avx(res);

				for (int j = RowPtr[u] + nloop * 8; j < RowPtr[u + 1]; j++) {
					sum += Val[j] * X[ColIdx[j]];
				}
				Y[u] = sum;
			}

		}
	}
	gettimeofday(&t2, NULL);
	float time_overall_parallel_avx2 = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
	float GFlops_parallel_avx2 = 2 * nnzR / time_overall_parallel_avx2 / pow(10,6);
    int errorcount_parallel_avx2 = 0;
	for (int i = 0; i < m; i++)
	    if (Y[i] != Y_golden[i])
		    errorcount_parallel_avx2++;

	printf("errorcount_parallel_avx2 = %i\n", errorcount_parallel_avx2);
	printf("GFlops_parallel_avx2 = %f\n", GFlops_parallel_avx2);
	
    free(X);
    free(Y);
    free(Y_golden);
    free(csrSplitter);
}