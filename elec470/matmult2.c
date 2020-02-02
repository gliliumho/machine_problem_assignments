/*
Kin Yee Ho
10049579

Run:
gcc -o matmult2 matmult2.c -fopenmp
./matmult2 [-m ROWS] [-n COLS] [-t NO_THREADS]

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

int array_m = 1000;  // Shared variable
int array_n = 100;  // Shared variable
int no_threads = 10;    // Can be local..

void parse_args(int argc, char *argv[])
{
    int i;
    // Take arguments to change array size and thread numbers
    for (i=0; i<argc; i++){
        if (strcmp(argv[i],"-n") == 0 && (i < argc-1))
            array_n = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-m") == 0 && (i < argc-1))
            array_m = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-t") == 0 && (i < argc-1))
            no_threads = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-h") == 0){
            printf("usage: ./matmult2 [-m ROWS] [-n COLS] [-t NO_THREADS]\n");
            exit(0);
        }
    }
}

void omp_matmult(int *p, int *a, int *b)
{
    int i,j;
    #pragma omp parallel for private(i,j)
    for(i=0; i<array_m; i++){
        int partial_sum=0;

        for(j=0; j<array_n; j++){
            partial_sum += a[i*array_n+j] * b[j];
            // int id = omp_get_thread_num();
            // printf("i=%d, j=%d, Thread=%d\n", i, j, id);
        }
        #pragma omp critical
        p[i] += partial_sum;
    }
}


void matmult(int *p, int *a, int *b)
{
    int i,j;
    for(i=0; i<array_m; i++)
        for(j=0; j<array_n; j++)
            p[i] += a[i*array_n + j] * b[j];
}


int main(int argc, char *argv[])
{
    // Seed the random number generator
    srand(time(NULL));

    parse_args(argc, argv);
    printf("m = %d, n = %d, no_threads = %d\tpart1b: OpenMP(embarrassing)\n", array_m, array_n, no_threads);

    double start_time, elapsedtime;
    int i,j;
    int correct;

    int *a, *b, *p, *p_seq;
    // Allocate memory for arrays
    // printf("Allocating memory..\n");
    a = (int*)malloc( array_m*array_n*sizeof(int));
    b = (int*)malloc( array_n*sizeof(int));
    p = (int*)calloc( array_m,sizeof(int));
    p_seq = (int*)calloc( array_m,sizeof(int));
    // printf("Done memory allocation.\n");

    // Randomize a[] and b[]
    for (i=0; i<array_n; i++){
        for (j=0; j<array_m; j++)
            a[j*array_n + i] = rand();
        b[i] = rand();
    }

    omp_set_num_threads(no_threads);

    // Calculate matrix multiplication in parallel
    start_time = omp_get_wtime();
    omp_matmult(p,a,b);
    elapsedtime = omp_get_wtime() - start_time;
    printf("Elapsed time on OpenMP matrix mult: \t%f s\n", elapsedtime);

    // Calculate matrix multiplication sequentially
    start_time = omp_get_wtime();
    matmult(p_seq,a,b);
    elapsedtime = omp_get_wtime() - start_time;
    printf("Elapsed time on seq. matrix mult: \t%f s\n", elapsedtime);


    // Verify for correctness of parallel calculation
    correct=1;
    // int sum = 0;
    // int sum_seq = 0;
    for(i=0; i<array_m; i++){
        if (p[i] != p_seq[i])
            correct=0;
        // sum += p[i];
        // sum_seq += p_seq[i];
    }


    if (correct==0){
        printf("Parallel results are INCORRECT!!!\n");
        // printf("Sum of parallel = %d, sum of seq = %d\n", sum, sum_seq);
    } else {
        // printf("Parallel results are correct.\n");
    }


}
