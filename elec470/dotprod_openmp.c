/*
Kin Yee Ho
10049579

Run:
gcc -o dotprod_openmp dotprod_openmp.c -fopenmp
./dotprod_openmp [-n array_size] [-t NO_THREADS]

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

int array_size = 1000;  // Shared variable
int no_threads = 10;    // Can be local..

int main(int argc, char *argv[])
{
    // Seed the random number generator
    srand(time(NULL));
    int i;

    // Take arguments to change array size and thread numbers
    for (i=0; i<argc; i++){
        if (strcmp(argv[i],"-n") == 0 && (i < argc-1))
            array_size = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-t") == 0 && (i < argc-1))
            no_threads = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-h") == 0){
            printf("usage: ./dotprod_openmp [-n ARRAY_SIZE] [-t NO_THREADS]\n");
            exit(0);
        }
    }
    printf("array_size = %d, no_threads = %d\tpart1a: OpenMP\n", array_size, no_threads);


    int seq_sum = 0;
    int sum = 0;
    int *a, *b;
    // Allocate memory for arrays
    a = (int*)malloc( array_size*sizeof(int));
    b = (int*)malloc( array_size*sizeof(int));

    for (i=0; i<array_size; i++){
        a[i] = rand();
        b[i] = rand();
    }


    omp_set_num_threads(no_threads);

    double start_time = omp_get_wtime();

    int partial_sum = 0;
#pragma omp parallel private(i, partial_sum)
{
    int id = omp_get_thread_num();
    partial_sum = 0;
    for(i=id; i<array_size; i+=no_threads)
        partial_sum += a[i] * b[i];

#pragma omp critical
    sum += partial_sum;
}

    double elapsedtime = omp_get_wtime() - start_time;
    printf("Elapsed time on OpenMP dot product: \t%f s\n", elapsedtime);

    start_time = omp_get_wtime();
    // Calculate dot product sequentially
    for (i=0; i<array_size; i++)
        seq_sum += a[i]*b[i];
    elapsedtime = omp_get_wtime() - start_time;
    printf("Elapsed time on seq. dot product: \t%f s\n", elapsedtime);

    // Verify dot product result
    if (seq_sum != sum){
        printf("Parallel results are INCORRECT!!!\n");
        // printf("The dot product is %d\n", sum);
    } else {
        // printf("Parallel results are correct.\n");
    }


}
