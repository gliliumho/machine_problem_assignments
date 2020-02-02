// gcc -o dotprod_openmp dotprod_openmp.c -fopenmp && ./dotprod_openmp -n 1000000 -t 10

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
    }
    printf("Array size = %d\n", array_size);
    printf("No. of threads = %d\n", no_threads);


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

    // int *partial_sum;
    // partial_sum = (int*)malloc( no_threads*sizeof(int));

    omp_set_num_threads(no_threads);

    int partial_sum = 0;
#pragma omp parallel private(i, partial_sum)
{
    int id = omp_get_thread_num();
    partial_sum = 0;
    for(i=id; i<array_size; i+=no_threads)
        partial_sum += a[i] * b[i];
        // partial_sum[id] += a[i] * b[i];

#pragma omp critical
    sum += partial_sum;
}
    // for(i=0; i<no_threads; i++)
    //     sum += partial_sum[i];


    // Calculate dot product sequentially
    for (i=0; i<array_size; i++)
        seq_sum += a[i]*b[i];

    // Verify dot product result
    if (seq_sum == sum){
        printf("Parallel results are correct.\n");
        printf("The dot product is %d\n", sum);
    } else {
        printf("Parallel results are INCORRECT!!!\n");
    }

    //return 0;

}
