// gcc -o dotprod_openmp dotprod_openmp.c -fopenmp && ./dotprod_openmp -n 1000000 -t 10

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

int array_m = 1000;  // Shared variable
int array_n = 100;  // Shared variable
int no_threads = 10;    // Can be local..

int main(int argc, char *argv[])
{
    // Seed the random number generator
    srand(time(NULL));
    //int i,j;

    // Take arguments to change array size and thread numbers
    for (int i=0; i<argc; i++){
        if (strcmp(argv[i],"-n") == 0 && (i < argc-1))
            array_n = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-m") == 0 && (i < argc-1))
            array_m = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-t") == 0 && (i < argc-1))
            no_threads = (int)strtol(argv[++i], NULL, 10);
    }
    printf("m = %d, n = %d\n", array_m, array_n);
    printf("No. of threads = %d\n", no_threads);


    int *a, *b, *p, *p_seq;
    // Allocate memory for arrays
    a = (int*)malloc( array_m*array_n*sizeof(int));
    b = (int*)malloc( array_n*sizeof(int));
    p = (int*)calloc( array_m,sizeof(int));
    p_seq = (int*)calloc( array_m,sizeof(int));

    for (int i=0; i<array_n; i++){
        for (int j=0; j<array_m; j++){
            a[j*array_n + i] = rand();
        }
        b[i] = rand();
    }


    omp_set_num_threads(no_threads);


#pragma omp parallel //private(partial_sum)
{
    //int id = omp_get_thread_num();

    // for(int i=id; i<array_m; i+=no_threads){
    for(int i=0; i<array_m; i++){
        int partial_sum = 0;

#pragma omp parallel for shared(partial_sum)
        for(int j=0; j<array_n; j++)
            partial_sum += a[i*array_n + j] * b[j];

        p[i] = partial_sum;
    }
}


    // Calculate matrix multiplication sequentially
    for(int i=0; i<array_m; i++)
        for(int j=0; j<array_n; j++)
            p_seq[i] += a[i*array_n + j] * b[j];

    // Verify for correctness of parallel calculation
    int correct=1;
    int sum = 0;
    int sum_seq = 0;
    for(int i=0; i<array_m; i++){
        if (p[i] != p_seq[i])
            correct=0;
        sum += p[i];
        sum_seq += p_seq[i];
    }


    if (correct){
        printf("Parallel results are correct.\n");
        printf("Sum of parallel = %d, sum of seq = %d\n", sum, sum_seq);
    } else {
        printf("Parallel results are INCORRECT!!!\n");
    }

    //return 0;

}
