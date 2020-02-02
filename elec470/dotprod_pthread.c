/*
Kin Yee Ho
10049579

Run:
gcc -o dotprod_pthread dotprod_pthread.c -lpthread
./dotprod_pthread [-n array_size] [-t NO_THREADS]

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>


int *a;                 // Shared array
int *b;                 // Shared array
int array_size = 1000;  // Shared variable
int no_threads = 10;    // Can be local..
int global_index = 0;   // Shared index
int sum = 0;            // Shared result
pthread_mutex_t mutex1; // lock
double lock_time=0;

void *slave(void *ignored)
{
    int local_index, partial_sum = 0;
    struct timespec starttime, endtime;
    double elapsedtime;
    // double lock_time=0;
    int i;

    // do {
    //     clock_gettime(CLOCK_MONOTONIC, &starttime);
    //     // Lock and get thread's index
    //     pthread_mutex_lock(&mutex1);
    //     clock_gettime(CLOCK_MONOTONIC, &endtime);
    //     lock_time += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;
    //
    //     local_index = global_index;
    //     global_index++;
    //     pthread_mutex_unlock(&mutex1);
    //
    //
    //     if (local_index < array_size)
    //         // Calculate partial dot product
    //         partial_sum += a[local_index] * b[local_index];
    //
    // } while (local_index < array_size);

///////////////////////////////////////////////////////////////////////////////
    clock_gettime(CLOCK_MONOTONIC, &starttime);
    // Lock and get thread's index
    pthread_mutex_lock(&mutex1);
    clock_gettime(CLOCK_MONOTONIC, &endtime);
    lock_time += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;

    local_index = global_index;
    global_index++;
    pthread_mutex_unlock(&mutex1);

    for(i=local_index; i<array_size; i+=no_threads)
        partial_sum += a[i] * b[i];
///////////////////////////////////////////////////////////////////////////////

    clock_gettime(CLOCK_MONOTONIC, &starttime);
    // Lock and add sum calculated by each thread
    pthread_mutex_lock(&mutex1);
    clock_gettime(CLOCK_MONOTONIC, &endtime);
    lock_time += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;

    sum += partial_sum;
    pthread_mutex_unlock(&mutex1);


    pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
    // Seed the random number generator
    srand(time(NULL));
    int i;
    struct timespec starttime, endtime;
    double elapsedtime;


    // Take arguments to change array size and thread numbers
    for (i=0; i<argc; i++){
        if (strcmp(argv[i],"-n") == 0 && (i < argc-1))
            array_size = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-t") == 0 && (i < argc-1))
            no_threads = (int)strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i],"-h") == 0){
            printf("usage: ./dotprod_pthread [-n ARRAY_SIZE] [-t NO_THREADS]\n");
            exit(0);
        }
    }
    printf("array_size = %d, no_threads = %d \tpart1a: pthread\n", array_size, no_threads);



    // Allocate memory for arrays
    // printf("Allocating memory..\n");
    a = (int*)malloc( array_size*sizeof(int));
    b = (int*)malloc( array_size*sizeof(int));
    // printf("Done memory allocation.\n");
    // printf("Initializing A[] and B[] with random values..\n");
    for (i=0; i<array_size; i++){
        a[i] = rand();
        b[i] = rand();
    }
    // printf("Done array initialization.\n");

    // Declare threads
    pthread_t thread[no_threads];
    // Initialize lock
    pthread_mutex_init(&mutex1,NULL);




    clock_gettime(CLOCK_MONOTONIC, &starttime);
    // Create threads with slave job..that sounded wrong
    for (i=0; i<no_threads; i++)
        if (pthread_create(&thread[i], NULL, slave, NULL) != 0)
            perror("Pthread_create fails");

    // Join threads (wait for their completion)
    for (i=0; i<no_threads; i++)
        if (pthread_join(thread[i], NULL) != 0)
            perror("Pthread_join fails");

    clock_gettime(CLOCK_MONOTONIC, &endtime);
    elapsedtime = (endtime.tv_sec - starttime.tv_sec);
    elapsedtime += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;
    printf("Elapsed time on pthreads dot product: \t%f s\n", elapsedtime);
    printf("Elapsed time on LOCK (all cores): \t%f s\n", lock_time);

    clock_gettime(CLOCK_MONOTONIC, &starttime);
    // Calculate dot product sequentially
    int seq_sum = 0;
    for (i=0; i<array_size; i++)
        seq_sum += a[i]*b[i];

    clock_gettime(CLOCK_MONOTONIC, &endtime);
    elapsedtime = (endtime.tv_sec - starttime.tv_sec);
    elapsedtime += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;
    printf("Elapsed time on seq. dot product: \t%f s\n", elapsedtime);



    // Verify dot product result
    if (seq_sum != sum){
        printf("Parallel results are INCORRECT!!!\n");
        // printf("The dot product is %d\n", sum);
    } else {
        // printf("Parallel results are correct.\n");
    }

    pthread_exit(NULL);

}
