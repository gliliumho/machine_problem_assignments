// gcc -o dotprod_pthread dotprod_pthread.c -lpthread && ./dotprod_pthread -n 1000000 -t 4

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


void *slave(void *ignored)
{
    int local_index, partial_sum = 0;
    do {
        // Lock and get thread's index
        pthread_mutex_lock(&mutex1);
        local_index = global_index;
        global_index++;
        pthread_mutex_unlock(&mutex1);

        if (local_index < array_size)
            // Calculate partial dot product
            partial_sum += a[local_index] * b[local_index];

    } while (local_index < array_size);

    // Lock and add sum calculated by each thread
    pthread_mutex_lock(&mutex1);
    sum += partial_sum;
    pthread_mutex_unlock(&mutex1);

    pthread_exit(NULL);
}


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
    // Allocate memory for arrays
    a = (int*)malloc( array_size*sizeof(int));
    b = (int*)malloc( array_size*sizeof(int));

    for (i=0; i<array_size; i++){
        a[i] = rand();
        b[i] = rand();
    }

    // Declare threads
    pthread_t thread[no_threads];
    // Initialize lock
    pthread_mutex_init(&mutex1,NULL);


    // Create threads with slave job..that sounded wrong
    for (i=0; i<no_threads; i++)
        if (pthread_create(&thread[i], NULL, slave, NULL) != 0)
            perror("Pthread_create fails");

    // Join threads (wait for their completion)
    for (i=0; i<no_threads; i++)
        if (pthread_join(thread[i], NULL) != 0)
            perror("Pthread_join fails");

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

    pthread_exit(NULL);
    //return 0;

}
