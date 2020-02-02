#!/bin/bash
cd ~/gitrepo/elec470_mp/
#gcc -o dotprod_openmp dotprod_openmp.c -fopenmp && ./dotprod_openmp $1 $2 $3 $4

# gcc -o matmult1 matmult1.c -fopenmp
# gcc -o matmult2 matmult2.c -fopenmp

for (( i=1; i<=$#; i++));
do
    if [ ${!i} == '-t' ]; then
        NUMTHREAD=$((i+1))
        export OMP_NUM_THREADS=${!NUMTHREAD}
        # echo $OMP_NUM_THREADS;
    fi
done

case $1 in
    "dotprod_pthread")
        gcc -o dotprod_pthread dotprod_pthread.c -lpthread
        ./dotprod_pthread $@
        ;;
    "dotprod_openmp")
        gcc -o dotprod_openmp dotprod_openmp.c -fopenmp
        ./dotprod_openmp $@
        ;;
    "matmult1")
        gcc -o matmult1 matmult1.c -fopenmp
        ./matmult1 $@
        ;;
    "matmult2")
        gcc -o matmult2 matmult2.c -fopenmp
        ./matmult2 $@
        ;;
    *)
        echo "Invalid first argument.\n"
        ;;
    esac

# for i in {1..50}
# do

    # ./matmult1 $1 $2 $3 $4 $5 $6 #| grep "correct\|INCORRECT"
    # ./matmult2 $1 $2 $3 $4 $5 $6  | grep "correct\|INCORRECT"
# done
