#!/bin/bash

if [ ! -d build ]
then
    mkdir build
    cd build && (cmake .. 1>/dev/null) && (make 1>/dev/null)
fi

if [ $1 == "ackley" ]
then
    ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b1
elif [ $1 == "rastrigin" ]
then
    ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b2
elif [ $1 == "rosenbrock" ]
then
    ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b3
elif [ $1 == "schwefel" ]
then
    ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b4
else
    if [ $1 == "dea" ]
    then
        echo "$4" | ./main $3
    else
        echo "$4" | ./ga $3
    fi
fi
