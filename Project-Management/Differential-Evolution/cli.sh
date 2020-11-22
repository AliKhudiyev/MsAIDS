#!/bin/bash

if [ $1 == "ackley" ]
then
    echo "ackley"
    cd build && (cmake .. 1>/dev/null) && (make 1>/dev/null) && (./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b1)
elif [ $1 == "rastrigin" ]
then
    echo "rastrigin"
    cd build && (cmake .. 1>/dev/null) && (make 1>/dev/null) && (./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b2)
elif [ $1 == "rosenbrock" ]
then
    echo "rosenbrock"
    cd build && (cmake .. 1>/dev/null) && (make 1>/dev/null) && (./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b3)
elif [ $1 == "schwefel" ]
then
    echo "schwefel"
    cd build && (cmake .. 1>/dev/null) && (make 1>/dev/null) && (./main --global-min --population=30 --generation=100000 --benchmark-run=30 -o2 -b4)
else
    sed -i  '' "s/N_DIMENSION .*./N_DIMENSION $1/g" src/main.cpp
    src/set_function.sh "$2"
    exit_status=$?

    if [ $exit_status == 0 ]
    then
        if [ ! -d build ]
        then
            mkdir build
        fi
        cd build && (cmake .. 1>/dev/null) && (make 1>/dev/null) && (echo "$4" | ./main $3)
    else
        echo "Incorrent function!"
    fi
fi
