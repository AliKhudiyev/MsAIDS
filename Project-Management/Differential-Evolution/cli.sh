#!/bin/bash

if [ ! -d build ]
then
    mkdir build
    cd build && (cmake .. 1>/dev/null) && (make 1>/dev/null)
fi

if [[ $# == 0 || $1 == "help" ]]
then
    echo "Usage: cli.sh [ackley/rastrigin/rosenbrock/schwefel]"
    echo "Usage: cli.sh [metaheuristic] [args] [appr. point(opt)]"
    printf "\tmetaheuristic: dea/ga\n"
    printf "\targs:\n"
    printf "\t\t--population-size\n"
    printf "\t\t-V: variables\n"
    printf "\t\t-F: function\n"
    printf "\t\t--generation\n"
    printf "\t\t--global-min/max\n"
    printf "\t\t--threshold\n"
    printf "\t\t--benchmark-run\n"
    printf "\t\t--verbose\n"
    printf "\t\t--visual\n"
    printf "\t\t-o: optimization\n"
elif [ $1 == "ackley" ]
then
    cd build && ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b1
elif [ $1 == "rastrigin" ]
then
    cd build && ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b2
elif [ $1 == "rosenbrock" ]
then
    cd build && ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b3
elif [ $1 == "schwefel" ]
then
    cd build && ./main --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b4
else
    if [ $1 == "dea" ]
    then
        cd build && echo "$3" | ./main $2
    else
        cd build && echo "$3" | ./ga $2
    fi
fi
