#!/bin/bash

if [ ! -d build ]
then
    mkdir build && cd build
    (cmake .. 1>/dev/null) && (make 1>/dev/null) && cd ..
fi

visual_flag=""
if [[ $# > 2 && $3 == "--visual" ]]
then
    visual_flag="--visual"
fi

if [[ $# == 0 || $1 == "help" ]]
then
    echo "Usage: cli.sh [ackley/rastrigin/rosenbrock/schwefel] [dea/ga] [--visual(opt)]"
    echo "Usage: cli.sh [metaheuristic] \"[args]\" [appr. point(opt)]"
    printf "\tmetaheuristic: dea/ga\n"
    printf "\targs:\n"
    printf "\t\t--population\n"
    printf "\t\t-V: variables\n"
    printf "\t\t-F: function\n"
    printf "\t\t--generation\n"
    printf "\t\t--global-min/max\n"
    printf "\t\t--threshold\n"
    printf "\t\t--benchmark-run\n"
    printf "\t\t--verbose\n"
    printf "\t\t--visual\n"
    printf "\t\t-o: optimization\n\n"

    printf "Example: ./cli.sh ackley dea\n"
    printf "Example: ./cli.sh dea \"--population=20 --threshold=0.001 -V x[0],x[1] -F -x[0]^2+x[1]^2 --visual\" -1.5\n"
elif [ $1 == "ackley" ]
then
    if [[ $# > 1 && $2 == "dea" ]]
    then
        cd build && ./dea --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b1 $visual_flag
    elif [[ $# > 1 && $2 == "ga" ]]
    then
        cd build && ./ga --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b1 $visual_flag
    fi
elif [ $1 == "rastrigin" ]
then
    if [[ $# > 1 && $2 == "dea" ]]
    then
        cd build && ./dea --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b2 $visual_flag
    elif [[ $# > 1 && $2 == "ga" ]]
    then
        cd build && ./ga --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b2 $visual_flag
    fi
elif [ $1 == "rosenbrock" ]
then
    if [[ $# > 1 && $2 == "dea" ]]
    then
        cd build && ./dea --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b3 $visual_flag
    elif [[ $# > 1 && $2 == "ga" ]]
    then
        cd build && ./ga --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b3 $visual_flag
    fi
elif [ $1 == "schwefel" ]
then
    if [[ $# > 1 && $2 == "dea" ]]
    then
        cd build && ./dea --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b4 $visual_flag
    elif [[ $# > 1 && $2 == "ga" ]]
    then
        cd build && ./ga --global-min --population=30 --generation=10000 --benchmark-run=30 -o2 -b4 $visual_flag
    fi
else
    if [ $1 == "dea" ]
    then
        cd build && (echo "$3" | ./dea $2)
    elif [ $2 == "ga" ]
    then
        cd build && (echo "$3" | ./ga $2)
    fi
fi
