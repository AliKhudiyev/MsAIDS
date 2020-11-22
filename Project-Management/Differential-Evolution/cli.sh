#!/bin/bash

sed -i  '' "s/N_DIMENSION .*./N_DIMENSION $1/g" src/main.cpp
src/set_function.sh "$2"
# sed -i  '' "s/return .*.;/return $2;/g" function.h
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
