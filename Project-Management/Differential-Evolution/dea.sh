#!/bin/bash

# sed -i  '' "s/N_DIMENSION .*./N_DIMENSION $1/g" main.cpp
# sed -i  '' "s/return .*.;/return $2;/g" function.h

cd build && make && ./main $3
