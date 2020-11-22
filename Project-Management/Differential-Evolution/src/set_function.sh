#!/bin/bash

sed -i  '' "s/return .*.;/return $1;/g" src/check_function.cpp
g++ src/check_function.cpp -o src/test 2>/dev/null && rm src/test && sed -i '' "s/return .*.;/return $1;/g" src/function.h
