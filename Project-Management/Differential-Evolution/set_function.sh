#!/bin/bash

sed -i  '' "s/return .*.;/return $1;/g" check_function.cpp
g++ check_function.cpp -o test 2>/dev/null && rm test && sed -i '' "s/return .*.;/return $1;/g" function.h
