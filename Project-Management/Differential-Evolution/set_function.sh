#!/bin/bash

sed -i  '' "s/return .*.;/return $1;/g" test.cpp
g++ test.cpp -o test 2>/dev/null && rm test
