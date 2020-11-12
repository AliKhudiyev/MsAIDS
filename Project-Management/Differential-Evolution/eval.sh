#!/bin/bash

read -p "Variables: " var
read -p "Function: " func

printf "$var\n$func\n" | python check.py

if [ $? == 0 ]
then
    echo "success"
    # ./dea.sh 1 "$func"
else
    echo "fail"
fi
