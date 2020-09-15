#!/bin/bash

# cd ~/pw01
mkdir fold 2>/dev/null
cd fold
touch ex8ex{1,2,3,4,45,05,89,A,B}
mkdir -p docs/dirU/{dirUA,dirUB,dirUC/dirUCA}
cd docs/dirU/dirUC/dirUCA

s1=yes
s2=no

if [ "$s1" == "$2" ]
then
    echo "test1 equal"
else
    echo "test1 not equal"
fi