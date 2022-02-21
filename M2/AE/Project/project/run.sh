#!/bin/bash

for i in {1..15}
do
    ./mini_project | tail -1 | cut -d ':' -f 2 >> losses.txt
done
