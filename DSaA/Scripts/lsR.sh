#!/bin/zsh

for file in $(ls $1)
do
    if [ -d $1/$file ]
    then
	echo "dir: $1/$file"
	~/Desktop/MsAIDS/DSaA/Scripts/lsR.sh $1/$file
    else
	echo "file: $file"
    fi
done
