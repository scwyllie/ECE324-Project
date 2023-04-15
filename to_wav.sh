#!/bin/bash
 
search_path=$1

for i in $(find $search_path -type f); do
    ext="${i##*.}"
    if [[ $ext = m4a ]]
    then
        p=${i%".m4a"}
        echo $i
		ffmpeg -i $i $p'.wav'
        # ffmpeg -v 0  -i $i $p'.wav' </dev/null > /dev/null 2>&1 &
    fi
done
