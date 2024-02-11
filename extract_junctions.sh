#!/bin/bash

# Initialize variables
input=""
output=""

# Function to show usage
usage() {
    echo "Usage: $0 -i <input> -o <output>"
    exit 1
}

# Parse command-line options
while getopts ":i:o:" opt; do
    case ${opt} in
        i )
            input=$OPTARG
            ;;
        o )
            output=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid Option: -$OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if input and output were provided
if [ -z "$input" ] || [ -z "$output" ]; then
    usage
fi


bam_path=$input
junc_path=$output

total_files=`find -P $bam_path/* -name '*.bam' | wc -l`
echo "Number of bam files: $total_files"

cd $bam_path
arr=( $(ls *.bam))

for ((i=0; i<$total_files; i+=1)); {
# echo ${arr[$i]};
regtools junctions extract -s XS -a 6 -m 50 -M 500000 ${arr[$i]} > $junc_path/${arr[$i]}.junc;
echo "[Done]$i : ${arr[$i]}";
}
