#!/bin/sh

mkdir -p preprocessed

# remove first line, label, change tabs to new lines
sed '1d' $1 | cut -f1,2 | sed 's/\t/\n/' > tmp1
sed '1d' $2 | sed 's/\t/\n/' > tmp2

# join and remove duplicates
cat tmp1 tmp2 | sort | uniq > preprocessed/data.txt
rm -rf tmp1 tmp2

