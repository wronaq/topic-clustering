#!/bin/sh

# remove first line
sed '1d' $1 > tmp1
sed '1d' $2 > tmp2

cat tmp1 tmp2 | cut -f2 > data.txt
rm -rf tmp1 tmp2

