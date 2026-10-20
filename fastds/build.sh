#!/bin/sh
# Builds FastDS (Cai, Hou, Wang, Luo & Lin, IJCAI 2020) from the code its
# co-author Yiyuan Wang published as NuMDS (github.com/yiyuanwang1988/NuMDS;
# the same header was first uploaded there as fastds-pure.h), with the gnn-co
# patch: a fractional cutoff in CPU seconds counted from the start of main
# (the original timed wall-clock time after reading and reducing the graph),
# a "t <seconds> <size>" line at every improvement and an "s <vertices>" line
# with the final dominating set. The code has no license, so it is downloaded
# here instead of being kept in the repo.
#   sh fastds/build.sh  ->  fastds/bin/fastds <graph.dimacs> <cutoff s> <seed>
set -e
cd "$(dirname "$0")"
commit=387edc2f4d5001c694a05e165fe85351a57e1b05
rm -rf build && mkdir -p build/src bin
while read -r sha f; do
    curl -sSL -o "build/src/$f" "https://raw.githubusercontent.com/yiyuanwang1988/NuMDS/$commit/$f"
    echo "$sha  build/src/$f" | sha256sum -c - 2>/dev/null \
        || echo "$sha  build/src/$f" | shasum -a 256 -c -
done <<SHAS
828e7ba3a66d1f06554425779c729f3e10bdb2b2c267796be788d5eac58e4c96 NuMDS.cpp
c15455db30d3966c7207684ab58a23a43891055d5e1d9d1e5afc5a2305f1962b NuMDS.h
e7c9c0c4abd6585cbffbeb04962f01cd9dfbf186bba89c18912222db2c70c9e6 basic.h
c6548182b7dce30c92302ddb400fafdabdda7717e0de3f34eb01998df672ccbb my_heap.h
SHAS
patch -s -p1 -d build/src < fastds_trace.patch
g++ -O3 -o bin/fastds build/src/NuMDS.cpp
echo "built $(pwd)/bin/fastds"
