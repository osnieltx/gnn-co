#!/bin/sh
# Builds NuMVC (Cai, Su, Luo & Sattar, JAIR 2013) from the authors' code with
# the gnn-co patch: an optional start cover, a fractional cutoff in CPU
# seconds (clock_gettime instead of times()), and a "t <seconds> <size>" line
# at every improvement. The authors' code has no license, so it is downloaded
# here instead of being kept in the repo.
#   sh numvc/build.sh   ->  numvc/bin/numvc <graph.dimacs> <target size|0> <seed> <cutoff s> [start cover]
set -e
cd "$(dirname "$0")"
url=http://lcs.ios.ac.cn/~caisw/Code/NuMVC-Code.zip
sha=c0445887df34cc4c1f46b373224c68922735ec6a9dedc12585c950e6bc69c606
rm -rf build && mkdir -p build bin
curl -sSL -o build/NuMVC-Code.zip "$url"
echo "$sha  build/NuMVC-Code.zip" | sha256sum -c - 2>/dev/null \
    || echo "$sha  build/NuMVC-Code.zip" | shasum -a 256 -c -
unzip -q build/NuMVC-Code.zip -d build
mkdir -p build/src
cp "build/NuMVC-Code/NuMVC code/numvc.cpp" "build/NuMVC-Code/NuMVC code/tsewf.h" build/src/
patch -s -p1 -d build/src < numvc_start.patch
g++ -O2 -o bin/numvc build/src/numvc.cpp
echo "built $(pwd)/bin/numvc"
