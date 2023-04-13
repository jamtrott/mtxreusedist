#!/usr/bin/gnuplot
#
# Plot a histogram of reference distances for the matrix HB/1138_bus.
#
# Copyright (C) 2023 James D. Trotter
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <https://www.gnu.org/licenses/>.
#
# Authors:
#   James D. Trotter <james@simula.no>
#
#
# Usage:
#
# First, download the matrix HB/1138_bus from the SuiteSparse Matrix
# Collection at https://sparse.tamu.edu/HB/1138_bus.
#
# Next, run the command:
#
#   ./mtxreusedist --histogram --line-size=32 --verbose 1138_bus.mtx >1138_bus.txt
#
# Finally, plot the histogram:
#
#   gnuplot 1138_bus.gnu
#

set terminal pngcairo size 1024,768
set output '1138_bus.png'

set style fill solid 1.0

unset key
set xrange [-0.5:128]
set yrange [0.5:3000]

set xtics out nomirror
set xtics 0,10,120
set xtics add ("∞" (4054+32-1)/32)
set logscale y
set ytics out nomirror

set title 'HB/1138\_bus'
set xlabel 'reference distance (32 elements per cache line)'
set ylabel 'number of accesses / nonzeros'

plot '1138_bus.txt' using (x0=$1):(y0=$2) with boxes linecolor '#8dd3c7'
