# mtxreusedist: program for computing reuse distance for sparse matrix-vector multiply
#
# Copyright (C) 2023 James D. Trotter
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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
# Authors: James D. Trotter <james@simula.no>
#
# Last modified: 2023-04-12

mtxreusedist = mtxreusedist

all: $(mtxreusedist)
clean:
	rm -f $(mtxreusedist_c_objects) $(mtxreusedist)
.PHONY: all clean

CFLAGS ?= -O2 -g -Wall
LDFLAGS ?= -lm

mtxreusedist_c_sources = mtxreusedist.c
mtxreusedist_c_headers =
mtxreusedist_c_objects := $(foreach x,$(mtxreusedist_c_sources),$(x:.c=.o))
$(mtxreusedist_c_objects): %.o: %.c $(mtxreusedist_c_headers)
	$(CC) -c $(CFLAGS) $< -o $@
$(mtxreusedist): $(mtxreusedist_c_objects)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
