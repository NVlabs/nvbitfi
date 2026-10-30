# Offline tests (no GPU, no NVBit needed)
#   make -C tests check                       built-in expected cases
#   tests/extract_sass.sh <binary or .so> | tests/dests_test
#                                             per-opcode summary on real SASS
CXX ?= g++

dests_test: dests_test.cpp ../common/dests.h ../common/arch.h
	$(CXX) -std=c++17 -O2 -Wall -I../common $< -o $@

check: dests_test
	./dests_test --check

clean:
	rm -f dests_test

.PHONY: check clean
