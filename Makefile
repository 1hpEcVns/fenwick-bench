.PHONY: all bench bench-rs bench-bare bench-all bench-run bench-run-rs bench-run-bare plot asm-check

CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++23
RUSTC ?= rustc

all: bench

bench: bench.cpp
	$(CXX) $(CXXFLAGS) bench.cpp -o bench

bench-rs: bench.rs
	$(RUSTC) --edition=2024 -O -C target-cpu=native bench.rs -o bench_rs

bench-bare: bench_no_std.rs
	$(RUSTC) --edition=2024 -O -C target-cpu=native -C panic=abort \
		-C link-arg=-nostartfiles -C link-arg=-static -C link-arg=-no-pie \
		-C link-arg=-fuse-ld=bfd bench_no_std.rs -o bench_bare

bench-run: bench
	taskset -c 0 ./bench > results.csv

bench-run-rs: bench-rs
	taskset -c 0 ./bench_rs > results_rs.csv

bench-run-bare: bench-bare
	taskset -c 0 ./bench_bare > results_bare.csv

bench-all: bench-run bench-run-rs

plot: bench-all
	nix develop --command python3 plot.py

asm-check:
	bash asm_check.sh
