#!/usr/bin/env bash
# Check whether the compiler auto-vectorizes each semigroup prefix loop.
# Usage: CXX=g++ RUSTC=rustc bash asm_check.sh
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-g++}
RUSTC=${RUSTC:-rustc}

cat > /tmp/asm_ops.cpp <<'EOF'
#include <cstddef>
#include <cstdint>
using u64 = std::uint64_t;
using u32 = std::uint32_t;
u64 psum(const u64* a, size_t i){u64 s=0;for(size_t j=0;j<=i;++j)s+=a[j];return s;}
u64 pmin(const u64* a, size_t i){u64 s=~u64(0);for(size_t j=0;j<=i;++j)s=s<a[j]?s:a[j];return s;}
u64 pmax(const u64* a, size_t i){u64 s=0;for(size_t j=0;j<=i;++j)s=s>a[j]?s:a[j];return s;}
u64 pand(const u64* a, size_t i){u64 s=~u64(0);for(size_t j=0;j<=i;++j)s&=a[j];return s;}
u64 por(const u64* a, size_t i){u64 s=0;for(size_t j=0;j<=i;++j)s|=a[j];return s;}
u64 pxor(const u64* a, size_t i){u64 s=0;for(size_t j=0;j<=i;++j)s^=a[j];return s;}
u32 psum32(const u32* a, size_t i){u32 s=0;for(size_t j=0;j<=i;++j)s+=a[j];return s;}
EOF

cat > /tmp/asm_ops.rs <<'EOF'
#[unsafe(no_mangle)] pub fn psum(a: &[u64], i: usize) -> u64 { let mut s = 0u64; for &v in &a[..=i] { s = s.wrapping_add(v); } s }
#[unsafe(no_mangle)] pub fn pmin(a: &[u64], i: usize) -> u64 { let mut s = u64::MAX; for &v in &a[..=i] { s = s.min(v); } s }
#[unsafe(no_mangle)] pub fn pmax(a: &[u64], i: usize) -> u64 { let mut s = 0u64; for &v in &a[..=i] { s = s.max(v); } s }
#[unsafe(no_mangle)] pub fn pand(a: &[u64], i: usize) -> u64 { let mut s = u64::MAX; for &v in &a[..=i] { s &= v; } s }
#[unsafe(no_mangle)] pub fn por(a: &[u64], i: usize) -> u64 { let mut s = 0u64; for &v in &a[..=i] { s |= v; } s }
#[unsafe(no_mangle)] pub fn pxor(a: &[u64], i: usize) -> u64 { let mut s = 0u64; for &v in &a[..=i] { s ^= v; } s }
#[unsafe(no_mangle)] pub fn psum32(a: &[u32], i: usize) -> u32 { let mut s = 0u32; for &v in &a[..=i] { s = s.wrapping_add(v); } s }
fn main() {}
EOF

"$CXX" -O3 -march=native -std=c++23 -S /tmp/asm_ops.cpp -o /tmp/asm_ops_cpp.s
"$RUSTC" --edition=2024 -O -C target-cpu=native --emit=asm /tmp/asm_ops.rs -o /tmp/asm_ops_rs.s

printf '%-5s %-22s %-8s %-8s\n' op instr gcc# rust#
for op in sum min max and or xor sum32; do
    case "$op" in
        sum) pat='vpaddq'; lab='vpaddq' ;;
        # min/max: GCC/LLVM 不用 vpminuq/vpmaxuq，而是符号位技巧 +
        # vpcmpgtq + blend；出现 vpcmpgtq 即说明已向量化。
        min) pat='vpminuq|vpcmpgtq'; lab='vpminuq | vpcmpgtq' ;;
        max) pat='vpmaxuq|vpcmpgtq'; lab='vpmaxuq | vpcmpgtq' ;;
        and) pat='vpand'; lab='vpand' ;;
        or)  pat='vpor'; lab='vpor' ;;
        xor) pat='vpxor'; lab='vpxor' ;;
        sum32) pat='vpaddd'; lab='vpaddd' ;;
    esac
    c=$(grep -cE "$pat" /tmp/asm_ops_cpp.s || true)
    r=$(grep -cE "$pat" /tmp/asm_ops_rs.s || true)
    printf '%-5s %-22s %-8s %-8s\n' "$op" "$lab" "$c" "$r"
done

printf '\nsum32 = u32 累加（预期 vpaddd，8 路/向量）\n'
