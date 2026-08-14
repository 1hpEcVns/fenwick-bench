// Bare-metal Rust (no_std / no_main) variant of the Fenwick benchmark.
// Runs directly on Linux x86-64 via syscalls: no libc, no std, no allocator.
// Hot kernels are identical to bench.rs; Vec/heap is replaced by static .bss
// buffers, std::time::Instant by clock_gettime, println by a syscall writer.
// Build:
//   rustc --edition=2024 -O -C target-cpu=native -C panic=abort \
//         -C link-arg=-nostartfiles -C link-arg=-static -C link-arg=-fuse-ld=bfd \
//         bench_no_std.rs -o bench_bare
#![no_std]
#![no_main]

use core::arch::asm;
use core::fmt::{self, Write};
use core::hint::black_box;
use core::ptr::addr_of_mut;

const ROUNDS: usize = 9;
const TARGET_NS: f64 = 3e6; // ~3 ms per timed pass
const Q_MIN: usize = 256;
const Q_MAX: usize = 4_000_000;
const MAX_N: usize = 1 << 20;
const HALF: usize = Q_MAX; // second half of the op buffers
const OPS_CAP: usize = 2 * Q_MAX;

// ---------------- static .bss memory (replaces heap) ----------------
static mut A: [u64; MAX_N] = [0; MAX_N];
static mut BIT: [u64; MAX_N + 1] = [0; MAX_N + 1];
static mut A0: [u64; MAX_N] = [0; MAX_N]; // bisect reset copy
static mut BIT0: [u64; MAX_N + 1] = [0; MAX_N + 1];
static mut A32: [u32; MAX_N] = [0; MAX_N];
static mut BIT32: [u32; MAX_N + 1] = [0; MAX_N + 1];
static mut POS: [usize; OPS_CAP] = [0; OPS_CAP];
static mut DELTA: [u64; OPS_CAP] = [0; OPS_CAP];
static mut ISQ: [u8; OPS_CAP] = [0; OPS_CAP];
static mut KS: [u64; OPS_CAP] = [0; OPS_CAP];
static mut LS: [usize; OPS_CAP] = [0; OPS_CAP];

unsafe fn arr64(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(A) as *mut u64, n) }
}
unsafe fn bit64(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(BIT) as *mut u64, n + 1) }
}
unsafe fn a064(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(A0) as *mut u64, n) }
}
unsafe fn bit064(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(BIT0) as *mut u64, n + 1) }
}
unsafe fn arr32(n: usize) -> &'static mut [u32] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(A32) as *mut u32, n) }
}
unsafe fn bit32(n: usize) -> &'static mut [u32] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(BIT32) as *mut u32, n + 1) }
}
unsafe fn pos(n: usize) -> &'static mut [usize] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(POS) as *mut usize, n) }
}
unsafe fn pos2(n: usize) -> &'static mut [usize] {
    unsafe { core::slice::from_raw_parts_mut((addr_of_mut!(POS) as *mut usize).add(HALF), n) }
}
unsafe fn delta(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(DELTA) as *mut u64, n) }
}
unsafe fn delta2(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut((addr_of_mut!(DELTA) as *mut u64).add(HALF), n) }
}
unsafe fn isq(n: usize) -> &'static mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(ISQ) as *mut u8, n) }
}
unsafe fn isq2(n: usize) -> &'static mut [u8] {
    unsafe { core::slice::from_raw_parts_mut((addr_of_mut!(ISQ) as *mut u8).add(HALF), n) }
}
unsafe fn ks(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(KS) as *mut u64, n) }
}
unsafe fn ks2(n: usize) -> &'static mut [u64] {
    unsafe { core::slice::from_raw_parts_mut((addr_of_mut!(KS) as *mut u64).add(HALF), n) }
}
unsafe fn ls(n: usize) -> &'static mut [usize] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(LS) as *mut usize, n) }
}
unsafe fn ls2(n: usize) -> &'static mut [usize] {
    unsafe { core::slice::from_raw_parts_mut((addr_of_mut!(LS) as *mut usize).add(HALF), n) }
}

// ---------------- Linux syscalls (no libc) ----------------
unsafe fn syscall3(n: u64, a: u64, b: u64, c: u64) -> u64 {
    let ret: u64;
    unsafe {
        asm!(
            "syscall",
            inlateout("rax") n => ret,
            in("rdi") a,
            in("rsi") b,
            in("rdx") c,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
    }
    ret
}

fn sys_write(s: &str) {
    unsafe {
        syscall3(1, 1, s.as_ptr() as u64, s.len() as u64);
    }
}

fn exit(code: u64) -> ! {
    unsafe {
        asm!(
            "syscall",
            in("rax") 60u64,
            in("rdi") code,
            options(noreturn)
        );
    }
}

// ---------------- timing: direct rdtsc instrumentation ----------------
// Raw clock_gettime/nanosleep syscalls are killed by this sandbox (SIGSEGV),
// so time is measured with rdtscp and converted with the CPUID base
// frequency (constant TSC; exact to ~0.5-1%).
fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!(
            "rdtscp",
            out("eax") lo,
            out("edx") hi,
            out("ecx") _,
            options(nomem, nostack)
        );
    }
    ((hi as u64) << 32) | lo as u64
}

fn cpuid(leaf: u32) -> (u32, u32, u32, u32) {
    let mut eax = leaf;
    let mut ecx: u32 = 0;
    let mut edx: u32 = 0;
    let ebx: u32;
    unsafe {
        asm!(
            "push rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "pop rbx",
            ebx_out = out(reg) ebx,
            inout("eax") eax,
            out("ecx") ecx,
            out("edx") edx,
            options(nostack)
        );
    }
    (eax, ebx, ecx, edx)
}

fn tsc_mhz() -> f64 {
    let (eax, _, _, _) = cpuid(0x16);
    if eax != 0 {
        return eax as f64; // CPUID.16H EAX: base frequency in MHz (TSC rate)
    }
    let (eax, ebx, ecx, _) = cpuid(0x15);
    if ecx != 0 && eax != 0 {
        ecx as f64 * ebx as f64 / eax as f64 / 1e6
    } else {
        3000.0
    }
}

fn ns_per_cycle() -> f64 {
    static mut C: f64 = 0.0;
    unsafe {
        let p = addr_of_mut!(C);
        if *p == 0.0 {
            *p = 1e9 / (tsc_mhz() * 1e6);
        }
        *p
    }
}

fn now_ns() -> f64 {
    rdtsc() as f64 * ns_per_cycle()
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let mut out = Out;
    let _ = write!(out, "PANIC: {info}\n");
    exit(134);
}

struct Out;
impl fmt::Write for Out {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        sys_write(s);
        Ok(())
    }
}

// core's {:.3} float formatting pulls in bignum code that misbehaves in this
// bare setup; print ns with 3 decimals by scaling to picoseconds as integers.
fn print_row(name: &str, mode: &str, n: usize, tb: f64, tt: f64) {
    let ps_b = (tb * 1000.0) as u64;
    let ps_t = (tt * 1000.0) as u64;
    let mut out = Out;
    let _ = write!(
        out,
        "{},{},{},{}.{:03},{}.{:03}\n",
        name,
        mode,
        n,
        ps_b / 1000,
        ps_b % 1000,
        ps_t / 1000,
        ps_t % 1000
    );
}

// No libc, so provide the mem* symbols LLVM emits for fill/copy in the harness.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(s: *mut u8, c: i32, n: usize) -> *mut u8 {
    for i in 0..n {
        unsafe { *s.add(i) = c as u8 };
    }
    s
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    for i in 0..n {
        unsafe { *dst.add(i) = *src.add(i) };
    }
    dst
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memmove(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    if (dst as usize) < (src as usize) {
        for i in 0..n {
            unsafe { *dst.add(i) = *src.add(i) };
        }
    } else {
        for i in (0..n).rev() {
            unsafe { *dst.add(i) = *src.add(i) };
        }
    }
    dst
}

// ---------------- workload definitions ----------------
#[derive(Copy, Clone, PartialEq)]
enum Mix {
    Query,
    QueryTail,
    Mixed50,
    Mixed25,
    Mixed75,
}

fn is_query_op(mix: Mix, i: usize) -> bool {
    match mix {
        Mix::Query | Mix::QueryTail => true,
        Mix::Mixed50 => i & 1 == 0,
        Mix::Mixed25 => i % 4 != 0,
        Mix::Mixed75 => i % 4 == 0,
    }
}

fn rng_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn gen_pos(mix: Mix, n: usize, seed: &mut u64) -> usize {
    if mix == Mix::QueryTail {
        n - ((rng_next(seed) >> 32) as usize % (n / 10 + 1)) - 1
    } else {
        ((rng_next(seed) >> 32) as usize) % n
    }
}

trait Semi: Copy {
    const ID: u64;
    fn op(a: u64, b: u64) -> u64;
    fn combine_range(hi: u64, lo: u64) -> u64 {
        hi.wrapping_sub(lo)
    }
    fn invertible() -> bool {
        false
    }
}

#[derive(Copy, Clone)]
struct Sum;
impl Semi for Sum {
    const ID: u64 = 0;
    fn op(a: u64, b: u64) -> u64 {
        a.wrapping_add(b)
    }
    fn invertible() -> bool {
        true
    }
}

#[derive(Copy, Clone)]
struct Min;
impl Semi for Min {
    const ID: u64 = u64::MAX;
    fn op(a: u64, b: u64) -> u64 {
        a.min(b)
    }
}

#[derive(Copy, Clone)]
struct Max;
impl Semi for Max {
    const ID: u64 = 0;
    fn op(a: u64, b: u64) -> u64 {
        a.max(b)
    }
}

#[derive(Copy, Clone)]
struct And;
impl Semi for And {
    const ID: u64 = u64::MAX;
    fn op(a: u64, b: u64) -> u64 {
        a & b
    }
}

#[derive(Copy, Clone)]
struct Or;
impl Semi for Or {
    const ID: u64 = 0;
    fn op(a: u64, b: u64) -> u64 {
        a | b
    }
}

#[derive(Copy, Clone)]
struct Xor;
impl Semi for Xor {
    const ID: u64 = 0;
    fn op(a: u64, b: u64) -> u64 {
        a ^ b
    }
    fn combine_range(hi: u64, lo: u64) -> u64 {
        hi ^ lo
    }
    fn invertible() -> bool {
        true
    }
}

// ---------------- brute force kernels ----------------
#[inline(always)]
fn brute_prefix<O: Semi>(a: &[u64], i: usize) -> u64 {
    let mut s = O::ID;
    for &v in unsafe { a.get_unchecked(..=i) } {
        s = O::op(s, v);
    }
    s
}

#[inline(always)]
fn brute_update<O: Semi>(a: &mut [u64], i: usize, d: u64) {
    if O::invertible() {
        let slot = unsafe { a.get_unchecked_mut(i) };
        *slot = O::op(*slot, d);
    }
}

// ---------------- BIT kernels ----------------
#[inline(always)]
fn bit_prefix<O: Semi>(bit: &[u64], i: usize) -> u64 {
    let mut s = O::ID;
    let mut k = i.wrapping_add(1);
    while k != 0 {
        s = O::op(s, *unsafe { bit.get_unchecked(k) });
        k &= k.wrapping_sub(1);
    }
    s
}

#[inline(always)]
fn bit_apply<O: Semi>(bit: &mut [u64], n: usize, i: usize, d: u64) {
    let mut k = i.wrapping_add(1);
    while k <= n {
        let slot = unsafe { bit.get_unchecked_mut(k) };
        *slot = O::op(*slot, d);
        k = k.wrapping_add(k & k.wrapping_neg());
    }
}

fn top_of(n: usize) -> usize {
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

#[inline(always)]
fn brute_bisect(a: &[u64], n: usize, k: u64) -> usize {
    let mut s = 0u64;
    let mut j = 0usize;
    while j < n && s < k {
        s = s.wrapping_add(*unsafe { a.get_unchecked(j) });
        j += 1;
    }
    j
}

#[inline(always)]
fn bit_bisect(bit: &[u64], n: usize, top: usize, k: u64) -> usize {
    let mut pos = 0usize;
    let mut m = top;
    let mut kk = k;
    while m != 0 {
        let next = pos + m;
        if next <= n {
            let v = *unsafe { bit.get_unchecked(next) };
            if v < kk {
                pos = next;
                kk -= v;
            }
        }
        m >>= 1;
    }
    if pos < n {
        pos + 1
    } else {
        n
    }
}

// ---------------- timed kernels (u64) ----------------
#[inline(never)]
fn run_mixed_brute<O: Semi>(a: &mut [u64], pos: &[usize], delta: &[u64], is_q: &[u8]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        let p = pos[i];
        if is_q[i] != 0 {
            acc = acc.wrapping_add(brute_prefix::<O>(a, p));
        } else {
            brute_update::<O>(a, p, delta[i]);
        }
    }
    acc
}

#[inline(never)]
fn run_mixed_bit<O: Semi>(bit: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        let p = pos[i];
        if is_q[i] != 0 {
            acc = acc.wrapping_add(bit_prefix::<O>(bit, p));
        } else {
            bit_apply::<O>(bit, n, p, delta[i]);
        }
    }
    acc
}

#[inline(never)]
fn run_query_brute<O: Semi>(a: &[u64], pos: &[usize]) -> u64 {
    let mut acc = 0u64;
    for &p in pos {
        acc = acc.wrapping_add(brute_prefix::<O>(a, p));
    }
    acc
}

#[inline(never)]
fn run_query_bit<O: Semi>(bit: &[u64], pos: &[usize]) -> u64 {
    let mut acc = 0u64;
    for &p in pos {
        acc = acc.wrapping_add(bit_prefix::<O>(bit, p));
    }
    acc
}

#[inline(never)]
fn run_range_brute<O: Semi>(a: &[u64], ls: &[usize], len: usize) -> u64 {
    let mut acc = 0u64;
    for &l in ls {
        let mut s = O::ID;
        for &v in unsafe { a.get_unchecked(l..l + len) } {
            s = O::op(s, v);
        }
        acc = acc.wrapping_add(s);
    }
    acc
}

#[inline(never)]
fn run_range_bit<O: Semi>(bit: &[u64], ls: &[usize], len: usize) -> u64 {
    let mut acc = 0u64;
    for &l in ls {
        let hi = bit_prefix::<O>(bit, l.wrapping_add(len).wrapping_sub(1));
        let lo = bit_prefix::<O>(bit, l.wrapping_sub(1));
        acc = acc.wrapping_add(O::combine_range(hi, lo));
    }
    acc
}

#[inline(never)]
fn run_bisect_mixed_brute(a: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8], ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        if is_q[i] != 0 {
            acc = acc.wrapping_add(brute_bisect(a, n, ks[i]) as u64);
        } else {
            let p = pos[i];
            let slot = unsafe { a.get_unchecked_mut(p) };
            *slot = slot.wrapping_add(delta[i]);
        }
    }
    acc
}

#[inline(never)]
fn run_bisect_mixed_bit(bit: &mut [u64], n: usize, top: usize, pos: &[usize], delta: &[u64], is_q: &[u8], ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        if is_q[i] != 0 {
            acc = acc.wrapping_add(bit_bisect(bit, n, top, ks[i]) as u64);
        } else {
            bit_apply::<Sum>(bit, n, pos[i], delta[i]);
        }
    }
    acc
}

#[inline(never)]
fn run_bisect_query_brute(a: &[u64], n: usize, ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for &k in ks {
        acc = acc.wrapping_add(brute_bisect(a, n, k) as u64);
    }
    acc
}

#[inline(never)]
fn run_bisect_query_bit(bit: &[u64], n: usize, top: usize, ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for &k in ks {
        acc = acc.wrapping_add(bit_bisect(bit, n, top, k) as u64);
    }
    acc
}

// ---------------- timed kernels (u32, sum only) ----------------
#[inline(always)]
fn brute_prefix_u32(a: &[u32], i: usize) -> u32 {
    let mut s = 0u32;
    for &v in unsafe { a.get_unchecked(..=i) } {
        s = s.wrapping_add(v);
    }
    s
}

#[inline(always)]
fn bit_prefix_u32(bit: &[u32], i: usize) -> u32 {
    let mut s = 0u32;
    let mut k = i.wrapping_add(1);
    while k != 0 {
        s = s.wrapping_add(*unsafe { bit.get_unchecked(k) });
        k &= k.wrapping_sub(1);
    }
    s
}

#[inline(always)]
fn bit_apply_u32(bit: &mut [u32], n: usize, i: usize, d: u32) {
    let mut k = i.wrapping_add(1);
    while k <= n {
        let slot = unsafe { bit.get_unchecked_mut(k) };
        *slot = slot.wrapping_add(d);
        k = k.wrapping_add(k & k.wrapping_neg());
    }
}

#[inline(never)]
fn run_query_brute_u32(a: &[u32], pos: &[usize]) -> u64 {
    let mut acc = 0u64;
    for &p in pos {
        acc = acc.wrapping_add(brute_prefix_u32(a, p) as u64);
    }
    acc
}

#[inline(never)]
fn run_query_bit_u32(bit: &[u32], pos: &[usize]) -> u64 {
    let mut acc = 0u64;
    for &p in pos {
        acc = acc.wrapping_add(bit_prefix_u32(bit, p) as u64);
    }
    acc
}

#[inline(never)]
fn run_mixed_brute_u32(a: &mut [u32], pos: &[usize], delta: &[u64], is_q: &[u8]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        let p = pos[i];
        if is_q[i] != 0 {
            acc = acc.wrapping_add(brute_prefix_u32(a, p) as u64);
        } else {
            let slot = unsafe { a.get_unchecked_mut(p) };
            *slot = slot.wrapping_add(delta[i] as u32);
        }
    }
    acc
}

#[inline(never)]
fn run_mixed_bit_u32(bit: &mut [u32], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        let p = pos[i];
        if is_q[i] != 0 {
            acc = acc.wrapping_add(bit_prefix_u32(bit, p) as u64);
        } else {
            bit_apply_u32(bit, n, p, delta[i] as u32);
        }
    }
    acc
}

#[inline(never)]
fn run_range_brute_u32(a: &[u32], ls: &[usize], len: usize) -> u64 {
    let mut acc = 0u64;
    for &l in ls {
        let mut s = 0u32;
        for &v in unsafe { a.get_unchecked(l..l + len) } {
            s = s.wrapping_add(v);
        }
        acc = acc.wrapping_add(s as u64);
    }
    acc
}

#[inline(never)]
fn run_range_bit_u32(bit: &[u32], ls: &[usize], len: usize) -> u64 {
    let mut acc = 0u64;
    for &l in ls {
        let hi = bit_prefix_u32(bit, l + len - 1);
        let lo = bit_prefix_u32(bit, l.wrapping_sub(1));
        acc = acc.wrapping_add(hi.wrapping_sub(lo) as u64);
    }
    acc
}

// ---------------- time wrappers ----------------
fn time_mixed_brute<O: Semi>(a: &mut [u64], pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = now_ns();
    let acc = run_mixed_brute::<O>(a, pos, delta, is_q);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_mixed_bit<O: Semi>(bit: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = now_ns();
    let acc = run_mixed_bit::<O>(bit, n, pos, delta, is_q);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_query_brute<O: Semi>(a: &[u64], pos: &[usize]) -> f64 {
    let t0 = now_ns();
    let acc = run_query_brute::<O>(a, pos);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_query_bit<O: Semi>(bit: &[u64], pos: &[usize]) -> f64 {
    let t0 = now_ns();
    let acc = run_query_bit::<O>(bit, pos);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_range_brute<O: Semi>(a: &[u64], ls: &[usize], len: usize) -> f64 {
    let t0 = now_ns();
    let acc = run_range_brute::<O>(a, ls, len);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_range_bit<O: Semi>(bit: &[u64], ls: &[usize], len: usize) -> f64 {
    let t0 = now_ns();
    let acc = run_range_bit::<O>(bit, ls, len);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_bisect_mixed_brute(a: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8], ks: &[u64]) -> f64 {
    let t0 = now_ns();
    let acc = run_bisect_mixed_brute(a, n, pos, delta, is_q, ks);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_bisect_mixed_bit(bit: &mut [u64], n: usize, top: usize, pos: &[usize], delta: &[u64], is_q: &[u8], ks: &[u64]) -> f64 {
    let t0 = now_ns();
    let acc = run_bisect_mixed_bit(bit, n, top, pos, delta, is_q, ks);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_bisect_query_brute(a: &[u64], n: usize, ks: &[u64]) -> f64 {
    let t0 = now_ns();
    let acc = run_bisect_query_brute(a, n, ks);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_bisect_query_bit(bit: &[u64], n: usize, top: usize, ks: &[u64]) -> f64 {
    let t0 = now_ns();
    let acc = run_bisect_query_bit(bit, n, top, ks);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_query_brute_u32(a: &[u32], pos: &[usize]) -> f64 {
    let t0 = now_ns();
    let acc = run_query_brute_u32(a, pos);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_query_bit_u32(bit: &[u32], pos: &[usize]) -> f64 {
    let t0 = now_ns();
    let acc = run_query_bit_u32(bit, pos);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_mixed_brute_u32(a: &mut [u32], pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = now_ns();
    let acc = run_mixed_brute_u32(a, pos, delta, is_q);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_mixed_bit_u32(bit: &mut [u32], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = now_ns();
    let acc = run_mixed_bit_u32(bit, n, pos, delta, is_q);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_range_brute_u32(a: &[u32], ls: &[usize], len: usize) -> f64 {
    let t0 = now_ns();
    let acc = run_range_brute_u32(a, ls, len);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

fn time_range_bit_u32(bit: &[u32], ls: &[usize], len: usize) -> f64 {
    let t0 = now_ns();
    let acc = run_range_bit_u32(bit, ls, len);
    let t1 = now_ns();
    black_box(acc);
    t1 - t0
}

// ---------------- calibration / stats helpers ----------------
fn clamp_q(per_ns: f64) -> usize {
    ((TARGET_NS / per_ns) as usize).clamp(Q_MIN, Q_MAX)
}

fn median(v: &[f64]) -> f64 {
    // no_std has no slice::sort (it lives in alloc); insertion sort is fine
    // for ROUNDS = 9 values.
    let mut a = [0f64; ROUNDS];
    a.copy_from_slice(v);
    for i in 1..a.len() {
        let x = a[i];
        let mut j = i;
        while j > 0 && a[j - 1] > x {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = x;
    }
    a[a.len() / 2]
}

fn gen_ops(pos: &mut [usize], delta: &mut [u64], is_q: &mut [u8], n: usize, mix: Mix, seed: &mut u64) {
    for i in 0..pos.len() {
        pos[i] = gen_pos(mix, n, seed);
        delta[i] = rng_next(seed) % 1024;
        is_q[i] = if is_query_op(mix, i) { 1 } else { 0 };
    }
}

fn gen_bisect_ops(pos: &mut [usize], delta: &mut [u64], is_q: &mut [u8], ks: &mut [u64], n: usize, mix: Mix, seed: &mut u64) {
    for i in 0..pos.len() {
        pos[i] = gen_pos(mix, n, seed);
        delta[i] = rng_next(seed) % 1024;
        let qq = is_query_op(mix, i);
        is_q[i] = if qq { 1 } else { 0 };
        ks[i] = if qq { rng_next(seed) % (n as u64 * 512) + 1 } else { 0 };
    }
}

// ---------------- verification ----------------
fn verify_query<O: Semi>(n: usize) {
    let mut seed = 0xd1b5_4a32_d192_ed03u64.wrapping_add(n as u64 * 0x9e37_79b9_7f4a_7c15);
    let a = unsafe { arr64(n) };
    let bit = unsafe { bit64(n) };
    bit.fill(O::ID);
    for i in 0..n {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<O>(bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let x = brute_prefix::<O>(a, p);
        let y = bit_prefix::<O>(bit, p);
        assert_eq!(x, y, "verify query n={n} i={i} p={p}");
    }
}

fn verify_mixed<O: Semi>(n: usize) {
    let mut seed = 0xd1b5_4a32_d192_ed03u64
        .wrapping_add(n as u64 * 0x9e37_79b9_7f4a_7c15)
        .wrapping_add(1);
    let a = unsafe { arr64(n) };
    let bit = unsafe { bit64(n) };
    bit.fill(O::ID);
    for i in 0..n {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<O>(bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let d = rng_next(&mut seed) % 1024;
        if i & 1 != 0 {
            brute_update::<O>(a, p, d);
            bit_apply::<O>(bit, n, p, d);
        } else {
            let x = brute_prefix::<O>(a, p);
            let y = bit_prefix::<O>(bit, p);
            assert_eq!(x, y, "verify mixed n={n} i={i} p={p}");
        }
    }
}

fn verify_range<O: Semi>(n: usize) {
    for len in [1usize, 2, 3, 7, 16, 64, 257, 1024] {
        let mut seed = 0xa076_1d64_78bd_642fu64.wrapping_add(len as u64 * 0x9e37_79b9_7f4a_7c15);
        let a = unsafe { arr64(n) };
        let bit = unsafe { bit64(n) };
        bit.fill(O::ID);
        for i in 0..n {
            let v = rng_next(&mut seed) % 1024;
            a[i] = v;
            bit_apply::<O>(bit, n, i, v);
        }
        for i in 0..500usize {
            let l = ((rng_next(&mut seed) >> 32) as usize) % (n - len + 1);
            let mut x = O::ID;
            for &v in unsafe { a.get_unchecked(l..l + len) } {
                x = O::op(x, v);
            }
            let hi = bit_prefix::<O>(bit, l + len - 1);
            let lo = bit_prefix::<O>(bit, l.wrapping_sub(1));
            assert_eq!(x, O::combine_range(hi, lo), "verify range len={len} i={i} l={l}");
        }
    }
}

fn verify_bisect(n: usize, mixed: bool) {
    let mut seed = 0x1234_5678_9abc_def0u64.wrapping_add(if mixed { 1 } else { 0 });
    let a = unsafe { arr64(n) };
    let bit = unsafe { bit64(n) };
    bit.fill(0);
    for i in 0..n {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<Sum>(bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let d = rng_next(&mut seed) % 1024;
        if mixed && i & 1 != 0 {
            let slot = unsafe { a.get_unchecked_mut(p) };
            *slot = slot.wrapping_add(d);
            bit_apply::<Sum>(bit, n, p, d);
        } else {
            let total: u64 = a.iter().fold(0u64, |acc, &v| acc.wrapping_add(v));
            let k = if total == 0 { 1 } else { rng_next(&mut seed) % total + 1 };
            let x = brute_bisect(a, n, k);
            let y = bit_bisect(bit, n, top_of(n), k);
            assert_eq!(x, y, "verify bisect mixed={mixed} n={n} i={i} k={k}");
        }
    }
}

fn verify_query_u32(n: usize) {
    let mut seed = 0x1357_9bdf_2468_ace0u64.wrapping_add(n as u64);
    let a = unsafe { arr32(n) };
    let bit = unsafe { bit32(n) };
    bit.fill(0);
    for i in 0..n {
        let v = (rng_next(&mut seed) % 1024) as u32;
        a[i] = v;
        bit_apply_u32(bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let x = brute_prefix_u32(a, p);
        let y = bit_prefix_u32(bit, p);
        assert_eq!(x, y, "verify sum32 query n={n} i={i} p={p}");
    }
}

fn verify_mixed_u32(n: usize) {
    let mut seed = 0x1357_9bdf_2468_ace0u64.wrapping_add(n as u64 + 0x1000);
    let a = unsafe { arr32(n) };
    let bit = unsafe { bit32(n) };
    bit.fill(0);
    for i in 0..n {
        let v = (rng_next(&mut seed) % 1024) as u32;
        a[i] = v;
        bit_apply_u32(bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let d = (rng_next(&mut seed) % 1024) as u32;
        if i & 1 != 0 {
            let slot = unsafe { a.get_unchecked_mut(p) };
            *slot = slot.wrapping_add(d);
            bit_apply_u32(bit, n, p, d);
        } else {
            let x = brute_prefix_u32(a, p);
            let y = bit_prefix_u32(bit, p);
            assert_eq!(x, y, "verify sum32 mixed n={n} i={i} p={p}");
        }
    }
}

fn verify_range_u32(n: usize) {
    for len in [1usize, 2, 3, 7, 16, 64, 257, 1024] {
        let mut seed = 0x1357_9bdf_2468_ace0u64.wrapping_add(n as u64 + len as u64);
        let a = unsafe { arr32(n) };
        let bit = unsafe { bit32(n) };
        bit.fill(0);
        for i in 0..n {
            let v = (rng_next(&mut seed) % 1024) as u32;
            a[i] = v;
            bit_apply_u32(bit, n, i, v);
        }
        for i in 0..500usize {
            let l = ((rng_next(&mut seed) >> 32) as usize) % (n - len + 1);
            let x = unsafe { a.get_unchecked(l..l + len) }
                .iter()
                .fold(0u32, |acc, &v| acc.wrapping_add(v));
            let hi = bit_prefix_u32(bit, l + len - 1);
            let lo = bit_prefix_u32(bit, l.wrapping_sub(1));
            assert_eq!(x, hi.wrapping_sub(lo), "verify sum32 range len={len} i={i} l={l}");
        }
    }
}

// ---------------- measurement ----------------
const ALL_N: [usize; 45] = [
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 96,
    128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144,
    393216, 524288, 786432, 1048576,
];

const ALL_L: [usize; 32] = [
    2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 80, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192,
];

fn measure_query<O: Semi>(name: &str, mix: Mix, mode: &str) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for &n in &ALL_N {
        verify_query::<O>(n);
        let a = unsafe { arr64(n) };
        let bit = unsafe { bit64(n) };
        bit.fill(O::ID);
        for i in 0..n {
            let v = rng_next(&mut seed) % 1024;
            a[i] = v;
            bit_apply::<O>(bit, n, i, v);
        }

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let pos = unsafe { pos(q0_brute) };
            let delta = unsafe { delta(q0_brute) };
            let isq = unsafe { isq(q0_brute) };
            gen_ops(pos, delta, isq, n, mix, &mut seed);
            let ns = time_query_brute::<O>(a, pos);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let pos = unsafe { pos(q0_bit) };
            let delta = unsafe { delta(q0_bit) };
            let isq = unsafe { isq(q0_bit) };
            gen_ops(pos, delta, isq, n, mix, &mut seed);
            let ns = time_query_bit::<O>(bit, pos);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let pos_b = unsafe { pos(q_brute) };
        let pos_t = unsafe { pos2(q_bit) };
        let del_b = unsafe { delta(q_brute) };
        let del_t = unsafe { delta2(q_bit) };
        let isb = unsafe { isq(q_brute) };
        let ist = unsafe { isq2(q_bit) };
        gen_ops(pos_b, del_b, isb, n, mix, &mut seed);
        gen_ops(pos_t, del_t, ist, n, mix, &mut seed);

        let mut s_brute = [0f64; ROUNDS];
        let mut s_bit = [0f64; ROUNDS];
        for r in 0..ROUNDS {
            s_brute[r] = time_query_brute::<O>(a, pos_b);
            s_bit[r] = time_query_bit::<O>(bit, pos_t);
        }

        let t_brute = median(&s_brute) / q_brute as f64;
        let t_bit = median(&s_bit) / q_bit as f64;
        print_row(name, mode, n, t_brute, t_bit);
    }
}

fn measure_mixed<O: Semi>(name: &str, mix: Mix, mode: &str) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64.wrapping_add(2);
    for &n in &ALL_N {
        verify_mixed::<O>(n);
        let a = unsafe { arr64(n) };
        let bit = unsafe { bit64(n) };
        bit.fill(O::ID);

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let pos = unsafe { pos(q0_brute) };
            let delta = unsafe { delta(q0_brute) };
            let isq = unsafe { isq(q0_brute) };
            gen_ops(pos, delta, isq, n, mix, &mut seed);
            let ns = time_mixed_brute::<O>(a, pos, delta, isq);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let pos = unsafe { pos(q0_bit) };
            let delta = unsafe { delta(q0_bit) };
            let isq = unsafe { isq(q0_bit) };
            gen_ops(pos, delta, isq, n, mix, &mut seed);
            let ns = time_mixed_bit::<O>(bit, n, pos, delta, isq);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let pos_b = unsafe { pos(q_brute) };
        let pos_t = unsafe { pos2(q_bit) };
        let del_b = unsafe { delta(q_brute) };
        let del_t = unsafe { delta2(q_bit) };
        let isb = unsafe { isq(q_brute) };
        let ist = unsafe { isq2(q_bit) };
        gen_ops(pos_b, del_b, isb, n, mix, &mut seed);
        gen_ops(pos_t, del_t, ist, n, mix, &mut seed);

        let mut s_brute = [0f64; ROUNDS];
        let mut s_bit = [0f64; ROUNDS];
        for r in 0..ROUNDS {
            s_brute[r] = time_mixed_brute::<O>(a, pos_b, del_b, isb);
            s_bit[r] = time_mixed_bit::<O>(bit, n, pos_t, del_t, ist);
        }

        let t_brute = median(&s_brute) / q_brute as f64;
        let t_bit = median(&s_bit) / q_bit as f64;
        print_row(name, mode, n, t_brute, t_bit);
    }
}

fn measure_range<O: Semi>(name: &str) {
    const N: usize = MAX_N;
    verify_range::<O>(N);
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let a = unsafe { arr64(N) };
    let bit = unsafe { bit64(N) };
    bit.fill(O::ID);
    for i in 0..N {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<O>(bit, N, i, v);
    }

    for &len in &ALL_L {
        let q0_brute = (4_000_000usize / len).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let ls = unsafe { ls(q0_brute) };
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_brute::<O>(a, ls, len);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let ls = unsafe { ls(q0_bit) };
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_bit::<O>(bit, ls, len);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let ls_b = unsafe { ls(q_brute) };
        let ls_t = unsafe { ls2(q_bit) };
        for l in ls_b.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }
        for l in ls_t.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }

        let mut s_brute = [0f64; ROUNDS];
        let mut s_bit = [0f64; ROUNDS];
        for r in 0..ROUNDS {
            s_brute[r] = time_range_brute::<O>(a, ls_b, len);
            s_bit[r] = time_range_bit::<O>(bit, ls_t, len);
        }

        let t_brute = median(&s_brute) / q_brute as f64;
        let t_bit = median(&s_bit) / q_bit as f64;
        print_row(name, "range", len, t_brute, t_bit);
    }
}

fn measure_bisect(mixed: bool) {
    let mode = if mixed { "mixed" } else { "query" };
    let mut seed = 0x9e37_79b9_7f4a_7c15u64.wrapping_add(if mixed { 3 } else { 4 } * 0x9e37_79b9_7f4a_7c15);
    for &n in &ALL_N {
        verify_bisect(n, mixed);
        let top = top_of(n);
        let a0 = unsafe { a064(n) };
        let bit0 = unsafe { bit064(n) };
        bit0.fill(0);
        for i in 0..n {
            let v = rng_next(&mut seed) % 1024;
            a0[i] = v;
            bit_apply::<Sum>(bit0, n, i, v);
        }
        let a = unsafe { arr64(n) };
        let bit = unsafe { bit64(n) };
        a.copy_from_slice(&*a0);
        bit.copy_from_slice(&*bit0);

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let pos = unsafe { pos(q0_brute) };
            let delta = unsafe { delta(q0_brute) };
            let isq = unsafe { isq(q0_brute) };
            let ks = unsafe { ks(q0_brute) };
            gen_bisect_ops(pos, delta, isq, ks, n, if mixed { Mix::Mixed50 } else { Mix::Query }, &mut seed);
            if mixed {
                a.copy_from_slice(&*a0);
            }
            let ns = if mixed {
                time_bisect_mixed_brute(a, n, pos, delta, isq, ks)
            } else {
                time_bisect_query_brute(a, n, ks)
            };
            q_brute = clamp_q(ns / q0_brute as f64);
            if mixed {
                bit.copy_from_slice(&*bit0);
            }
        }
        {
            let pos = unsafe { pos(q0_bit) };
            let delta = unsafe { delta(q0_bit) };
            let isq = unsafe { isq(q0_bit) };
            let ks = unsafe { ks(q0_bit) };
            gen_bisect_ops(pos, delta, isq, ks, n, if mixed { Mix::Mixed50 } else { Mix::Query }, &mut seed);
            if mixed {
                bit.copy_from_slice(&*bit0);
            }
            let ns = if mixed {
                time_bisect_mixed_bit(bit, n, top, pos, delta, isq, ks)
            } else {
                time_bisect_query_bit(bit, n, top, ks)
            };
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let pos_b = unsafe { pos(q_brute) };
        let pos_t = unsafe { pos2(q_bit) };
        let del_b = unsafe { delta(q_brute) };
        let del_t = unsafe { delta2(q_bit) };
        let isb = unsafe { isq(q_brute) };
        let ist = unsafe { isq2(q_bit) };
        let ks_b = unsafe { ks(q_brute) };
        let ks_t = unsafe { ks2(q_bit) };
        gen_bisect_ops(pos_b, del_b, isb, ks_b, n, if mixed { Mix::Mixed50 } else { Mix::Query }, &mut seed);
        gen_bisect_ops(pos_t, del_t, ist, ks_t, n, if mixed { Mix::Mixed50 } else { Mix::Query }, &mut seed);

        let mut s_brute = [0f64; ROUNDS];
        let mut s_bit = [0f64; ROUNDS];
        for r in 0..ROUNDS {
            if mixed {
                a.copy_from_slice(&*a0);
                bit.copy_from_slice(&*bit0);
            }
            s_brute[r] = if mixed {
                time_bisect_mixed_brute(a, n, pos_b, del_b, isb, ks_b)
            } else {
                time_bisect_query_brute(a, n, ks_b)
            };
            if mixed {
                bit.copy_from_slice(&*bit0);
            }
            s_bit[r] = if mixed {
                time_bisect_mixed_bit(bit, n, top, pos_t, del_t, ist, ks_t)
            } else {
                time_bisect_query_bit(bit, n, top, ks_t)
            };
        }

        let t_brute = median(&s_brute) / q_brute as f64;
        let t_bit = median(&s_bit) / q_bit as f64;
        print_row("sum_bisect", mode, n, t_brute, t_bit);
    }
}

fn measure_u32_query() {
    let mut seed = 0x5a5a_5a5a_5a5a_5a5au64;
    for &n in &ALL_N {
        verify_query_u32(n);
        let a = unsafe { arr32(n) };
        let bit = unsafe { bit32(n) };
        bit.fill(0);
        for i in 0..n {
            let v = (rng_next(&mut seed) % 1024) as u32;
            a[i] = v;
            bit_apply_u32(bit, n, i, v);
        }

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let pos = unsafe { pos(q0_brute) };
            let delta = unsafe { delta(q0_brute) };
            let isq = unsafe { isq(q0_brute) };
            gen_ops(pos, delta, isq, n, Mix::Query, &mut seed);
            let ns = time_query_brute_u32(a, pos);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let pos = unsafe { pos(q0_bit) };
            let delta = unsafe { delta(q0_bit) };
            let isq = unsafe { isq(q0_bit) };
            gen_ops(pos, delta, isq, n, Mix::Query, &mut seed);
            let ns = time_query_bit_u32(bit, pos);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let pos_b = unsafe { pos(q_brute) };
        let pos_t = unsafe { pos2(q_bit) };
        let del_b = unsafe { delta(q_brute) };
        let del_t = unsafe { delta2(q_bit) };
        let isb = unsafe { isq(q_brute) };
        let ist = unsafe { isq2(q_bit) };
        gen_ops(pos_b, del_b, isb, n, Mix::Query, &mut seed);
        gen_ops(pos_t, del_t, ist, n, Mix::Query, &mut seed);

        let mut s_brute = [0f64; ROUNDS];
        let mut s_bit = [0f64; ROUNDS];
        for r in 0..ROUNDS {
            s_brute[r] = time_query_brute_u32(a, pos_b);
            s_bit[r] = time_query_bit_u32(bit, pos_t);
        }

        let t_brute = median(&s_brute) / q_brute as f64;
        let t_bit = median(&s_bit) / q_bit as f64;
        print_row("sum32", "query", n, t_brute, t_bit);
    }
}

fn measure_u32_mixed() {
    let mut seed = 0x5a5a_5a5a_5a5a_5a5au64.wrapping_add(0x1000);
    for &n in &ALL_N {
        verify_mixed_u32(n);
        let a = unsafe { arr32(n) };
        let bit = unsafe { bit32(n) };
        bit.fill(0);

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let pos = unsafe { pos(q0_brute) };
            let delta = unsafe { delta(q0_brute) };
            let isq = unsafe { isq(q0_brute) };
            gen_ops(pos, delta, isq, n, Mix::Mixed50, &mut seed);
            let ns = time_mixed_brute_u32(a, pos, delta, isq);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let pos = unsafe { pos(q0_bit) };
            let delta = unsafe { delta(q0_bit) };
            let isq = unsafe { isq(q0_bit) };
            gen_ops(pos, delta, isq, n, Mix::Mixed50, &mut seed);
            let ns = time_mixed_bit_u32(bit, n, pos, delta, isq);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let pos_b = unsafe { pos(q_brute) };
        let pos_t = unsafe { pos2(q_bit) };
        let del_b = unsafe { delta(q_brute) };
        let del_t = unsafe { delta2(q_bit) };
        let isb = unsafe { isq(q_brute) };
        let ist = unsafe { isq2(q_bit) };
        gen_ops(pos_b, del_b, isb, n, Mix::Mixed50, &mut seed);
        gen_ops(pos_t, del_t, ist, n, Mix::Mixed50, &mut seed);

        let mut s_brute = [0f64; ROUNDS];
        let mut s_bit = [0f64; ROUNDS];
        for r in 0..ROUNDS {
            s_brute[r] = time_mixed_brute_u32(a, pos_b, del_b, isb);
            s_bit[r] = time_mixed_bit_u32(bit, n, pos_t, del_t, ist);
        }

        let t_brute = median(&s_brute) / q_brute as f64;
        let t_bit = median(&s_bit) / q_bit as f64;
        print_row("sum32", "mixed", n, t_brute, t_bit);
    }
}

fn measure_u32_range() {
    const N: usize = MAX_N;
    verify_range_u32(N);
    let mut seed = 0x5a5a_5a5a_5a5a_5a5au64.wrapping_add(0x2000);
    let a = unsafe { arr32(N) };
    let bit = unsafe { bit32(N) };
    bit.fill(0);
    for i in 0..N {
        let v = (rng_next(&mut seed) % 1024) as u32;
        a[i] = v;
        bit_apply_u32(bit, N, i, v);
    }

    for &len in &ALL_L {
        let q0_brute = (4_000_000usize / len).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let ls = unsafe { ls(q0_brute) };
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_brute_u32(a, ls, len);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let ls = unsafe { ls(q0_bit) };
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_bit_u32(bit, ls, len);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let ls_b = unsafe { ls(q_brute) };
        let ls_t = unsafe { ls2(q_bit) };
        for l in ls_b.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }
        for l in ls_t.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }

        let mut s_brute = [0f64; ROUNDS];
        let mut s_bit = [0f64; ROUNDS];
        for r in 0..ROUNDS {
            s_brute[r] = time_range_brute_u32(a, ls_b, len);
            s_bit[r] = time_range_bit_u32(bit, ls_t, len);
        }

        let t_brute = median(&s_brute) / q_brute as f64;
        let t_bit = median(&s_bit) / q_bit as f64;
        print_row("sum32", "range", len, t_brute, t_bit);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    measure_query::<Sum>("sum", Mix::Query, "query");
    measure_query::<Min>("min", Mix::Query, "query");
    measure_query::<Max>("max", Mix::Query, "query");
    measure_query::<And>("and", Mix::Query, "query");
    measure_query::<Or>("or", Mix::Query, "query");
    measure_query::<Xor>("xor", Mix::Query, "query");
    measure_mixed::<Sum>("sum", Mix::Mixed50, "mixed");
    measure_mixed::<Xor>("xor", Mix::Mixed50, "mixed");
    measure_range::<Sum>("sum");
    measure_range::<Xor>("xor");
    measure_bisect(true);
    measure_bisect(false);
    measure_mixed::<Sum>("sum", Mix::Mixed25, "mixed_25");
    measure_mixed::<Sum>("sum", Mix::Mixed75, "mixed_75");
    measure_query::<Sum>("sum", Mix::QueryTail, "query_tail");
    measure_u32_query();
    measure_u32_mixed();
    measure_u32_range();
    exit(0);
}
