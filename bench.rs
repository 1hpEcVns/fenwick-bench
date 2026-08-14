// Fenwick tree (BIT) vs plain-array brute force across semigroup operations.
// ops: sum, min, max, and, or, xor.
// modes: mixed (1:1 point-update + prefix-query, invertible ops only),
//        query (prefix queries only), range (range sums, N=2^20, invertible ops).
// Build: rustc --edition=2024 -O -C target-cpu=native bench.rs -o bench_rs
use std::hint::black_box;
use std::time::Instant;

const ROUNDS: usize = 9;
const TARGET_NS: f64 = 3e6; // ~3 ms per timed pass
const Q_MIN: usize = 256;
const Q_MAX: usize = 4_000_000;

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

fn rng_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// ---------------- brute force: plain array ----------------
#[inline(always)]
fn brute_prefix<O: Semi>(a: &[u64], i: usize) -> u64 {
    let mut s = O::ID;
    // i < a.len() is guaranteed by every caller (pos = rng % n).
    for &v in unsafe { a.get_unchecked(..=i) } {
        s = O::op(s, v);
    }
    s
}

#[inline(always)]
fn brute_update<O: Semi>(a: &mut [u64], i: usize, d: u64) {
    // Only instantiated for invertible ops (sum/xor).
    if O::invertible() {
        let slot = unsafe { a.get_unchecked_mut(i) }; // i < a.len()
        *slot = O::op(*slot, d);
    }
}

// ---------------- Fenwick tree (BIT), tree[1..n] ----------------
#[inline(always)]
fn bit_prefix<O: Semi>(bit: &[u64], i: usize) -> u64 {
    // i = usize::MAX means "before 0": i+1 wraps to 0 and the loop is skipped.
    let mut s = O::ID;
    let mut k = i.wrapping_add(1);
    while k != 0 {
        // k <= n and bit.len() == n + 1, so the index is always in bounds.
        s = O::op(s, *unsafe { bit.get_unchecked(k) });
        k &= k.wrapping_sub(1);
    }
    s
}

#[inline(always)]
fn bit_apply<O: Semi>(bit: &mut [u64], n: usize, i: usize, d: u64) {
    // Merge one element into the tree (build, or invertible point update).
    let mut k = i.wrapping_add(1);
    while k <= n {
        let slot = unsafe { bit.get_unchecked_mut(k) }; // k in [1, n]
        *slot = O::op(*slot, d);
        k = k.wrapping_add(k & k.wrapping_neg());
    }
}

// ---------------- timed kernels ----------------
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
        // l + len <= a.len() is guaranteed by the caller.
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

// ---------------- BIT binary search (树状数组上二分), sum only ----------------
fn top_of(n: usize) -> usize {
    1usize << (usize::BITS - 1 - n.leading_zeros())
}

// First prefix that reaches k: number of elements consumed (0..n).
#[inline(always)]
fn brute_bisect(a: &[u64], n: usize, k: u64) -> usize {
    let mut s = 0u64;
    let mut j = 0usize;
    while j < n && s < k {
        s = s.wrapping_add(*unsafe { a.get_unchecked(j) }); // j < n <= a.len()
        j += 1;
    }
    j
}

// Same semantics via binary lifting on the BIT (values must be non-negative).
#[inline(always)]
fn bit_bisect(bit: &[u64], n: usize, top: usize, k: u64) -> usize {
    let mut pos = 0usize;
    let mut m = top;
    let mut kk = k;
    while m != 0 {
        let next = pos + m;
        if next <= n {
            let v = *unsafe { bit.get_unchecked(next) }; // next in [1, n]
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

#[inline(never)]
fn run_bisect_mixed_brute(a: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8], ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        if is_q[i] != 0 {
            acc = acc.wrapping_add(brute_bisect(a, n, ks[i]) as u64);
        } else {
            let p = pos[i]; // p < n
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

fn time_mixed_brute<O: Semi>(a: &mut [u64], pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = Instant::now();
    let acc = run_mixed_brute::<O>(a, pos, delta, is_q);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_mixed_bit<O: Semi>(bit: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = Instant::now();
    let acc = run_mixed_bit::<O>(bit, n, pos, delta, is_q);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_query_brute<O: Semi>(a: &[u64], pos: &[usize]) -> f64 {
    let t0 = Instant::now();
    let acc = run_query_brute::<O>(a, pos);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_query_bit<O: Semi>(bit: &[u64], pos: &[usize]) -> f64 {
    let t0 = Instant::now();
    let acc = run_query_bit::<O>(bit, pos);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_range_brute<O: Semi>(a: &[u64], ls: &[usize], len: usize) -> f64 {
    let t0 = Instant::now();
    let acc = run_range_brute::<O>(a, ls, len);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_range_bit<O: Semi>(bit: &[u64], ls: &[usize], len: usize) -> f64 {
    let t0 = Instant::now();
    let acc = run_range_bit::<O>(bit, ls, len);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_bisect_mixed_brute(a: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8], ks: &[u64]) -> f64 {
    let t0 = Instant::now();
    let acc = run_bisect_mixed_brute(a, n, pos, delta, is_q, ks);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_bisect_mixed_bit(bit: &mut [u64], n: usize, top: usize, pos: &[usize], delta: &[u64], is_q: &[u8], ks: &[u64]) -> f64 {
    let t0 = Instant::now();
    let acc = run_bisect_mixed_bit(bit, n, top, pos, delta, is_q, ks);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_bisect_query_brute(a: &[u64], n: usize, ks: &[u64]) -> f64 {
    let t0 = Instant::now();
    let acc = run_bisect_query_brute(a, n, ks);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_bisect_query_bit(bit: &[u64], n: usize, top: usize, ks: &[u64]) -> f64 {
    let t0 = Instant::now();
    let acc = run_bisect_query_bit(bit, n, top, ks);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn clamp_q(per_ns: f64) -> usize {
    ((TARGET_NS / per_ns) as usize).clamp(Q_MIN, Q_MAX)
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn gen_ops(n: usize, q: usize, mix: Mix, seed: &mut u64) -> (Vec<usize>, Vec<u64>, Vec<u8>) {
    let mut pos = Vec::with_capacity(q);
    let mut delta = Vec::with_capacity(q);
    let mut is_q = Vec::with_capacity(q);
    for i in 0..q {
        pos.push(gen_pos(mix, n, seed));
        delta.push(rng_next(seed) % 1024);
        is_q.push(if is_query_op(mix, i) { 1 } else { 0 });
    }
    (pos, delta, is_q)
}

fn gen_bisect_ops(n: usize, q: usize, mixed: bool, seed: &mut u64) -> (Vec<usize>, Vec<u64>, Vec<u8>, Vec<u64>) {
    let mut pos = Vec::with_capacity(q);
    let mut delta = Vec::with_capacity(q);
    let mut is_q = Vec::with_capacity(q);
    let mut ks = Vec::with_capacity(q);
    for i in 0..q {
        pos.push(((rng_next(seed) >> 32) as usize) % n);
        delta.push(rng_next(seed) % 1024);
        let qq = !mixed || i & 1 == 0;
        is_q.push(if qq { 1 } else { 0 });
        ks.push(if qq { rng_next(seed) % (n as u64 * 512) + 1 } else { 0 });
    }
    (pos, delta, is_q, ks)
}

// ---------------- verification ----------------
fn verify_query<O: Semi>(n: usize) {
    let mut seed = 0xd1b5_4a32_d192_ed03u64.wrapping_add(n as u64 * 0x9e37_79b9_7f4a_7c15);
    let mut a = vec![0u64; n];
    let mut bit = vec![O::ID; n + 1];
    for i in 0..n {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<O>(&mut bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let x = brute_prefix::<O>(&a, p);
        let y = bit_prefix::<O>(&bit, p);
        assert_eq!(x, y, "verify query n={n} i={i} p={p}");
    }
}

fn verify_mixed<O: Semi>(n: usize) {
    let mut seed = 0xd1b5_4a32_d192_ed03u64
        .wrapping_add(n as u64 * 0x9e37_79b9_7f4a_7c15)
        .wrapping_add(1);
    let mut a = vec![0u64; n];
    let mut bit = vec![O::ID; n + 1];
    for i in 0..n {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<O>(&mut bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let d = rng_next(&mut seed) % 1024;
        if i & 1 != 0 {
            brute_update::<O>(&mut a, p, d);
            bit_apply::<O>(&mut bit, n, p, d);
        } else {
            let x = brute_prefix::<O>(&a, p);
            let y = bit_prefix::<O>(&bit, p);
            assert_eq!(x, y, "verify mixed n={n} i={i} p={p}");
        }
    }
}

fn verify_range<O: Semi>(n: usize) {
    for len in [1usize, 2, 3, 7, 16, 64, 257, 1024] {
        let mut seed = 0xa076_1d64_78bd_642fu64
            .wrapping_add(len as u64 * 0x9e37_79b9_7f4a_7c15);
        let mut a = vec![0u64; n];
        let mut bit = vec![O::ID; n + 1];
        for i in 0..n {
            let v = rng_next(&mut seed) % 1024;
            a[i] = v;
            bit_apply::<O>(&mut bit, n, i, v);
        }
        for i in 0..500usize {
            let l = ((rng_next(&mut seed) >> 32) as usize) % (n - len + 1);
            let mut x = O::ID;
            for &v in &a[l..l + len] {
                x = O::op(x, v);
            }
            let hi = bit_prefix::<O>(&bit, l + len - 1);
            let lo = bit_prefix::<O>(&bit, l.wrapping_sub(1));
            let y = O::combine_range(hi, lo);
            assert_eq!(x, y, "verify range len={len} i={i} l={l}");
        }
    }
}

fn verify_bisect(n: usize, mixed: bool) {
    let mut seed = 0x1234_5678_9abc_def0u64.wrapping_add(if mixed { 1 } else { 0 });
    let mut a = vec![0u64; n];
    let mut bit = vec![0u64; n + 1];
    for i in 0..n {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<Sum>(&mut bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let d = rng_next(&mut seed) % 1024;
        if mixed && i & 1 != 0 {
            a[p] = a[p].wrapping_add(d);
            bit_apply::<Sum>(&mut bit, n, p, d);
        } else {
            let total: u64 = a.iter().fold(0u64, |acc, &v| acc.wrapping_add(v));
            let k = if total == 0 { 1 } else { rng_next(&mut seed) % total + 1 };
            let x = brute_bisect(&a, n, k);
            let y = bit_bisect(&bit, n, top_of(n), k);
            assert_eq!(x, y, "verify bisect mixed={mixed} n={n} i={i} k={k}");
        }
    }
}

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

// ---------------- measurement ----------------
fn measure_query<O: Semi>(name: &str, mix: Mix, mode: &str) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for &n in &ALL_N {
        verify_query::<O>(n);
        let mut a = vec![0u64; n];
        let mut bit = vec![O::ID; n + 1];
        for i in 0..n {
            let v = rng_next(&mut seed) % 1024;
            a[i] = v;
            bit_apply::<O>(&mut bit, n, i, v);
        }

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let (pos, _, _) = gen_ops(n, q0_brute, mix, &mut seed);
            let ns = time_query_brute::<O>(&a, &pos);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let (pos, _, _) = gen_ops(n, q0_bit, mix, &mut seed);
            let ns = time_query_bit::<O>(&bit, &pos);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let (pos_b, _, _) = gen_ops(n, q_brute, mix, &mut seed);
        let (pos_t, _, _) = gen_ops(n, q_bit, mix, &mut seed);

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(time_query_brute::<O>(&a, &pos_b));
            s_bit.push(time_query_bit::<O>(&bit, &pos_t));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("{name},{mode},{n},{t_brute:.3},{t_bit:.3}");
    }
}

fn measure_mixed<O: Semi>(name: &str, mix: Mix, mode: &str) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64.wrapping_add(2);
    for &n in &ALL_N {
        verify_mixed::<O>(n);
        let mut a = vec![0u64; n];
        let mut bit = vec![O::ID; n + 1];

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let (pos, delta, is_q) = gen_ops(n, q0_brute, mix, &mut seed);
            let ns = time_mixed_brute::<O>(&mut a, &pos, &delta, &is_q);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let (pos, delta, is_q) = gen_ops(n, q0_bit, mix, &mut seed);
            let ns = time_mixed_bit::<O>(&mut bit, n, &pos, &delta, &is_q);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let (pos_b, del_b, isb) = gen_ops(n, q_brute, mix, &mut seed);
        let (pos_t, del_t, ist) = gen_ops(n, q_bit, mix, &mut seed);

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(time_mixed_brute::<O>(&mut a, &pos_b, &del_b, &isb));
            s_bit.push(time_mixed_bit::<O>(&mut bit, n, &pos_t, &del_t, &ist));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("{name},{mode},{n},{t_brute:.3},{t_bit:.3}");
    }
}

fn measure_range<O: Semi>(name: &str) {
    const N: usize = 1 << 20;
    verify_range::<O>(N);
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let mut a = vec![0u64; N];
    let mut bit = vec![O::ID; N + 1];
    for i in 0..N {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_apply::<O>(&mut bit, N, i, v);
    }

    for &len in &ALL_L {
        let q0_brute = (4_000_000usize / len).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let mut ls = vec![0usize; q0_brute];
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_brute::<O>(&a, &ls, len);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let mut ls = vec![0usize; q0_bit];
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_bit::<O>(&bit, &ls, len);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let mut ls_b = vec![0usize; q_brute];
        let mut ls_t = vec![0usize; q_bit];
        for l in ls_b.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }
        for l in ls_t.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(time_range_brute::<O>(&a, &ls_b, len));
            s_bit.push(time_range_bit::<O>(&bit, &ls_t, len));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("{name},range,{len},{t_brute:.3},{t_bit:.3}");
    }
}

fn measure_bisect(mixed: bool) {
    let mode = if mixed { "mixed" } else { "query" };
    let mut seed = 0x9e37_79b9_7f4a_7c15u64.wrapping_add(if mixed { 3 } else { 4 } * 0x9e37_79b9_7f4a_7c15);
    for &n in &ALL_N {
        verify_bisect(n, mixed);
        let top = top_of(n);
        let mut a0 = vec![0u64; n];
        let mut bit0 = vec![0u64; n + 1];
        for i in 0..n {
            let v = rng_next(&mut seed) % 1024;
            a0[i] = v;
            bit_apply::<Sum>(&mut bit0, n, i, v);
        }
        let mut a = a0.clone();
        let mut bit = bit0.clone();

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let (pos, delta, is_q, ks) = gen_bisect_ops(n, q0_brute, mixed, &mut seed);
            if mixed {
                a = a0.clone();
            }
            let ns = if mixed {
                time_bisect_mixed_brute(&mut a, n, &pos, &delta, &is_q, &ks)
            } else {
                time_bisect_query_brute(&a, n, &ks)
            };
            q_brute = clamp_q(ns / q0_brute as f64);
            if mixed {
                bit = bit0.clone();
            }
        }
        {
            let (pos, delta, is_q, ks) = gen_bisect_ops(n, q0_bit, mixed, &mut seed);
            if mixed {
                bit = bit0.clone();
            }
            let ns = if mixed {
                time_bisect_mixed_bit(&mut bit, n, top, &pos, &delta, &is_q, &ks)
            } else {
                time_bisect_query_bit(&bit, n, top, &ks)
            };
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let (pos_b, del_b, isb, ks_b) = gen_bisect_ops(n, q_brute, mixed, &mut seed);
        let (pos_t, del_t, ist, ks_t) = gen_bisect_ops(n, q_bit, mixed, &mut seed);

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            if mixed {
                a = a0.clone();
                bit = bit0.clone();
            }
            s_brute.push(if mixed {
                time_bisect_mixed_brute(&mut a, n, &pos_b, &del_b, &isb, &ks_b)
            } else {
                time_bisect_query_brute(&a, n, &ks_b)
            });
            if mixed {
                bit = bit0.clone();
            }
            s_bit.push(if mixed {
                time_bisect_mixed_bit(&mut bit, n, top, &pos_t, &del_t, &ist, &ks_t)
            } else {
                time_bisect_query_bit(&bit, n, top, &ks_t)
            });
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("sum_bisect,{mode},{n},{t_brute:.3},{t_bit:.3}");
    }
}

// ---------------- u32 element type (sum only) ----------------
#[inline(always)]
fn brute_prefix_u32(a: &[u32], i: usize) -> u32 {
    let mut s = 0u32;
    for &v in unsafe { a.get_unchecked(..=i) } { // i < a.len()
        s = s.wrapping_add(v);
    }
    s
}

#[inline(always)]
fn bit_prefix_u32(bit: &[u32], i: usize) -> u32 {
    let mut s = 0u32;
    let mut k = i.wrapping_add(1);
    while k != 0 {
        s = s.wrapping_add(*unsafe { bit.get_unchecked(k) }); // k <= n
        k &= k.wrapping_sub(1);
    }
    s
}

#[inline(always)]
fn bit_apply_u32(bit: &mut [u32], n: usize, i: usize, d: u32) {
    let mut k = i.wrapping_add(1);
    while k <= n {
        let slot = unsafe { bit.get_unchecked_mut(k) }; // k in [1, n]
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
            let slot = unsafe { a.get_unchecked_mut(p) }; // p < n
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
        for &v in unsafe { a.get_unchecked(l..l + len) } { // l + len <= a.len()
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

fn time_query_brute_u32(a: &[u32], pos: &[usize]) -> f64 {
    let t0 = Instant::now();
    let acc = run_query_brute_u32(a, pos);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_query_bit_u32(bit: &[u32], pos: &[usize]) -> f64 {
    let t0 = Instant::now();
    let acc = run_query_bit_u32(bit, pos);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_mixed_brute_u32(a: &mut [u32], pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = Instant::now();
    let acc = run_mixed_brute_u32(a, pos, delta, is_q);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_mixed_bit_u32(bit: &mut [u32], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = Instant::now();
    let acc = run_mixed_bit_u32(bit, n, pos, delta, is_q);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_range_brute_u32(a: &[u32], ls: &[usize], len: usize) -> f64 {
    let t0 = Instant::now();
    let acc = run_range_brute_u32(a, ls, len);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_range_bit_u32(bit: &[u32], ls: &[usize], len: usize) -> f64 {
    let t0 = Instant::now();
    let acc = run_range_bit_u32(bit, ls, len);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn verify_query_u32(n: usize) {
    let mut seed = 0x1357_9bdf_2468_ace0u64.wrapping_add(n as u64);
    let mut a = vec![0u32; n];
    let mut bit = vec![0u32; n + 1];
    for i in 0..n {
        let v = (rng_next(&mut seed) % 1024) as u32;
        a[i] = v;
        bit_apply_u32(&mut bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let x = brute_prefix_u32(&a, p);
        let y = bit_prefix_u32(&bit, p);
        assert_eq!(x, y, "verify sum32 query n={n} i={i} p={p}");
    }
}

fn verify_mixed_u32(n: usize) {
    let mut seed = 0x1357_9bdf_2468_ace0u64.wrapping_add(n as u64 + 0x1000);
    let mut a = vec![0u32; n];
    let mut bit = vec![0u32; n + 1];
    for i in 0..n {
        let v = (rng_next(&mut seed) % 1024) as u32;
        a[i] = v;
        bit_apply_u32(&mut bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let d = (rng_next(&mut seed) % 1024) as u32;
        if i & 1 != 0 {
            a[p] = a[p].wrapping_add(d);
            bit_apply_u32(&mut bit, n, p, d);
        } else {
            let x = brute_prefix_u32(&a, p);
            let y = bit_prefix_u32(&bit, p);
            assert_eq!(x, y, "verify sum32 mixed n={n} i={i} p={p}");
        }
    }
}

fn verify_range_u32(n: usize) {
    for len in [1usize, 2, 3, 7, 16, 64, 257, 1024] {
        let mut seed = 0x1357_9bdf_2468_ace0u64.wrapping_add(n as u64 + len as u64);
        let mut a = vec![0u32; n];
        let mut bit = vec![0u32; n + 1];
        for i in 0..n {
            let v = (rng_next(&mut seed) % 1024) as u32;
            a[i] = v;
            bit_apply_u32(&mut bit, n, i, v);
        }
        for i in 0..500usize {
            let l = ((rng_next(&mut seed) >> 32) as usize) % (n - len + 1);
            let x = a[l..l + len].iter().fold(0u32, |acc, &v| acc.wrapping_add(v));
            let hi = bit_prefix_u32(&bit, l + len - 1);
            let lo = bit_prefix_u32(&bit, l.wrapping_sub(1));
            assert_eq!(x, hi.wrapping_sub(lo), "verify sum32 range len={len} i={i} l={l}");
        }
    }
}

fn measure_u32_query() {
    let mut seed = 0x5a5a_5a5a_5a5a_5a5au64;
    for &n in &ALL_N {
        verify_query_u32(n);
        let mut a = vec![0u32; n];
        let mut bit = vec![0u32; n + 1];
        for i in 0..n {
            let v = (rng_next(&mut seed) % 1024) as u32;
            a[i] = v;
            bit_apply_u32(&mut bit, n, i, v);
        }

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let (pos, _, _) = gen_ops(n, q0_brute, Mix::Query, &mut seed);
            let ns = time_query_brute_u32(&a, &pos);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let (pos, _, _) = gen_ops(n, q0_bit, Mix::Query, &mut seed);
            let ns = time_query_bit_u32(&bit, &pos);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let (pos_b, _, _) = gen_ops(n, q_brute, Mix::Query, &mut seed);
        let (pos_t, _, _) = gen_ops(n, q_bit, Mix::Query, &mut seed);

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(time_query_brute_u32(&a, &pos_b));
            s_bit.push(time_query_bit_u32(&bit, &pos_t));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("sum32,query,{n},{t_brute:.3},{t_bit:.3}");
    }
}

fn measure_u32_mixed() {
    let mut seed = 0x5a5a_5a5a_5a5a_5a5au64.wrapping_add(0x1000);
    for &n in &ALL_N {
        verify_mixed_u32(n);
        let mut a = vec![0u32; n];
        let mut bit = vec![0u32; n + 1];

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let (pos, delta, is_q) = gen_ops(n, q0_brute, Mix::Mixed50, &mut seed);
            let ns = time_mixed_brute_u32(&mut a, &pos, &delta, &is_q);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let (pos, delta, is_q) = gen_ops(n, q0_bit, Mix::Mixed50, &mut seed);
            let ns = time_mixed_bit_u32(&mut bit, n, &pos, &delta, &is_q);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let (pos_b, del_b, isb) = gen_ops(n, q_brute, Mix::Mixed50, &mut seed);
        let (pos_t, del_t, ist) = gen_ops(n, q_bit, Mix::Mixed50, &mut seed);

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(time_mixed_brute_u32(&mut a, &pos_b, &del_b, &isb));
            s_bit.push(time_mixed_bit_u32(&mut bit, n, &pos_t, &del_t, &ist));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("sum32,mixed,{n},{t_brute:.3},{t_bit:.3}");
    }
}

fn measure_u32_range() {
    const N: usize = 1 << 20;
    verify_range_u32(N);
    let mut seed = 0x5a5a_5a5a_5a5a_5a5au64.wrapping_add(0x2000);
    let mut a = vec![0u32; N];
    let mut bit = vec![0u32; N + 1];
    for i in 0..N {
        let v = (rng_next(&mut seed) % 1024) as u32;
        a[i] = v;
        bit_apply_u32(&mut bit, N, i, v);
    }

    for &len in &ALL_L {
        let q0_brute = (4_000_000usize / len).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let q_brute;
        let q_bit;
        {
            let mut ls = vec![0usize; q0_brute];
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_brute_u32(&a, &ls, len);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let mut ls = vec![0usize; q0_bit];
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_bit_u32(&bit, &ls, len);
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let mut ls_b = vec![0usize; q_brute];
        let mut ls_t = vec![0usize; q_bit];
        for l in ls_b.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }
        for l in ls_t.iter_mut() {
            *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
        }

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(time_range_brute_u32(&a, &ls_b, len));
            s_bit.push(time_range_bit_u32(&bit, &ls_t, len));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("sum32,range,{len},{t_brute:.3},{t_bit:.3}");
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let all = which == "all";

    if all || which == "query" {
        measure_query::<Sum>("sum", Mix::Query, "query");
        measure_query::<Min>("min", Mix::Query, "query");
        measure_query::<Max>("max", Mix::Query, "query");
        measure_query::<And>("and", Mix::Query, "query");
        measure_query::<Or>("or", Mix::Query, "query");
        measure_query::<Xor>("xor", Mix::Query, "query");
    }
    if all || which == "mixed" {
        measure_mixed::<Sum>("sum", Mix::Mixed50, "mixed");
        measure_mixed::<Xor>("xor", Mix::Mixed50, "mixed");
    }
    if all || which == "range" {
        measure_range::<Sum>("sum");
        measure_range::<Xor>("xor");
    }
    if all || which == "bisect" {
        measure_bisect(true);
        measure_bisect(false);
    }
    if all || which == "fraction" {
        measure_mixed::<Sum>("sum", Mix::Mixed25, "mixed_25");
        measure_mixed::<Sum>("sum", Mix::Mixed75, "mixed_75");
    }
    if all || which == "tail" {
        measure_query::<Sum>("sum", Mix::QueryTail, "query_tail");
    }
    if all || which == "sum32" {
        measure_u32_query();
        measure_u32_mixed();
        measure_u32_range();
    }
    if which == "sum" {
        measure_query::<Sum>("sum", Mix::Query, "query");
        measure_mixed::<Sum>("sum", Mix::Mixed50, "mixed");
        measure_range::<Sum>("sum");
    }
    if which == "xor" {
        measure_query::<Xor>("xor", Mix::Query, "query");
        measure_mixed::<Xor>("xor", Mix::Mixed50, "mixed");
        measure_range::<Xor>("xor");
    }
    if which == "min" {
        measure_query::<Min>("min", Mix::Query, "query");
    }
    if which == "max" {
        measure_query::<Max>("max", Mix::Query, "query");
    }
    if which == "and" {
        measure_query::<And>("and", Mix::Query, "query");
    }
    if which == "or" {
        measure_query::<Or>("or", Mix::Query, "query");
    }
}
