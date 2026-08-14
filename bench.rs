// Fenwick tree (BIT) vs plain-array brute force: find the crossover size.
// Build: rustc --edition=2024 -O -C target-cpu=native bench.rs -o bench_rs
use std::hint::black_box;
use std::time::Instant;

const ROUNDS: usize = 9;
const TARGET_NS: f64 = 3e6; // ~3 ms per timed pass
const Q_MIN: usize = 256;
const Q_MAX: usize = 4_000_000;

fn rng_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// ---------------- brute force: plain array ----------------
#[inline(always)]
fn brute_prefix(a: &[u64], i: usize) -> u64 {
    let mut s = 0u64;
    for &v in &a[..=i] {
        s = s.wrapping_add(v);
    }
    s
}

#[inline(always)]
fn brute_update(a: &mut [u64], i: usize, d: u64) {
    a[i] = a[i].wrapping_add(d);
}

// ---------------- Fenwick tree (BIT), tree[1..n] ----------------
#[inline(always)]
fn bit_prefix(bit: &[u64], i: usize) -> u64 {
    // i = usize::MAX means "before 0": i+1 wraps to 0 and the loop is skipped.
    let mut s = 0u64;
    let mut k = i.wrapping_add(1);
    while k != 0 {
        s = s.wrapping_add(bit[k]);
        k &= k.wrapping_sub(1);
    }
    s
}

#[inline(always)]
fn bit_update(bit: &mut [u64], n: usize, i: usize, d: u64) {
    let mut k = i.wrapping_add(1);
    while k <= n {
        bit[k] = bit[k].wrapping_add(d);
        k = k.wrapping_add(k & k.wrapping_neg());
    }
}

// ---------------- timed kernels ----------------
#[inline(never)]
fn run_mixed_brute(a: &mut [u64], pos: &[usize], delta: &[u64], is_q: &[u8]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        let p = pos[i];
        if is_q[i] != 0 {
            acc = acc.wrapping_add(brute_prefix(a, p));
        } else {
            brute_update(a, p, delta[i]);
        }
    }
    acc
}

#[inline(never)]
fn run_mixed_bit(bit: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        let p = pos[i];
        if is_q[i] != 0 {
            acc = acc.wrapping_add(bit_prefix(bit, p));
        } else {
            bit_update(bit, n, p, delta[i]);
        }
    }
    acc
}

#[inline(never)]
fn run_query_brute(a: &[u64], pos: &[usize]) -> u64 {
    let mut acc = 0u64;
    for &p in pos {
        acc = acc.wrapping_add(brute_prefix(a, p));
    }
    acc
}

#[inline(never)]
fn run_query_bit(bit: &[u64], pos: &[usize]) -> u64 {
    let mut acc = 0u64;
    for &p in pos {
        acc = acc.wrapping_add(bit_prefix(bit, p));
    }
    acc
}

#[inline(never)]
fn run_range_brute(a: &[u64], ls: &[usize], len: usize) -> u64 {
    let mut acc = 0u64;
    for &l in ls {
        let mut s = 0u64;
        for &v in &a[l..l + len] {
            s = s.wrapping_add(v);
        }
        acc = acc.wrapping_add(s);
    }
    acc
}

#[inline(never)]
fn run_range_bit(bit: &[u64], ls: &[usize], len: usize) -> u64 {
    let mut acc = 0u64;
    for &l in ls {
        let hi = bit_prefix(bit, l.wrapping_add(len).wrapping_sub(1));
        let lo = bit_prefix(bit, l.wrapping_sub(1));
        acc = acc.wrapping_add(hi.wrapping_sub(lo));
    }
    acc
}

fn time_mixed_brute(a: &mut [u64], pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = Instant::now();
    let acc = run_mixed_brute(a, pos, delta, is_q);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_mixed_bit(bit: &mut [u64], n: usize, pos: &[usize], delta: &[u64], is_q: &[u8]) -> f64 {
    let t0 = Instant::now();
    let acc = run_mixed_bit(bit, n, pos, delta, is_q);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_query_brute(a: &[u64], pos: &[usize]) -> f64 {
    let t0 = Instant::now();
    let acc = run_query_brute(a, pos);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_query_bit(bit: &[u64], pos: &[usize]) -> f64 {
    let t0 = Instant::now();
    let acc = run_query_bit(bit, pos);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_range_brute(a: &[u64], ls: &[usize], len: usize) -> f64 {
    let t0 = Instant::now();
    let acc = run_range_brute(a, ls, len);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
}

fn time_range_bit(bit: &[u64], ls: &[usize], len: usize) -> f64 {
    let t0 = Instant::now();
    let acc = run_range_bit(bit, ls, len);
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

fn gen_ops(n: usize, q: usize, mixed: bool, seed: &mut u64) -> (Vec<usize>, Vec<u64>, Vec<u8>) {
    let mut pos = Vec::with_capacity(q);
    let mut delta = Vec::with_capacity(q);
    let mut is_q = Vec::with_capacity(q);
    for i in 0..q {
        pos.push(((rng_next(seed) >> 32) as usize) % n);
        delta.push(rng_next(seed) % 1024);
        is_q.push(if !mixed || i & 1 == 0 { 1 } else { 0 });
    }
    (pos, delta, is_q)
}

fn verify_n(n: usize, mixed: bool) {
    let mut seed = 0xd1b5_4a32_d192_ed03u64;
    let mut a = vec![0u64; n];
    let mut bit = vec![0u64; n + 1];
    for i in 0..n {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_update(&mut bit, n, i, v);
    }
    for i in 0..4000usize {
        let p = ((rng_next(&mut seed) >> 32) as usize) % n;
        let d = rng_next(&mut seed) % 1024;
        if mixed && i & 1 != 0 {
            brute_update(&mut a, p, d);
            bit_update(&mut bit, n, p, d);
        } else {
            let x = brute_prefix(&a, p);
            let y = bit_prefix(&bit, p);
            assert_eq!(x, y, "verify mixed={mixed} n={n} i={i} p={p}");
        }
    }
}

fn verify_range(n: usize) {
    for len in [1usize, 2, 3, 7, 16, 64, 257, 1024] {
        let mut seed = 0xa076_1d64_78bd_642fu64.wrapping_add(len as u64);
        let mut a = vec![0u64; n];
        let mut bit = vec![0u64; n + 1];
        for i in 0..n {
            let v = rng_next(&mut seed) % 1024;
            a[i] = v;
            bit_update(&mut bit, n, i, v);
        }
        for i in 0..500usize {
            let l = ((rng_next(&mut seed) >> 32) as usize) % (n - len + 1);
            let x = a[l..l + len].iter().fold(0u64, |acc, &v| acc.wrapping_add(v));
            let y = bit_prefix(&bit, l + len - 1) - bit_prefix(&bit, l.wrapping_sub(1));
            assert_eq!(x, y, "verify range len={len} i={i} l={l}");
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

fn measure_n_sweep(mixed: bool) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mode = if mixed { "mixed" } else { "query" };
    for &n in &ALL_N {
        verify_n(n, mixed);
        let mut a = vec![0u64; n];
        let mut bit = vec![0u64; n + 1];

        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;

        let q_brute;
        let q_bit;
        {
            let (pos, delta, is_q) = gen_ops(n, q0_brute, mixed, &mut seed);
            let ns = if mixed {
                time_mixed_brute(&mut a, &pos, &delta, &is_q)
            } else {
                time_query_brute(&a, &pos)
            };
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let (pos, delta, is_q) = gen_ops(n, q0_bit, mixed, &mut seed);
            let ns = if mixed {
                time_mixed_bit(&mut bit, n, &pos, &delta, &is_q)
            } else {
                time_query_bit(&bit, &pos)
            };
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let (pos_b, del_b, isb) = gen_ops(n, q_brute, mixed, &mut seed);
        let (pos_t, del_t, ist) = gen_ops(n, q_bit, mixed, &mut seed);

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(if mixed {
                time_mixed_brute(&mut a, &pos_b, &del_b, &isb)
            } else {
                time_query_brute(&a, &pos_b)
            });
            s_bit.push(if mixed {
                time_mixed_bit(&mut bit, n, &pos_t, &del_t, &ist)
            } else {
                time_query_bit(&bit, &pos_t)
            });
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("{mode},{n},{t_brute:.3},{t_bit:.3}");
    }
}

fn measure_range() {
    const N: usize = 1 << 20;
    verify_range(N);
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let mut a = vec![0u64; N];
    let mut bit = vec![0u64; N + 1];
    for i in 0..N {
        let v = rng_next(&mut seed) % 1024;
        a[i] = v;
        bit_update(&mut bit, N, i, v);
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
            let ns = time_range_brute(&a, &ls, len);
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let mut ls = vec![0usize; q0_bit];
            for l in ls.iter_mut() {
                *l = ((rng_next(&mut seed) >> 32) as usize) % (N - len + 1);
            }
            let ns = time_range_bit(&bit, &ls, len);
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
            s_brute.push(time_range_brute(&a, &ls_b, len));
            s_bit.push(time_range_bit(&bit, &ls_t, len));
        }

        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;
        println!("range,{len},{t_brute:.3},{t_bit:.3}");
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    if which == "mixed" || which == "all" {
        measure_n_sweep(true);
    }
    if which == "query" || which == "all" {
        measure_n_sweep(false);
    }
    if which == "range" || which == "all" {
        measure_range();
    }
}
