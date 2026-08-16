// Nested hybrid: Fenwick over block aggregates (upper layer) + plain-array
// brute scan inside each block (lower layer), vs plain brute and plain BIT.
//
//   prefix(p):  Fenwick-prefix over the c = p/B complete blocks before p's
//               block (O(log nb)), then scan a[cB..=p] directly (<= B elements).
//   update(p,d): a[p] += d, then Fenwick-update the block aggregate (O(log nb)).
//
// Only invertible semigroups (sum/xor) support the mixed/update modes.
// Build: rustc --edition=2024 -O -C target-cpu=native bench_nested_brute.rs -o bench_nested_rs
use std::hint::black_box;
use std::time::Instant;

const ROUNDS: usize = 9;
const TARGET_NS: f64 = 3e6; // ~3 ms per timed pass
const Q_MIN: usize = 256;
const Q_MAX: usize = 4_000_000;

const ALL_N: [usize; 45] = [
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 96,
    128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144,
    393216, 524288, 786432, 1048576,
];

// Block sizes (powers of two so block index / offset use shifts and masks).
const ALL_B: [usize; 9] = [2, 4, 8, 16, 32, 64, 128, 256, 512];

#[derive(Copy, Clone, PartialEq)]
enum Mix {
    Query,
    Mixed50,
    Mixed25,
    Mixed75,
}

fn is_query_op(mix: Mix, i: usize) -> bool {
    match mix {
        Mix::Query => true,
        Mix::Mixed50 => i & 1 == 0,
        Mix::Mixed25 => i % 4 != 0,
        Mix::Mixed75 => i % 4 == 0,
    }
}

trait Semi: Copy {
    const ID: u64;
    fn op(a: u64, b: u64) -> u64;
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
struct Xor;
impl Semi for Xor {
    const ID: u64 = 0;
    fn op(a: u64, b: u64) -> u64 {
        a ^ b
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

fn gen_pos(n: usize, seed: &mut u64) -> usize {
    ((rng_next(seed) >> 32) as usize) % n
}

// ---------------- plain brute force ----------------
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

// ---------------- plain Fenwick tree (BIT) ----------------
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

// ---------------- nested: Fenwick over blocks + brute inside block ----------------
#[inline(always)]
fn block_of(p: usize, b: usize) -> usize {
    p >> b.trailing_zeros()
}

#[inline(always)]
fn nested_prefix<O: Semi>(a: &[u64], bit: &[u64], b: usize, p: usize) -> u64 {
    let c = block_of(p, b); // complete blocks before p's block
    let mut s = bit_prefix::<O>(bit, c.wrapping_sub(1));
    let start = p & !(b - 1);
    for &v in unsafe { a.get_unchecked(start..=p) } {
        s = O::op(s, v);
    }
    s
}

#[inline(always)]
fn nested_update<O: Semi>(a: &mut [u64], bit: &mut [u64], b: usize, nb: usize, p: usize, d: u64) {
    let slot = unsafe { a.get_unchecked_mut(p) };
    *slot = O::op(*slot, d);
    bit_apply::<O>(bit, nb, block_of(p, b), d);
}

fn nested_build<O: Semi>(vals: &[u64], n: usize, b: usize) -> (Vec<u64>, Vec<u64>, usize) {
    let nb = n.div_ceil(b);
    let mut a = vec![0u64; nb * b];
    let mut sums = vec![O::ID; nb];
    for i in 0..n {
        a[i] = vals[i];
        let bi = block_of(i, b);
        sums[bi] = O::op(sums[bi], vals[i]);
    }
    let mut bit = vec![O::ID; nb + 1];
    for i in 0..nb {
        bit[i + 1] = sums[i];
    }
    for i in 1..=nb {
        let j = i + (i & i.wrapping_neg());
        if j <= nb {
            bit[j] = O::op(bit[j], bit[i]);
        }
    }
    (a, bit, nb)
}

// ---------------- timed kernels ----------------
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
fn run_query_nested<O: Semi>(a: &[u64], bit: &[u64], b: usize, pos: &[usize]) -> u64 {
    let mut acc = 0u64;
    for &p in pos {
        acc = acc.wrapping_add(nested_prefix::<O>(a, bit, b, p));
    }
    acc
}

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
fn run_mixed_nested<O: Semi>(
    a: &mut [u64],
    bit: &mut [u64],
    b: usize,
    nb: usize,
    pos: &[usize],
    delta: &[u64],
    is_q: &[u8],
) -> u64 {
    let mut acc = 0u64;
    for i in 0..pos.len() {
        let p = pos[i];
        if is_q[i] != 0 {
            acc = acc.wrapping_add(nested_prefix::<O>(a, bit, b, p));
        } else {
            nested_update::<O>(a, bit, b, nb, p, delta[i]);
        }
    }
    acc
}

fn time_query<O: Semi>(a: &[u64], pos: &[usize]) -> f64 {
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

fn time_query_nested<O: Semi>(a: &[u64], bit: &[u64], b: usize, pos: &[usize]) -> f64 {
    let t0 = Instant::now();
    let acc = run_query_nested::<O>(a, bit, b, pos);
    let t1 = Instant::now();
    black_box(acc);
    t1.duration_since(t0).as_secs_f64() * 1e9
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

fn time_mixed_nested<O: Semi>(
    a: &mut [u64],
    bit: &mut [u64],
    b: usize,
    nb: usize,
    pos: &[usize],
    delta: &[u64],
    is_q: &[u8],
) -> f64 {
    let t0 = Instant::now();
    let acc = run_mixed_nested::<O>(a, bit, b, nb, pos, delta, is_q);
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
        pos.push(gen_pos(n, seed));
        delta.push(rng_next(seed) % 1024);
        is_q.push(if is_query_op(mix, i) { 1 } else { 0 });
    }
    (pos, delta, is_q)
}

// ---------------- verification ----------------
// nested vs plain BIT at every size (both O(log)); brute-vs-BIT cross-check
// only at small n where the brute prefix scan is cheap.
fn verify_nested<O: Semi>(vals: &[u64], n: usize, b: usize) {
    let mut seed = 0xfeed_face_cafe_beefu64
        ^ (n as u64).rotate_left(17)
        ^ (b as u64).rotate_left(41);
    let mut a = vals.to_vec();
    let mut bit = vec![O::ID; n + 1];
    for i in 0..n {
        bit_apply::<O>(&mut bit, n, i, vals[i]);
    }
    let (mut na, mut nbit, nb) = nested_build::<O>(vals, n, b);
    for i in 0..3000usize {
        let p = gen_pos(n, &mut seed);
        let d = rng_next(&mut seed) % 1024;
        if i & 1 != 0 {
            brute_update::<O>(&mut a, p, d);
            bit_apply::<O>(&mut bit, n, p, d);
            nested_update::<O>(&mut na, &mut nbit, b, nb, p, d);
        } else {
            let x = bit_prefix::<O>(&bit, p);
            let y = nested_prefix::<O>(&na, &nbit, b, p);
            assert_eq!(x, y, "verify nested {} n={n} b={b} i={i} p={p}", std::any::type_name::<O>());
            if n <= 4096 {
                assert_eq!(brute_prefix::<O>(&a, p), x, "verify brute/BIT {} n={n} i={i}", std::any::type_name::<O>());
            }
        }
    }
}

// ---------------- measurement ----------------
fn measure<O: Semi>(name: &str, mix: Mix, mode: &str) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64
        ^ (name.as_bytes().iter().fold(0u64, |acc, &c| acc.wrapping_mul(131).wrapping_add(c as u64))
            .rotate_left(21)
            + mode.len() as u64 * 0x9e37_79b9_7f4a_7c15);
    let mut summary: Vec<(usize, usize, f64, f64, f64)> = Vec::new(); // n, best B, nested, brute, BIT

    for &n in &ALL_N {
        let mut vals = Vec::with_capacity(n);
        for _ in 0..n {
            vals.push(rng_next(&mut seed) % 1024);
        }
        let mut a = vals.clone();
        let mut bit = vec![O::ID; n + 1];
        for i in 0..n {
            bit_apply::<O>(&mut bit, n, i, vals[i]);
        }

        // calibrate and time plain brute + plain BIT once per n
        let q0_brute = (4_000_000usize / (n / 2).max(1)).clamp(Q_MIN, 262144);
        let q0_bit = 131_072usize;
        let (q_brute, q_bit);
        {
            let (pos, delta, is_q) = gen_ops(n, q0_brute, mix, &mut seed);
            let ns = if mix == Mix::Query {
                time_query::<O>(&a, &pos)
            } else {
                time_mixed_brute::<O>(&mut a, &pos, &delta, &is_q)
            };
            q_brute = clamp_q(ns / q0_brute as f64);
        }
        {
            let (pos, delta, is_q) = gen_ops(n, q0_bit, mix, &mut seed);
            let ns = if mix == Mix::Query {
                time_query_bit::<O>(&bit, &pos)
            } else {
                time_mixed_bit::<O>(&mut bit, n, &pos, &delta, &is_q)
            };
            q_bit = clamp_q(ns / q0_bit as f64);
        }

        let (pos_b, del_b, isb) = gen_ops(n, q_brute, mix, &mut seed);
        let (pos_t, del_t, ist) = gen_ops(n, q_bit, mix, &mut seed);

        let mut s_brute = Vec::with_capacity(ROUNDS);
        let mut s_bit = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            s_brute.push(if mix == Mix::Query {
                time_query::<O>(&a, &pos_b)
            } else {
                time_mixed_brute::<O>(&mut a, &pos_b, &del_b, &isb)
            });
            s_bit.push(if mix == Mix::Query {
                time_query_bit::<O>(&bit, &pos_t)
            } else {
                time_mixed_bit::<O>(&mut bit, n, &pos_t, &del_t, &ist)
            });
        }
        let t_brute = median(&mut s_brute) / q_brute as f64;
        let t_bit = median(&mut s_bit) / q_bit as f64;

        let mut best_b = ALL_B[0];
        let mut best_t = f64::INFINITY;
        for &b in &ALL_B {
            verify_nested::<O>(&vals, n, b);
            let (na0, nbit0, nb) = nested_build::<O>(&vals, n, b);
            let mut na = na0;
            let mut nbit = nbit0;

            let q0_nested = 131_072usize;
            let q_nested;
            {
                let (pos, delta, is_q) = gen_ops(n, q0_nested, mix, &mut seed);
                let ns = if mix == Mix::Query {
                    time_query_nested::<O>(&na, &nbit, b, &pos)
                } else {
                    time_mixed_nested::<O>(&mut na, &mut nbit, b, nb, &pos, &delta, &is_q)
                };
                q_nested = clamp_q(ns / q0_nested as f64);
            }

            let (pos_nt, del_nt, ist_nt) = gen_ops(n, q_nested, mix, &mut seed);
            let mut s_nested = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                s_nested.push(if mix == Mix::Query {
                    time_query_nested::<O>(&na, &nbit, b, &pos_nt)
                } else {
                    time_mixed_nested::<O>(&mut na, &mut nbit, b, nb, &pos_nt, &del_nt, &ist_nt)
                });
            }
            let t_nested = median(&mut s_nested) / q_nested as f64;
            println!("{name},{mode},{n},{b},{t_brute:.3},{t_bit:.3},{t_nested:.3}");
            if t_nested < best_t {
                best_t = t_nested;
                best_b = b;
            }
        }
        summary.push((n, best_b, best_t, t_brute, t_bit));
    }

    // summary (stderr so stdout stays pure CSV)
    let mut wins = 0usize;
    let mut bcount = vec![0usize; ALL_B.len()];
    let mut win_rows: Vec<(usize, usize, f64, f64, f64)> = Vec::new();
    for &(n, b, tn, tb, tt) in &summary {
        if tn < tb && tn < tt {
            wins += 1;
            bcount[ALL_B.iter().position(|&x| x == b).unwrap()] += 1;
            win_rows.push((n, b, tn, tb, tt));
        }
    }
    eprintln!("\n== {name}/{mode}: nested beats both brute and BIT ==");
    eprintln!("sizes where nested is fastest: {wins}/{}", ALL_N.len());
    if !win_rows.is_empty() {
        let mut bs: Vec<usize> = ALL_B.to_vec();
        bs.retain(|&b| bcount[ALL_B.iter().position(|&x| x == b).unwrap()] > 0);
        eprintln!("best-B histogram: {:?}", bs.iter().map(|&b| (b, bcount[ALL_B.iter().position(|&x| x == b).unwrap()])).collect::<Vec<_>>());
        eprintln!("sample rows (n, B, nested ns, brute ns, BIT ns, speedup vs best plain):");
        for &(n, b, tn, tb, tt) in win_rows.iter().rev().take(12) {
            let best_plain = tb.min(tt);
            eprintln!("  n={n:>7} B={b:>3} nested={tn:7.1} brute={tb:7.1} bit={tt:7.1}  {:.2}x vs best plain", best_plain / tn);
        }
    } else {
        eprintln!("nested never wins; closest gaps:");
        let mut gaps: Vec<(f64, usize, usize, f64)> = summary
            .iter()
            .map(|&(n, b, tn, tb, tt)| (tn / tb.min(tt), n, b, tn))
            .collect();
        gaps.sort_by(|x, y| x.0.total_cmp(&y.0));
        for &(r, n, b, tn) in gaps.iter().take(8) {
            eprintln!("  n={n:>7} B={b:>3} nested/best_plain={r:.3} (nested {tn:.1} ns)");
        }
    }
}

fn main() {
    // optional filters: ./bench_nested_rs [op] [mode]
    let want_op = std::env::args().nth(1);
    let want_mode = std::env::args().nth(2);
    let combos: [(&str, Mix, &str); 8] = [
        ("sum", Mix::Query, "query"),
        ("sum", Mix::Mixed50, "mixed"),
        ("sum", Mix::Mixed25, "mixed_25"),
        ("sum", Mix::Mixed75, "mixed_75"),
        ("xor", Mix::Query, "query"),
        ("xor", Mix::Mixed50, "mixed"),
        ("xor", Mix::Mixed25, "mixed_25"),
        ("xor", Mix::Mixed75, "mixed_75"),
    ];
    for (name, mix, mode) in combos {
        if let Some(op) = &want_op {
            if op != name {
                continue;
            }
        }
        if let Some(m) = &want_mode {
            if m != mode {
                continue;
            }
        }
        match name {
            "sum" => measure::<Sum>(name, mix, mode),
            _ => measure::<Xor>(name, mix, mode),
        }
    }
}
