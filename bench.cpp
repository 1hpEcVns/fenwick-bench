// Fenwick tree (BIT) vs plain-array brute force across semigroup operations.
// ops: sum, min, max, and, or, xor.
// modes: mixed (1:1 point-update + prefix-query, invertible ops only),
//        query (prefix queries only), range (range sums, N=2^20, invertible ops).
// Build: g++ -O3 -march=native -std=c++23 bench.cpp -o bench
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u8 = std::uint8_t;

enum class Op : int { Sum = 0, Min, Max, And, Or, Xor };
enum class Mix : int { Query = 0, QueryTail = 1, Mixed50 = 2, Mixed25 = 3, Mixed75 = 4 };

static const char* op_name(Op op) {
    switch (op) {
    case Op::Sum: return "sum";
    case Op::Min: return "min";
    case Op::Max: return "max";
    case Op::And: return "and";
    case Op::Or:  return "or";
    case Op::Xor: return "xor";
    }
    return "?";
}

static bool op_invertible(Op op) { return op == Op::Sum || op == Op::Xor; }

static inline bool is_query_op(Mix mix, size_t i) {
    switch (mix) {
    case Mix::Query:
    case Mix::QueryTail: return true;
    case Mix::Mixed50:   return (i & 1) == 0;
    case Mix::Mixed25:   return (i % 4) != 0;
    case Mix::Mixed75:   return (i % 4) == 0;
    }
    return true;
}

static inline size_t gen_pos(Mix mix, size_t n, std::mt19937_64& rng) {
    if (mix == Mix::QueryTail)
        return n - (size_t)(rng() % (n / 10 + 1)) - 1;
    return (size_t)(rng() % n);
}

static inline void black_box_u64(u64 x) {
    asm volatile("" : "+r"(x) : : "memory");
}

static inline double now_ns() {
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <Op OP>
static inline u64 identity() {
    if constexpr (OP == Op::Sum) return 0;
    if constexpr (OP == Op::Min) return ~u64(0);
    if constexpr (OP == Op::Max) return 0;
    if constexpr (OP == Op::And) return ~u64(0);
    if constexpr (OP == Op::Or) return 0;
    if constexpr (OP == Op::Xor) return 0;
    return 0;
}

template <Op OP>
static inline u64 binop(u64 a, u64 b) {
    if constexpr (OP == Op::Sum) return a + b;
    if constexpr (OP == Op::Min) return a < b ? a : b;
    if constexpr (OP == Op::Max) return a > b ? a : b;
    if constexpr (OP == Op::And) return a & b;
    if constexpr (OP == Op::Or) return a | b;
    if constexpr (OP == Op::Xor) return a ^ b;
    return 0;
}

// ---------------- brute force: plain array ----------------
template <Op OP>
static inline u64 brute_prefix(const u64* a, size_t i) {
    u64 s = identity<OP>();
    for (size_t j = 0; j <= i; ++j) s = binop<OP>(s, a[j]);
    return s;
}

template <Op OP>
static inline void brute_update(u64* a, size_t i, u64 d) {
    // Only instantiated for invertible ops (sum/xor).
    if constexpr (OP == Op::Sum) a[i] += d;
    if constexpr (OP == Op::Xor) a[i] ^= d;
}

// ---------------- Fenwick tree (BIT), tree[1..n] ----------------
template <Op OP>
static inline u64 bit_prefix(const u64* bit, size_t i) {
    // i = SIZE_MAX means "before 0": i+1 wraps to 0 and the loop is skipped.
    u64 s = identity<OP>();
    size_t k = i + 1;
    while (k != 0) {
        s = binop<OP>(s, bit[k]);
        k &= k - 1;
    }
    return s;
}

template <Op OP>
static inline void bit_apply(u64* bit, size_t n, size_t i, u64 d) {
    // Merge one element into the tree (build, or invertible point update).
    for (size_t k = i + 1; k <= n; k += k & (~k + 1))
        bit[k] = binop<OP>(bit[k], d);
}

// ---------------- timed kernels ----------------
template <Op OP>
static u64 run_mixed_brute(u64* a, const size_t* pos, const u64* delta,
                           const u8* is_q, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += brute_prefix<OP>(a, pos[i]);
        else brute_update<OP>(a, pos[i], delta[i]);
    }
    return acc;
}

template <Op OP>
static u64 run_mixed_bit(u64* bit, size_t n, const size_t* pos,
                         const u64* delta, const u8* is_q, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += bit_prefix<OP>(bit, pos[i]);
        else bit_apply<OP>(bit, n, pos[i], delta[i]);
    }
    return acc;
}

template <Op OP>
static u64 run_query_brute(const u64* a, const size_t* pos, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += brute_prefix<OP>(a, pos[i]);
    return acc;
}

template <Op OP>
static u64 run_query_bit(const u64* bit, const size_t* pos, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += bit_prefix<OP>(bit, pos[i]);
    return acc;
}

template <Op OP>
static u64 run_range_brute(const u64* a, const size_t* ls, size_t L, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        size_t l = ls[i];
        u64 s = identity<OP>();
        for (size_t j = l; j < l + L; ++j) s = binop<OP>(s, a[j]);
        acc += s;
    }
    return acc;
}

template <Op OP>
static u64 run_range_bit(const u64* bit, const size_t* ls, size_t L, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        size_t l = ls[i];
        u64 hi = bit_prefix<OP>(bit, l + L - 1);
        u64 lo = bit_prefix<OP>(bit, l - 1);
        if constexpr (OP == Op::Sum) acc += hi - lo;
        if constexpr (OP == Op::Xor) acc += hi ^ lo;
    }
    return acc;
}

// ---------------- BIT binary search (树状数组上二分), sum only ----------------
static inline size_t top_of(size_t n) {
    return size_t(1) << (63 - __builtin_clzll(n));
}

// First prefix that reaches k: number of elements consumed (0..n).
static inline size_t brute_bisect(const u64* a, size_t n, u64 k) {
    u64 s = 0;
    size_t j = 0;
    while (j < n && s < k) {
        s += a[j];
        ++j;
    }
    return j;
}

// Same semantics via binary lifting on the BIT (values must be non-negative).
static inline size_t bit_bisect(const u64* bit, size_t n, size_t top, u64 k) {
    size_t pos = 0;
    size_t m = top;
    while (m != 0) {
        size_t next = pos + m;
        if (next <= n && bit[next] < k) {
            pos = next;
            k -= bit[next];
        }
        m >>= 1;
    }
    return pos < n ? pos + 1 : n;
}

static u64 run_bisect_mixed_brute(u64* a, size_t n, const size_t* pos,
                                  const u64* delta, const u8* is_q,
                                  const u64* ks, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += brute_bisect(a, n, ks[i]);
        else a[pos[i]] += delta[i];
    }
    return acc;
}

static u64 run_bisect_mixed_bit(u64* bit, size_t n, size_t top,
                                const size_t* pos, const u64* delta,
                                const u8* is_q, const u64* ks, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += bit_bisect(bit, n, top, ks[i]);
        else bit_apply<Op::Sum>(bit, n, pos[i], delta[i]);
    }
    return acc;
}

static u64 run_bisect_query_brute(const u64* a, size_t n, const u64* ks, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += brute_bisect(a, n, ks[i]);
    return acc;
}

static u64 run_bisect_query_bit(const u64* bit, size_t n, size_t top,
                                const u64* ks, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += bit_bisect(bit, n, top, ks[i]);
    return acc;
}

template <Op OP>
static double time_mixed_brute(u64* a, const std::vector<size_t>& pos,
                               const std::vector<u64>& delta,
                               const std::vector<u8>& is_q) {
    double t0 = now_ns();
    u64 acc = run_mixed_brute<OP>(a, pos.data(), delta.data(), is_q.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

template <Op OP>
static double time_mixed_bit(u64* bit, size_t n, const std::vector<size_t>& pos,
                             const std::vector<u64>& delta,
                             const std::vector<u8>& is_q) {
    double t0 = now_ns();
    u64 acc = run_mixed_bit<OP>(bit, n, pos.data(), delta.data(), is_q.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

template <Op OP>
static double time_query_brute(const u64* a, const std::vector<size_t>& pos) {
    double t0 = now_ns();
    u64 acc = run_query_brute<OP>(a, pos.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

template <Op OP>
static double time_query_bit(const u64* bit, const std::vector<size_t>& pos) {
    double t0 = now_ns();
    u64 acc = run_query_bit<OP>(bit, pos.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

template <Op OP>
static double time_range_brute(const u64* a, const std::vector<size_t>& ls,
                               size_t L) {
    double t0 = now_ns();
    u64 acc = run_range_brute<OP>(a, ls.data(), L, ls.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

template <Op OP>
static double time_range_bit(const u64* bit, const std::vector<size_t>& ls,
                             size_t L) {
    double t0 = now_ns();
    u64 acc = run_range_bit<OP>(bit, ls.data(), L, ls.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_bisect_mixed_brute(u64* a, size_t n,
                                      const std::vector<size_t>& pos,
                                      const std::vector<u64>& delta,
                                      const std::vector<u8>& is_q,
                                      const std::vector<u64>& ks) {
    double t0 = now_ns();
    u64 acc = run_bisect_mixed_brute(a, n, pos.data(), delta.data(), is_q.data(),
                                     ks.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_bisect_mixed_bit(u64* bit, size_t n, size_t top,
                                    const std::vector<size_t>& pos,
                                    const std::vector<u64>& delta,
                                    const std::vector<u8>& is_q,
                                    const std::vector<u64>& ks) {
    double t0 = now_ns();
    u64 acc = run_bisect_mixed_bit(bit, n, top, pos.data(), delta.data(),
                                   is_q.data(), ks.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_bisect_query_brute(const u64* a, size_t n,
                                      const std::vector<u64>& ks) {
    double t0 = now_ns();
    u64 acc = run_bisect_query_brute(a, n, ks.data(), ks.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_bisect_query_bit(const u64* bit, size_t n, size_t top,
                                    const std::vector<u64>& ks) {
    double t0 = now_ns();
    u64 acc = run_bisect_query_bit(bit, n, top, ks.data(), ks.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

// ---------------- calibration helpers ----------------
static constexpr size_t ROUNDS = 9;
static constexpr double TARGET_NS = 3e6;  // ~3 ms per timed pass
static constexpr size_t Q_MIN = 256;
static constexpr size_t Q_MAX = 4'000'000;

static size_t clamp_q(double per_ns) {
    size_t q = (size_t)(TARGET_NS / per_ns);
    return std::clamp(q, Q_MIN, Q_MAX);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------- op generation ----------------
static void gen_ops(size_t n, size_t q, Mix mix, std::mt19937_64& rng,
                    std::vector<size_t>& pos, std::vector<u64>& delta,
                    std::vector<u8>& is_q) {
    pos.resize(q);
    delta.resize(q);
    is_q.resize(q);
    for (size_t i = 0; i < q; ++i) {
        pos[i] = gen_pos(mix, n, rng);
        delta[i] = rng() % 1024;
        is_q[i] = is_query_op(mix, i) ? 1 : 0;
    }
}

static void gen_bisect_ops(size_t n, size_t q, bool mixed, std::mt19937_64& rng,
                           std::vector<size_t>& pos, std::vector<u64>& delta,
                           std::vector<u8>& is_q, std::vector<u64>& ks) {
    pos.resize(q);
    delta.resize(q);
    is_q.resize(q);
    ks.resize(q);
    for (size_t i = 0; i < q; ++i) {
        pos[i] = (size_t)(rng() % n);
        delta[i] = rng() % 1024;
        is_q[i] = (!mixed || (i & 1) == 0) ? 1 : 0;
        ks[i] = is_q[i] ? (rng() % (n * 512) + 1) : 0;
    }
}

// ---------------- verification ----------------
template <Op OP>
static void verify_query(size_t n) {
    std::mt19937_64 rng(0xd1b54a32d192ed03ULL + (u64)OP * 0x9e3779b97f4a7c15ULL);
    std::vector<u64> a(n), bit(n + 1, identity<OP>());
    for (size_t i = 0; i < n; ++i) {
        u64 v = rng() % 1024;
        a[i] = v;
        bit_apply<OP>(bit.data(), n, i, v);
    }
    for (size_t i = 0; i < 4000; ++i) {
        size_t p = (size_t)(rng() % n);
        u64 x = brute_prefix<OP>(a.data(), p);
        u64 y = bit_prefix<OP>(bit.data(), p);
        if (x != y) {
            std::printf("VERIFY FAIL op=%s query n=%zu i=%zu p=%zu brute=%llu bit=%llu\n",
                        op_name(OP), n, i, p,
                        (unsigned long long)x, (unsigned long long)y);
            std::abort();
        }
    }
}

template <Op OP>
static void verify_mixed(size_t n) {
    std::mt19937_64 rng(0xd1b54a32d192ed03ULL + (u64)OP * 0x9e3779b97f4a7c15ULL + 1);
    std::vector<u64> a(n), bit(n + 1, identity<OP>());
    for (size_t i = 0; i < n; ++i) {
        u64 v = rng() % 1024;
        a[i] = v;
        bit_apply<OP>(bit.data(), n, i, v);
    }
    for (size_t i = 0; i < 4000; ++i) {
        size_t p = (size_t)(rng() % n);
        u64 d = rng() % 1024;
        if ((i & 1) != 0) {
            brute_update<OP>(a.data(), p, d);
            bit_apply<OP>(bit.data(), n, p, d);
        } else {
            u64 x = brute_prefix<OP>(a.data(), p);
            u64 y = bit_prefix<OP>(bit.data(), p);
            if (x != y) {
                std::printf("VERIFY FAIL op=%s mixed n=%zu i=%zu p=%zu brute=%llu bit=%llu\n",
                            op_name(OP), n, i, p,
                            (unsigned long long)x, (unsigned long long)y);
                std::abort();
            }
        }
    }
}

template <Op OP>
static void verify_range(size_t n) {
    for (size_t L : {1u, 2u, 3u, 7u, 16u, 64u, 257u, 1024u}) {
        std::mt19937_64 rng(0xa0761d6478bd642fULL + (u64)OP * 0x9e3779b97f4a7c15ULL + L);
        std::vector<u64> a(n), bit(n + 1, identity<OP>());
        for (size_t i = 0; i < n; ++i) {
            u64 v = rng() % 1024;
            a[i] = v;
            bit_apply<OP>(bit.data(), n, i, v);
        }
        for (size_t i = 0; i < 500; ++i) {
            size_t l = (size_t)(rng() % (n - L + 1));
            u64 x = identity<OP>();
            for (size_t j = l; j < l + L; ++j) x = binop<OP>(x, a[j]);
            u64 hi = bit_prefix<OP>(bit.data(), l + L - 1);
            u64 lo = bit_prefix<OP>(bit.data(), l - 1);
            u64 y = (OP == Op::Sum) ? hi - lo : hi ^ lo;
            if (x != y) {
                std::printf("VERIFY FAIL op=%s range L=%zu i=%zu l=%zu brute=%llu bit=%llu\n",
                            op_name(OP), L, i, l,
                            (unsigned long long)x, (unsigned long long)y);
                std::abort();
            }
        }
    }
}

static void verify_bisect(size_t n, bool mixed) {
    std::mt19937_64 rng(0x123456789abcdef0ULL + (mixed ? 1 : 0));
    std::vector<u64> a(n), bit(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        u64 v = rng() % 1024;
        a[i] = v;
        bit_apply<Op::Sum>(bit.data(), n, i, v);
    }
    for (size_t i = 0; i < 4000; ++i) {
        size_t p = (size_t)(rng() % n);
        u64 d = rng() % 1024;
        if (mixed && (i & 1) != 0) {
            a[p] += d;
            bit_apply<Op::Sum>(bit.data(), n, p, d);
        } else {
            u64 total = 0;
            for (size_t j = 0; j < n; ++j) total += a[j];
            u64 k = total == 0 ? 1 : rng() % total + 1;
            size_t x = brute_bisect(a.data(), n, k);
            size_t y = bit_bisect(bit.data(), n, top_of(n), k);
            if (x != y) {
                std::printf("VERIFY FAIL bisect mixed=%d n=%zu i=%zu k=%llu brute=%zu bit=%zu\n",
                            (int)mixed, n, i, (unsigned long long)k, x, y);
                std::abort();
            }
        }
    }
}

// ---------------- n sweep ----------------
static const std::vector<size_t>& all_n() {
    static const std::vector<size_t> v = {
        4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 96,
        128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
        8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072,
        196608, 262144, 393216, 524288, 786432, 1048576
    };
    return v;
}

template <Op OP>
static void measure_query(Op op, Mix mix = Mix::Query, const char* mode = "query") {
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL + (u64)OP * 0x9e3779b97f4a7c15ULL);
    for (size_t n : all_n()) {
        verify_query<OP>(n);
        std::vector<u64> a(n), bit(n + 1, identity<OP>());
        for (size_t i = 0; i < n; ++i) {
            u64 v = rng() % 1024;
            a[i] = v;
            bit_apply<OP>(bit.data(), n, i, v);
        }

        size_t q0_brute = std::clamp(size_t(4'000'000) / std::max<size_t>(n / 2, 1),
                                     Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;

        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_brute, mix, rng, pos, delta, is_q);
            double ns = time_query_brute<OP>(a.data(), pos);
            q_brute = clamp_q(ns / (double)q0_brute);
        }
        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_bit, mix, rng, pos, delta, is_q);
            double ns = time_query_bit<OP>(bit.data(), pos);
            q_bit = clamp_q(ns / (double)q0_bit);
        }

        std::vector<size_t> pos_b, pos_t;
        std::vector<u64> del_b, del_t;
        std::vector<u8> isb, ist;
        gen_ops(n, q_brute, mix, rng, pos_b, del_b, isb);
        gen_ops(n, q_bit, mix, rng, pos_t, del_t, ist);

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(time_query_brute<OP>(a.data(), pos_b));
            s_bit.push_back(time_query_bit<OP>(bit.data(), pos_t));
        }

        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("%s,%s,%zu,%.3f,%.3f\n", op_name(op), mode, n, t_brute, t_bit);
        std::fflush(stdout);
    }
}

template <Op OP>
static void measure_mixed(Op op, Mix mix = Mix::Mixed50, const char* mode = "mixed") {
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL + (u64)OP * 0x9e3779b97f4a7c15ULL + 2);
    for (size_t n : all_n()) {
        verify_mixed<OP>(n);
        std::vector<u64> a(n), bit(n + 1, identity<OP>());

        size_t q0_brute = std::clamp(size_t(4'000'000) / std::max<size_t>(n / 2, 1),
                                     Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;

        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_brute, mix, rng, pos, delta, is_q);
            double ns = time_mixed_brute<OP>(a.data(), pos, delta, is_q);
            q_brute = clamp_q(ns / (double)q0_brute);
        }
        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_bit, mix, rng, pos, delta, is_q);
            double ns = time_mixed_bit<OP>(bit.data(), n, pos, delta, is_q);
            q_bit = clamp_q(ns / (double)q0_bit);
        }

        std::vector<size_t> pos_b, pos_t;
        std::vector<u64> del_b, del_t;
        std::vector<u8> isb, ist;
        gen_ops(n, q_brute, mix, rng, pos_b, del_b, isb);
        gen_ops(n, q_bit, mix, rng, pos_t, del_t, ist);

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(time_mixed_brute<OP>(a.data(), pos_b, del_b, isb));
            s_bit.push_back(time_mixed_bit<OP>(bit.data(), n, pos_t, del_t, ist));
        }

        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("%s,%s,%zu,%.3f,%.3f\n", op_name(op), mode, n, t_brute, t_bit);
        std::fflush(stdout);
    }
}

// ---------------- L sweep: range sums on a fixed-size array ----------------
static const std::vector<size_t>& all_L() {
    static const std::vector<size_t> v = {
        2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,
        60, 64, 80, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192
    };
    return v;
}

template <Op OP>
static void measure_range(Op op) {
    const size_t N = 1u << 20;
    verify_range<OP>(N);
    std::mt19937_64 rng(0x243f6a8885a308d3ULL + (u64)OP * 0x9e3779b97f4a7c15ULL);
    std::vector<u64> a(N), bit(N + 1, identity<OP>());
    for (size_t i = 0; i < N; ++i) {
        u64 v = rng() % 1024;
        a[i] = v;
        bit_apply<OP>(bit.data(), N, i, v);
    }

    for (size_t L : all_L()) {
        size_t q0_brute = std::clamp(size_t(4'000'000) / L, Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;

        {
            std::vector<size_t> ls(q0_brute);
            for (size_t& l : ls) l = (size_t)(rng() % (N - L + 1));
            double ns = time_range_brute<OP>(a.data(), ls, L);
            q_brute = clamp_q(ns / (double)q0_brute);
        }
        {
            std::vector<size_t> ls(q0_bit);
            for (size_t& l : ls) l = (size_t)(rng() % (N - L + 1));
            double ns = time_range_bit<OP>(bit.data(), ls, L);
            q_bit = clamp_q(ns / (double)q0_bit);
        }

        std::vector<size_t> ls_b(q_brute), ls_t(q_bit);
        for (size_t& l : ls_b) l = (size_t)(rng() % (N - L + 1));
        for (size_t& l : ls_t) l = (size_t)(rng() % (N - L + 1));

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(time_range_brute<OP>(a.data(), ls_b, L));
            s_bit.push_back(time_range_bit<OP>(bit.data(), ls_t, L));
        }

        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("%s,range,%zu,%.3f,%.3f\n", op_name(op), L, t_brute, t_bit);
        std::fflush(stdout);
    }
}

static void measure_bisect(bool mixed) {
    const char* mode = mixed ? "mixed" : "query";
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL + (mixed ? 3 : 4) * 0x9e3779b97f4a7c15ULL);
    for (size_t n : all_n()) {
        verify_bisect(n, mixed);
        size_t top = top_of(n);
        std::vector<u64> a0(n), bit0(n + 1, 0);
        for (size_t i = 0; i < n; ++i) {
            u64 v = rng() % 1024;
            a0[i] = v;
            bit_apply<Op::Sum>(bit0.data(), n, i, v);
        }
        std::vector<u64> a = a0, bit = bit0;

        size_t q0_brute = std::clamp(size_t(4'000'000) / std::max<size_t>(n / 2, 1),
                                     Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;

        {
            std::vector<size_t> pos;
            std::vector<u64> delta, ks;
            std::vector<u8> is_q;
            gen_bisect_ops(n, q0_brute, mixed, rng, pos, delta, is_q, ks);
            if (mixed) a = a0;
            double ns = mixed ? time_bisect_mixed_brute(a.data(), n, pos, delta, is_q, ks)
                              : time_bisect_query_brute(a.data(), n, ks);
            q_brute = clamp_q(ns / (double)q0_brute);
            if (mixed) bit = bit0;
        }
        {
            std::vector<size_t> pos;
            std::vector<u64> delta, ks;
            std::vector<u8> is_q;
            gen_bisect_ops(n, q0_bit, mixed, rng, pos, delta, is_q, ks);
            if (mixed) bit = bit0;
            double ns = mixed ? time_bisect_mixed_bit(bit.data(), n, top, pos, delta, is_q, ks)
                              : time_bisect_query_bit(bit.data(), n, top, ks);
            q_bit = clamp_q(ns / (double)q0_bit);
        }

        std::vector<size_t> pos_b, pos_t;
        std::vector<u64> del_b, del_t, ks_b, ks_t;
        std::vector<u8> isb, ist;
        gen_bisect_ops(n, q_brute, mixed, rng, pos_b, del_b, isb, ks_b);
        gen_bisect_ops(n, q_bit, mixed, rng, pos_t, del_t, ist, ks_t);

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            if (mixed) {
                a = a0;
                bit = bit0;
            }
            s_brute.push_back(mixed ? time_bisect_mixed_brute(a.data(), n, pos_b, del_b, isb, ks_b)
                                    : time_bisect_query_brute(a.data(), n, ks_b));
            if (mixed) bit = bit0;
            s_bit.push_back(mixed ? time_bisect_mixed_bit(bit.data(), n, top, pos_t, del_t, ist, ks_t)
                                  : time_bisect_query_bit(bit.data(), n, top, ks_t));
        }

        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("sum_bisect,%s,%zu,%.3f,%.3f\n", mode, n, t_brute, t_bit);
        std::fflush(stdout);
    }
}

// ---------------- u32 element type (sum only) ----------------
static inline u32 brute_prefix_u32(const u32* a, size_t i) {
    u32 s = 0;
    for (size_t j = 0; j <= i; ++j) s += a[j];
    return s;
}

static inline u32 bit_prefix_u32(const u32* bit, size_t i) {
    u32 s = 0;
    size_t k = i + 1;
    while (k != 0) {
        s += bit[k];
        k &= k - 1;
    }
    return s;
}

static inline void bit_apply_u32(u32* bit, size_t n, size_t i, u32 d) {
    for (size_t k = i + 1; k <= n; k += k & (~k + 1)) bit[k] += d;
}

static u64 run_query_brute_u32(const u32* a, const size_t* pos, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += brute_prefix_u32(a, pos[i]);
    return acc;
}

static u64 run_query_bit_u32(const u32* bit, const size_t* pos, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += bit_prefix_u32(bit, pos[i]);
    return acc;
}

static u64 run_mixed_brute_u32(u32* a, const size_t* pos, const u64* delta,
                               const u8* is_q, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += brute_prefix_u32(a, pos[i]);
        else a[pos[i]] += (u32)delta[i];
    }
    return acc;
}

static u64 run_mixed_bit_u32(u32* bit, size_t n, const size_t* pos,
                             const u64* delta, const u8* is_q, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += bit_prefix_u32(bit, pos[i]);
        else bit_apply_u32(bit, n, pos[i], (u32)delta[i]);
    }
    return acc;
}

static u64 run_range_brute_u32(const u32* a, const size_t* ls, size_t L, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        size_t l = ls[i];
        u32 s = 0;
        for (size_t j = l; j < l + L; ++j) s += a[j];
        acc += s;
    }
    return acc;
}

static u64 run_range_bit_u32(const u32* bit, const size_t* ls, size_t L, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        size_t l = ls[i];
        acc += bit_prefix_u32(bit, l + L - 1) - bit_prefix_u32(bit, l - 1);
    }
    return acc;
}

static double time_query_brute_u32(const u32* a, const std::vector<size_t>& pos) {
    double t0 = now_ns();
    u64 acc = run_query_brute_u32(a, pos.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_query_bit_u32(const u32* bit, const std::vector<size_t>& pos) {
    double t0 = now_ns();
    u64 acc = run_query_bit_u32(bit, pos.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_mixed_brute_u32(u32* a, const std::vector<size_t>& pos,
                                   const std::vector<u64>& delta,
                                   const std::vector<u8>& is_q) {
    double t0 = now_ns();
    u64 acc = run_mixed_brute_u32(a, pos.data(), delta.data(), is_q.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_mixed_bit_u32(u32* bit, size_t n, const std::vector<size_t>& pos,
                                 const std::vector<u64>& delta,
                                 const std::vector<u8>& is_q) {
    double t0 = now_ns();
    u64 acc = run_mixed_bit_u32(bit, n, pos.data(), delta.data(), is_q.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_range_brute_u32(const u32* a, const std::vector<size_t>& ls,
                                   size_t L) {
    double t0 = now_ns();
    u64 acc = run_range_brute_u32(a, ls.data(), L, ls.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_range_bit_u32(const u32* bit, const std::vector<size_t>& ls,
                                 size_t L) {
    double t0 = now_ns();
    u64 acc = run_range_bit_u32(bit, ls.data(), L, ls.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static void verify_query_u32(size_t n) {
    std::mt19937_64 rng(0x13579bdf2468ace0ULL + n);
    std::vector<u32> a(n), bit(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        u32 v = (u32)(rng() % 1024);
        a[i] = v;
        bit_apply_u32(bit.data(), n, i, v);
    }
    for (size_t i = 0; i < 4000; ++i) {
        size_t p = (size_t)(rng() % n);
        u32 x = brute_prefix_u32(a.data(), p);
        u32 y = bit_prefix_u32(bit.data(), p);
        if (x != y) {
            std::printf("VERIFY FAIL sum32 query n=%zu i=%zu p=%zu brute=%u bit=%u\n",
                        n, i, p, (unsigned)x, (unsigned)y);
            std::abort();
        }
    }
}

static void verify_mixed_u32(size_t n) {
    std::mt19937_64 rng(0x13579bdf2468ace0ULL + n + 0x1000);
    std::vector<u32> a(n), bit(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        u32 v = (u32)(rng() % 1024);
        a[i] = v;
        bit_apply_u32(bit.data(), n, i, v);
    }
    for (size_t i = 0; i < 4000; ++i) {
        size_t p = (size_t)(rng() % n);
        u32 d = (u32)(rng() % 1024);
        if ((i & 1) != 0) {
            a[p] += d;
            bit_apply_u32(bit.data(), n, p, d);
        } else {
            u32 x = brute_prefix_u32(a.data(), p);
            u32 y = bit_prefix_u32(bit.data(), p);
            if (x != y) {
                std::printf("VERIFY FAIL sum32 mixed n=%zu i=%zu p=%zu brute=%u bit=%u\n",
                            n, i, p, (unsigned)x, (unsigned)y);
                std::abort();
            }
        }
    }
}

static void verify_range_u32(size_t n) {
    for (size_t L : {1u, 2u, 3u, 7u, 16u, 64u, 257u, 1024u}) {
        std::mt19937_64 rng(0x13579bdf2468ace0ULL + n + L);
        std::vector<u32> a(n), bit(n + 1, 0);
        for (size_t i = 0; i < n; ++i) {
            u32 v = (u32)(rng() % 1024);
            a[i] = v;
            bit_apply_u32(bit.data(), n, i, v);
        }
        for (size_t i = 0; i < 500; ++i) {
            size_t l = (size_t)(rng() % (n - L + 1));
            u32 x = 0;
            for (size_t j = l; j < l + L; ++j) x += a[j];
            u32 y = bit_prefix_u32(bit.data(), l + L - 1) - bit_prefix_u32(bit.data(), l - 1);
            if (x != y) {
                std::printf("VERIFY FAIL sum32 range L=%zu i=%zu l=%zu brute=%u bit=%u\n",
                            L, i, l, (unsigned)x, (unsigned)y);
                std::abort();
            }
        }
    }
}

static void measure_u32_query() {
    std::mt19937_64 rng(0x5a5a5a5a5a5a5a5aULL);
    for (size_t n : all_n()) {
        verify_query_u32(n);
        std::vector<u32> a(n), bit(n + 1, 0);
        for (size_t i = 0; i < n; ++i) {
            u32 v = (u32)(rng() % 1024);
            a[i] = v;
            bit_apply_u32(bit.data(), n, i, v);
        }

        size_t q0_brute = std::clamp(size_t(4'000'000) / std::max<size_t>(n / 2, 1),
                                     Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;
        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_brute, Mix::Query, rng, pos, delta, is_q);
            double ns = time_query_brute_u32(a.data(), pos);
            q_brute = clamp_q(ns / (double)q0_brute);
        }
        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_bit, Mix::Query, rng, pos, delta, is_q);
            double ns = time_query_bit_u32(bit.data(), pos);
            q_bit = clamp_q(ns / (double)q0_bit);
        }

        std::vector<size_t> pos_b, pos_t;
        std::vector<u64> del_b, del_t;
        std::vector<u8> isb, ist;
        gen_ops(n, q_brute, Mix::Query, rng, pos_b, del_b, isb);
        gen_ops(n, q_bit, Mix::Query, rng, pos_t, del_t, ist);

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(time_query_brute_u32(a.data(), pos_b));
            s_bit.push_back(time_query_bit_u32(bit.data(), pos_t));
        }
        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("sum32,query,%zu,%.3f,%.3f\n", n, t_brute, t_bit);
        std::fflush(stdout);
    }
}

static void measure_u32_mixed() {
    std::mt19937_64 rng(0x5a5a5a5a5a5a5a5aULL + 0x1000);
    for (size_t n : all_n()) {
        verify_mixed_u32(n);
        std::vector<u32> a(n), bit(n + 1, 0);

        size_t q0_brute = std::clamp(size_t(4'000'000) / std::max<size_t>(n / 2, 1),
                                     Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;
        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_brute, Mix::Mixed50, rng, pos, delta, is_q);
            double ns = time_mixed_brute_u32(a.data(), pos, delta, is_q);
            q_brute = clamp_q(ns / (double)q0_brute);
        }
        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_bit, Mix::Mixed50, rng, pos, delta, is_q);
            double ns = time_mixed_bit_u32(bit.data(), n, pos, delta, is_q);
            q_bit = clamp_q(ns / (double)q0_bit);
        }

        std::vector<size_t> pos_b, pos_t;
        std::vector<u64> del_b, del_t;
        std::vector<u8> isb, ist;
        gen_ops(n, q_brute, Mix::Mixed50, rng, pos_b, del_b, isb);
        gen_ops(n, q_bit, Mix::Mixed50, rng, pos_t, del_t, ist);

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(time_mixed_brute_u32(a.data(), pos_b, del_b, isb));
            s_bit.push_back(time_mixed_bit_u32(bit.data(), n, pos_t, del_t, ist));
        }
        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("sum32,mixed,%zu,%.3f,%.3f\n", n, t_brute, t_bit);
        std::fflush(stdout);
    }
}

static void measure_u32_range() {
    const size_t N = 1u << 20;
    verify_range_u32(N);
    std::mt19937_64 rng(0x5a5a5a5a5a5a5a5aULL + 0x2000);
    std::vector<u32> a(N), bit(N + 1, 0);
    for (size_t i = 0; i < N; ++i) {
        u32 v = (u32)(rng() % 1024);
        a[i] = v;
        bit_apply_u32(bit.data(), N, i, v);
    }
    for (size_t L : all_L()) {
        size_t q0_brute = std::clamp(size_t(4'000'000) / L, Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;
        {
            std::vector<size_t> ls(q0_brute);
            for (size_t& l : ls) l = (size_t)(rng() % (N - L + 1));
            double ns = time_range_brute_u32(a.data(), ls, L);
            q_brute = clamp_q(ns / (double)q0_brute);
        }
        {
            std::vector<size_t> ls(q0_bit);
            for (size_t& l : ls) l = (size_t)(rng() % (N - L + 1));
            double ns = time_range_bit_u32(bit.data(), ls, L);
            q_bit = clamp_q(ns / (double)q0_bit);
        }

        std::vector<size_t> ls_b(q_brute), ls_t(q_bit);
        for (size_t& l : ls_b) l = (size_t)(rng() % (N - L + 1));
        for (size_t& l : ls_t) l = (size_t)(rng() % (N - L + 1));

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(time_range_brute_u32(a.data(), ls_b, L));
            s_bit.push_back(time_range_bit_u32(bit.data(), ls_t, L));
        }
        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("sum32,range,%zu,%.3f,%.3f\n", L, t_brute, t_bit);
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    std::string which = argc > 1 ? argv[1] : "all";
    bool all = which == "all";

    if (all || which == "query") {
        measure_query<Op::Sum>(Op::Sum);
        measure_query<Op::Min>(Op::Min);
        measure_query<Op::Max>(Op::Max);
        measure_query<Op::And>(Op::And);
        measure_query<Op::Or>(Op::Or);
        measure_query<Op::Xor>(Op::Xor);
    }
    if (all || which == "mixed") {
        measure_mixed<Op::Sum>(Op::Sum);
        measure_mixed<Op::Xor>(Op::Xor);
    }
    if (all || which == "range") {
        measure_range<Op::Sum>(Op::Sum);
        measure_range<Op::Xor>(Op::Xor);
    }
    if (all || which == "bisect") {
        measure_bisect(true);
        measure_bisect(false);
    }
    if (all || which == "fraction") {
        measure_mixed<Op::Sum>(Op::Sum, Mix::Mixed25, "mixed_25");
        measure_mixed<Op::Sum>(Op::Sum, Mix::Mixed75, "mixed_75");
    }
    if (all || which == "tail") {
        measure_query<Op::Sum>(Op::Sum, Mix::QueryTail, "query_tail");
    }
    if (all || which == "sum32") {
        measure_u32_query();
        measure_u32_mixed();
        measure_u32_range();
    }
    if (which == "sum") {
        measure_query<Op::Sum>(Op::Sum);
        measure_mixed<Op::Sum>(Op::Sum);
        measure_range<Op::Sum>(Op::Sum);
    }
    if (which == "xor") {
        measure_query<Op::Xor>(Op::Xor);
        measure_mixed<Op::Xor>(Op::Xor);
        measure_range<Op::Xor>(Op::Xor);
    }
    if (which == "min") measure_query<Op::Min>(Op::Min);
    if (which == "max") measure_query<Op::Max>(Op::Max);
    if (which == "and") measure_query<Op::And>(Op::And);
    if (which == "or") measure_query<Op::Or>(Op::Or);
    return 0;
}
