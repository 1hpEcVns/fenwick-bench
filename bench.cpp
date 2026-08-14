// Fenwick tree (BIT) vs plain-array brute force: find the crossover size.
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
using u8 = std::uint8_t;

static inline void black_box_u64(u64 x) {
    asm volatile("" : "+r"(x) : : "memory");
}

static inline double now_ns() {
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---------------- brute force: plain array ----------------
static inline u64 brute_prefix(const u64* a, size_t i) {
    u64 s = 0;
    for (size_t j = 0; j <= i; ++j) s += a[j];
    return s;
}

static inline void brute_update(u64* a, size_t i, u64 d) { a[i] += d; }

// ---------------- Fenwick tree (BIT), tree[1..n] ----------------
static inline u64 bit_prefix(const u64* bit, size_t i) {
    // i = SIZE_MAX means "before 0": i+1 wraps to 0 and the loop is skipped.
    u64 s = 0;
    size_t k = i + 1;
    while (k != 0) {
        s += bit[k];
        k &= k - 1;
    }
    return s;
}

static inline void bit_update(u64* bit, size_t n, size_t i, u64 d) {
    for (size_t k = i + 1; k <= n; k += k & (~k + 1)) bit[k] += d;
}

// ---------------- timed kernels ----------------
static u64 run_mixed_brute(u64* a, const size_t* pos, const u64* delta,
                           const u8* is_q, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += brute_prefix(a, pos[i]);
        else brute_update(a, pos[i], delta[i]);
    }
    return acc;
}

static u64 run_mixed_bit(u64* bit, size_t n, const size_t* pos,
                         const u64* delta, const u8* is_q, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        if (is_q[i]) acc += bit_prefix(bit, pos[i]);
        else bit_update(bit, n, pos[i], delta[i]);
    }
    return acc;
}

static u64 run_query_brute(const u64* a, const size_t* pos, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += brute_prefix(a, pos[i]);
    return acc;
}

static u64 run_query_bit(const u64* bit, const size_t* pos, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) acc += bit_prefix(bit, pos[i]);
    return acc;
}

static u64 run_range_brute(const u64* a, const size_t* ls, size_t L, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        size_t l = ls[i];
        u64 s = 0;
        for (size_t j = l; j < l + L; ++j) s += a[j];
        acc += s;
    }
    return acc;
}

static u64 run_range_bit(const u64* bit, const size_t* ls, size_t L, size_t q) {
    u64 acc = 0;
    for (size_t i = 0; i < q; ++i) {
        size_t l = ls[i];
        acc += bit_prefix(bit, l + L - 1) - bit_prefix(bit, l - 1);
    }
    return acc;
}

static double time_mixed_brute(u64* a, const std::vector<size_t>& pos,
                               const std::vector<u64>& delta,
                               const std::vector<u8>& is_q) {
    double t0 = now_ns();
    u64 acc = run_mixed_brute(a, pos.data(), delta.data(), is_q.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_mixed_bit(u64* bit, size_t n, const std::vector<size_t>& pos,
                             const std::vector<u64>& delta,
                             const std::vector<u8>& is_q) {
    double t0 = now_ns();
    u64 acc = run_mixed_bit(bit, n, pos.data(), delta.data(), is_q.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_query_brute(const u64* a, const std::vector<size_t>& pos) {
    double t0 = now_ns();
    u64 acc = run_query_brute(a, pos.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_query_bit(const u64* bit, const std::vector<size_t>& pos) {
    double t0 = now_ns();
    u64 acc = run_query_bit(bit, pos.data(), pos.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_range_brute(const u64* a, const std::vector<size_t>& ls,
                               size_t L) {
    double t0 = now_ns();
    u64 acc = run_range_brute(a, ls.data(), L, ls.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

static double time_range_bit(const u64* bit, const std::vector<size_t>& ls,
                             size_t L) {
    double t0 = now_ns();
    u64 acc = run_range_bit(bit, ls.data(), L, ls.size());
    double t1 = now_ns();
    black_box_u64(acc);
    return t1 - t0;
}

// ---------------- calibration helpers ----------------
static constexpr size_t ROUNDS = 9;
static constexpr double TARGET_NS = 3e6;  // ~3 ms per timed pass
static constexpr size_t Q_MIN = 256;
static constexpr size_t Q_MAX = 4'000'000;

static size_t clamp_q(double per_ns, size_t q0) {
    size_t q = (size_t)(TARGET_NS / per_ns);
    return std::clamp(q, Q_MIN, Q_MAX);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------- op generation ----------------
static void gen_ops(size_t n, size_t q, bool mixed, std::mt19937_64& rng,
                    std::vector<size_t>& pos, std::vector<u64>& delta,
                    std::vector<u8>& is_q) {
    pos.resize(q);
    delta.resize(q);
    is_q.resize(q);
    for (size_t i = 0; i < q; ++i) {
        pos[i] = (size_t)(rng() % n);
        delta[i] = rng() % 1024;
        is_q[i] = (!mixed || (i & 1) == 0) ? 1 : 0;
    }
}

// ---------------- verification ----------------
static void verify_n(size_t n, bool mixed) {
    std::mt19937_64 rng(0xd1b54a32d192ed03ULL);
    std::vector<u64> a(n), bit(n + 1);
    for (size_t i = 0; i < n; ++i) {
        u64 v = rng() % 1024;
        a[i] = v;
        bit_update(bit.data(), n, i, v);
    }
    for (size_t i = 0; i < 4000; ++i) {
        size_t p = (size_t)(rng() % n);
        u64 d = rng() % 1024;
        if (mixed && (i & 1) != 0) {
            brute_update(a.data(), p, d);
            bit_update(bit.data(), n, p, d);
        } else {
            u64 x = brute_prefix(a.data(), p);
            u64 y = bit_prefix(bit.data(), p);
            if (x != y) {
                std::printf("VERIFY FAIL mixed=%d n=%zu i=%zu p=%zu brute=%llu bit=%llu\n",
                            (int)mixed, n, i, p,
                            (unsigned long long)x, (unsigned long long)y);
                std::abort();
            }
        }
    }
}

static void verify_range(size_t n) {
    for (size_t L : {1u, 2u, 3u, 7u, 16u, 64u, 257u, 1024u}) {
        std::mt19937_64 rng(0xa0761d6478bd642fULL + L);
        std::vector<u64> a(n), bit(n + 1);
        for (size_t i = 0; i < n; ++i) {
            u64 v = rng() % 1024;
            a[i] = v;
            bit_update(bit.data(), n, i, v);
        }
        for (size_t i = 0; i < 500; ++i) {
            size_t l = (size_t)(rng() % (n - L + 1));
            u64 x = 0;
            for (size_t j = l; j < l + L; ++j) x += a[j];
            u64 y = bit_prefix(bit.data(), l + L - 1) - bit_prefix(bit.data(), l - 1);
            if (x != y) {
                std::printf("VERIFY FAIL range L=%zu i=%zu l=%zu brute=%llu bit=%llu\n",
                            L, i, l, (unsigned long long)x, (unsigned long long)y);
                std::abort();
            }
        }
    }
}

// ---------------- n sweep: mixed / query-only ----------------
static const std::vector<size_t>& all_n() {
    static const std::vector<size_t> v = {
        4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 96,
        128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
        8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072,
        196608, 262144, 393216, 524288, 786432, 1048576
    };
    return v;
}

static void measure_n_sweep(bool mixed) {
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    const char* mode = mixed ? "mixed" : "query";
    for (size_t n : all_n()) {
        verify_n(n, mixed);
        std::vector<u64> a(n), bit(n + 1);

        size_t q0_brute = std::clamp(size_t(4'000'000) / std::max<size_t>(n / 2, 1),
                                     Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;

        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_brute, mixed, rng, pos, delta, is_q);
            double ns = mixed ? time_mixed_brute(a.data(), pos, delta, is_q)
                              : time_query_brute(a.data(), pos);
            q_brute = clamp_q(ns / (double)q0_brute, q0_brute);
        }
        {
            std::vector<size_t> pos;
            std::vector<u64> delta;
            std::vector<u8> is_q;
            gen_ops(n, q0_bit, mixed, rng, pos, delta, is_q);
            double ns = mixed ? time_mixed_bit(bit.data(), n, pos, delta, is_q)
                              : time_query_bit(bit.data(), pos);
            q_bit = clamp_q(ns / (double)q0_bit, q0_bit);
        }

        std::vector<size_t> pos_b, pos_t;
        std::vector<u64> del_b, del_t;
        std::vector<u8> isb, ist;
        gen_ops(n, q_brute, mixed, rng, pos_b, del_b, isb);
        gen_ops(n, q_bit, mixed, rng, pos_t, del_t, ist);

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(mixed ? time_mixed_brute(a.data(), pos_b, del_b, isb)
                                    : time_query_brute(a.data(), pos_b));
            s_bit.push_back(mixed ? time_mixed_bit(bit.data(), n, pos_t, del_t, ist)
                                  : time_query_bit(bit.data(), pos_t));
        }

        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("%s,%zu,%.3f,%.3f\n", mode, n, t_brute, t_bit);
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

static void measure_range() {
    const size_t N = 1u << 20;
    verify_range(N);
    std::mt19937_64 rng(0x243f6a8885a308d3ULL);
    std::vector<u64> a(N), bit(N + 1);
    for (size_t i = 0; i < N; ++i) {
        u64 v = rng() % 1024;
        a[i] = v;
        bit_update(bit.data(), N, i, v);
    }

    for (size_t L : all_L()) {
        size_t q0_brute = std::clamp(size_t(4'000'000) / L, Q_MIN, size_t(262144));
        size_t q0_bit = 131072;
        size_t q_brute, q_bit;

        {
            std::vector<size_t> ls(q0_brute);
            for (size_t& l : ls) l = (size_t)(rng() % (N - L + 1));
            double ns = time_range_brute(a.data(), ls, L);
            q_brute = clamp_q(ns / (double)q0_brute, q0_brute);
        }
        {
            std::vector<size_t> ls(q0_bit);
            for (size_t& l : ls) l = (size_t)(rng() % (N - L + 1));
            double ns = time_range_bit(bit.data(), ls, L);
            q_bit = clamp_q(ns / (double)q0_bit, q0_bit);
        }

        std::vector<size_t> ls_b(q_brute), ls_t(q_bit);
        for (size_t& l : ls_b) l = (size_t)(rng() % (N - L + 1));
        for (size_t& l : ls_t) l = (size_t)(rng() % (N - L + 1));

        std::vector<double> s_brute, s_bit;
        for (size_t r = 0; r < ROUNDS; ++r) {
            s_brute.push_back(time_range_brute(a.data(), ls_b, L));
            s_bit.push_back(time_range_bit(bit.data(), ls_t, L));
        }

        double t_brute = median(s_brute) / (double)q_brute;
        double t_bit = median(s_bit) / (double)q_bit;
        std::printf("range,%zu,%.3f,%.3f\n", L, t_brute, t_bit);
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    std::string which = argc > 1 ? argv[1] : "all";
    if (which == "mixed" || which == "all") measure_n_sweep(true);
    if (which == "query" || which == "all") measure_n_sweep(false);
    if (which == "range" || which == "all") measure_range();
    return 0;
}
