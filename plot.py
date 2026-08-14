#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OPS = ["sum", "min", "max", "and", "or", "xor"]
DYNAMIC = [("sum", "mixed"), ("sum", "range"), ("xor", "mixed"), ("xor", "range")]
BISECT = [("sum_bisect", "mixed"), ("sum_bisect", "query")]
FRACTIONS = [("sum", "mixed_25"), ("sum", "mixed"), ("sum", "mixed_75")]
SUM32 = [("sum32", "query"), ("sum32", "mixed"), ("sum32", "range")]


def load(path):
    return pd.read_csv(path, names=["op", "mode", "n", "brute_ns", "bit_ns"])


def boundary(sub):
    """Return (last size where brute wins, first size where BIT wins)."""
    faster = (sub["brute_ns"] < sub["bit_ns"]).to_numpy()
    xs = sub["n"].to_numpy()
    for i in range(len(xs) - 1):
        if faster[i] and not faster[i + 1]:
            return int(xs[i]), int(xs[i + 1])
    return None


def who_wins(sub):
    """'brute' or 'BIT' based on the first measured size."""
    if sub.empty:
        return "?"
    return "brute" if (sub["brute_ns"] < sub["bit_ns"]).iloc[0] else "BIT"


def xlabel(mode):
    return "L (range length)" if mode == "range" else "n (array size)"


def draw_curves(ax, df, op, mode, curves):
    """curves: list of (col, label, color, marker, linestyle)."""
    sub = df[(df["op"] == op) & (df["mode"] == mode)]
    if sub.empty:
        return None
    for col, label, color, marker, ls in curves:
        ax.plot(sub["n"], sub[col], marker=marker, markersize=4,
                linewidth=1.3, color=color, label=label, linestyle=ls)
    b = boundary(sub)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(f"{op} / {mode}")
    ax.set_xlabel(xlabel(mode))
    ax.set_ylabel("ns / op")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    if b is not None:
        ax.annotate(f"brute ≤ {b[0]} / BIT ≥ {b[1]}",
                    xy=(0.03, 0.90), xytext=(0.03, 0.90),
                    textcoords="axes fraction", color="#1f77b4", fontsize=9)
    else:
        ax.annotate(f"no crossover ({who_wins(sub)} always faster)",
                    xy=(0.03, 0.90), xytext=(0.03, 0.90),
                    textcoords="axes fraction", color="#7f7f7f", fontsize=9)
    return b


def plot_source(csv_path, out_prefix, title_label):
    df = load(csv_path)
    curves = [
        ("brute_ns", "brute", "#d62728", "o", "-"),
        ("bit_ns", "BIT", "#1f77b4", "s", "-"),
    ]

    # query mode: 6 ops
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, op in zip(axes.flat, OPS):
        draw_curves(ax, df, op, "query", curves)
    fig.suptitle(f"{title_label} — prefix-query brute vs BIT (all ops)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out_prefix}_query.webp", dpi=150)
    plt.close(fig)

    # dynamic modes: sum/xor mixed+range
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (op, mode) in zip(axes.flat, DYNAMIC):
        draw_curves(ax, df, op, mode, curves)
    fig.suptitle(f"{title_label} — dynamic modes (mixed 1:1, range N=2^20)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out_prefix}_dynamic.webp", dpi=150)
    plt.close(fig)

    # ratio figures
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, op in zip(axes.flat, OPS):
        sub = df[(df["op"] == op) & (df["mode"] == "query")]
        ax.plot(sub["n"], sub["bit_ns"] / sub["brute_ns"], marker="o",
                markersize=4, color="#1f77b4")
        ax.axhline(1.0, color="black", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{op} / query")
        ax.set_xlabel(xlabel("query"))
        ax.set_ylabel("ratio (BIT / brute)")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(f"{title_label} — query-mode ratio, below 1 = brute faster",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out_prefix}_ratio_query.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (op, mode) in zip(axes.flat, DYNAMIC):
        sub = df[(df["op"] == op) & (df["mode"] == mode)]
        ax.plot(sub["n"], sub["bit_ns"] / sub["brute_ns"], marker="o",
                markersize=4, color="#1f77b4")
        ax.axhline(1.0, color="black", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{op} / {mode}")
        ax.set_xlabel(xlabel(mode))
        ax.set_ylabel("ratio (BIT / brute)")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(f"{title_label} — dynamic-mode ratio, below 1 = brute faster",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out_prefix}_ratio_dynamic.webp", dpi=150)
    plt.close(fig)

    # BIT binary search (树状数组上二分): sum only
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (op, mode) in zip(axes, BISECT):
        draw_curves(ax, df, op, mode, curves)
    fig.suptitle(f"{title_label} — BIT binary search (sum, prefix >= k)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_prefix}_bisect.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (op, mode) in zip(axes, BISECT):
        sub = df[(df["op"] == op) & (df["mode"] == mode)]
        ax.plot(sub["n"], sub["bit_ns"] / sub["brute_ns"], marker="o",
                markersize=4, color="#1f77b4")
        ax.axhline(1.0, color="black", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{op} / {mode}")
        ax.set_xlabel("n (array size)")
        ax.set_ylabel("ratio (BIT / brute)")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(f"{title_label} — BIT-bisect ratio, below 1 = brute faster",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_prefix}_ratio_bisect.webp", dpi=150)
    plt.close(fig)

    # update-fraction sweep (sum)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (op, mode) in zip(axes, FRACTIONS):
        draw_curves(ax, df, op, mode, curves)
        ax.set_title(f"sum / {mode}")
    fig.suptitle(f"{title_label} — sum mixed update fraction (25% / 50% / 75%)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_prefix}_fractions.webp", dpi=150)
    plt.close(fig)

    # tail-heavy queries (sum)
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_curves(ax, df, "sum", "query_tail", curves)
    fig.suptitle(f"{title_label} — sum tail-heavy prefix queries (p in [0.9n, n))",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_prefix}_tail.webp", dpi=150)
    plt.close(fig)

    # u32 element type (sum)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (op, mode) in zip(axes, SUM32):
        draw_curves(ax, df, op, mode, curves)
        ax.set_title(f"sum32 / {mode}")
    fig.suptitle(f"{title_label} — u32 sum (query / mixed / range)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_prefix}_sum32.webp", dpi=150)
    plt.close(fig)

    # ratio figures for the new families
    for name, combos, title in [
        ("fractions", FRACTIONS, "update fraction"),
        ("tail", [("sum", "query_tail")], "tail-heavy queries"),
        ("sum32", SUM32, "u32 sum"),
    ]:
        fig, axes = plt.subplots(1, len(combos), figsize=(8 * len(combos), 5))
        if len(combos) == 1:
            axes = [axes]
        for ax, (op, mode) in zip(axes, combos):
            sub = df[(df["op"] == op) & (df["mode"] == mode)]
            ax.plot(sub["n"], sub["bit_ns"] / sub["brute_ns"], marker="o",
                    markersize=4, color="#1f77b4")
            ax.axhline(1.0, color="black", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{op} / {mode}")
            ax.set_xlabel(xlabel(mode))
            ax.set_ylabel("ratio (BIT / brute)")
            ax.grid(True, which="both", alpha=0.25)
        fig.suptitle(f"{title_label} — {title} ratio, below 1 = brute faster",
                     fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(f"{out_prefix}_ratio_{name}.webp", dpi=150)
        plt.close(fig)

    print(f"\n{title_label} crossover (last brute-win / first BIT-win):")
    for op in OPS:
        b = boundary(df[(df["op"] == op) & (df["mode"] == "query")])
        print(f"  {op:4s} query: " +
              (f"brute ≤ {b[0]} / BIT ≥ {b[1]}" if b
               else f"no crossover ({who_wins(df[(df['op'] == op) & (df['mode'] == 'query')])} always faster)"))
    for op, mode in DYNAMIC:
        b = boundary(df[(df["op"] == op) & (df["mode"] == mode)])
        print(f"  {op:4s} {mode}: " +
              (f"brute ≤ {b[0]} / BIT ≥ {b[1]}" if b
               else f"no crossover ({who_wins(df[(df['op'] == op) & (df['mode'] == mode)])} always faster)"))
    for op, mode in BISECT:
        b = boundary(df[(df["op"] == op) & (df["mode"] == mode)])
        print(f"  {op:10s} {mode}: " +
              (f"brute ≤ {b[0]} / BIT ≥ {b[1]}" if b
               else f"no crossover ({who_wins(df[(df['op'] == op) & (df['mode'] == mode)])} always faster)"))
    for op, mode in FRACTIONS:
        b = boundary(df[(df["op"] == op) & (df["mode"] == mode)])
        print(f"  {op:4s} {mode}: " +
              (f"brute ≤ {b[0]} / BIT ≥ {b[1]}" if b
               else f"no crossover ({who_wins(df[(df['op'] == op) & (df['mode'] == mode)])} always faster)"))
    for op, mode in [("sum", "query_tail")] + SUM32:
        b = boundary(df[(df["op"] == op) & (df["mode"] == mode)])
        print(f"  {op:8s} {mode}: " +
              (f"brute ≤ {b[0]} / BIT ≥ {b[1]}" if b
               else f"no crossover ({who_wins(df[(df['op'] == op) & (df['mode'] == mode)])} always faster)"))


def plot_cpp_vs_rust():
    cpp = load("results.csv")
    rust = load("results_rs.csv")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, op in zip(axes.flat, OPS):
        draw_curves(ax, cpp, op, "query", [
            ("bit_ns", "C++23 BIT", "#1f77b4", "s", "-"),
            ("brute_ns", "C++23 brute", "#7f7f7f", "o", "-"),
        ])
        draw_curves(ax, rust, op, "query", [
            ("bit_ns", "Rust BIT", "#d62728", "s", "--"),
            ("brute_ns", "Rust brute", "#ff7f0e", "o", "--"),
        ])
    fig.suptitle("C++23 vs Rust — prefix-query (dashed = Rust)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("cpp_vs_rust_query.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (op, mode) in zip(axes.flat, DYNAMIC):
        draw_curves(ax, cpp, op, mode, [
            ("bit_ns", "C++23 BIT", "#1f77b4", "s", "-"),
            ("brute_ns", "C++23 brute", "#7f7f7f", "o", "-"),
        ])
        draw_curves(ax, rust, op, mode, [
            ("bit_ns", "Rust BIT", "#d62728", "s", "--"),
            ("brute_ns", "Rust brute", "#ff7f0e", "o", "--"),
        ])
    fig.suptitle("C++23 vs Rust — dynamic modes (dashed = Rust)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("cpp_vs_rust_dynamic.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (op, mode) in zip(axes, BISECT):
        draw_curves(ax, cpp, op, mode, [
            ("bit_ns", "C++23 BIT", "#1f77b4", "s", "-"),
            ("brute_ns", "C++23 brute", "#7f7f7f", "o", "-"),
        ])
        draw_curves(ax, rust, op, mode, [
            ("bit_ns", "Rust BIT", "#d62728", "s", "--"),
            ("brute_ns", "Rust brute", "#ff7f0e", "o", "--"),
        ])
    fig.suptitle("C++23 vs Rust — BIT binary search (dashed = Rust)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("cpp_vs_rust_bisect.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (op, mode) in zip(axes, FRACTIONS):
        draw_curves(ax, cpp, op, mode, [
            ("bit_ns", "C++23 BIT", "#1f77b4", "s", "-"),
            ("brute_ns", "C++23 brute", "#7f7f7f", "o", "-"),
        ])
        draw_curves(ax, rust, op, mode, [
            ("bit_ns", "Rust BIT", "#d62728", "s", "--"),
            ("brute_ns", "Rust brute", "#ff7f0e", "o", "--"),
        ])
        ax.set_title(f"sum / {mode}")
    fig.suptitle("C++23 vs Rust — update-fraction sweep (dashed = Rust)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("cpp_vs_rust_fractions.webp", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (op, mode) in zip(axes, SUM32):
        draw_curves(ax, cpp, op, mode, [
            ("bit_ns", "C++23 BIT", "#1f77b4", "s", "-"),
            ("brute_ns", "C++23 brute", "#7f7f7f", "o", "-"),
        ])
        draw_curves(ax, rust, op, mode, [
            ("bit_ns", "Rust BIT", "#d62728", "s", "--"),
            ("brute_ns", "Rust brute", "#ff7f0e", "o", "--"),
        ])
        ax.set_title(f"sum32 / {mode}")
    fig.suptitle("C++23 vs Rust — u32 sum (dashed = Rust)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("cpp_vs_rust_sum32.webp", dpi=150)
    plt.close(fig)


def plot_best_family(df_a, df_b, combos, fname, title,
                     label_a="C++23 best", label_b="Rust best",
                     ratio_label="Rust / C++ best"):
    """Per language take the faster of brute/BIT as 'best', then compare."""
    n = len(combos)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8.5))
    if n == 1:
        axes = axes.reshape(2, 1)
    for col, (op, mode) in enumerate(combos):
        c = df_a[(df_a["op"] == op) & (df_a["mode"] == mode)]
        r = df_b[(df_b["op"] == op) & (df_b["mode"] == mode)]
        cb = c[["brute_ns", "bit_ns"]].min(axis=1)
        rb = r[["brute_ns", "bit_ns"]].min(axis=1)

        ax = axes[0, col]
        ax.plot(c["n"], cb, marker="o", markersize=4, linewidth=1.3,
                color="#1f77b4", label=label_a)
        ax.plot(r["n"], rb, marker="s", markersize=4, linewidth=1.3,
                ls="--", color="#d62728", label=label_b)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{op} / {mode}")
        ax.set_xlabel(xlabel(mode))
        if col == 0:
            ax.set_ylabel("ns / op (best)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        m = c.merge(r, on="n", suffixes=("_c", "_r"))
        if not m.empty:
            cbest = m[["brute_ns_c", "bit_ns_c"]].min(axis=1)
            rbest = m[["brute_ns_r", "bit_ns_r"]].min(axis=1)
            ax.plot(m["n"], rbest / cbest, marker="o", markersize=4,
                    color="#7f7f7f")
            ax.axhline(1.0, color="black", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel(mode))
        if col == 0:
            ax.set_ylabel(ratio_label)
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def best_stats(cpp, rust, combos):
    print("\nBest-of (min(brute, BIT)): Rust vs C++23")
    print(f"{'op':<10s} {'mode':<12s} {'geomean R/C':>12s} {'Rust-win%':>10s}")
    rows = []
    for op, mode in combos:
        c = cpp[(cpp["op"] == op) & (cpp["mode"] == mode)]
        r = rust[(rust["op"] == op) & (rust["mode"] == mode)]
        m = c.merge(r, on="n", suffixes=("_c", "_r"))
        if m.empty:
            continue
        cbest = m[["brute_ns_c", "bit_ns_c"]].min(axis=1)
        rbest = m[["brute_ns_r", "bit_ns_r"]].min(axis=1)
        ratio = rbest / cbest
        gmean = float(np.exp(np.log(ratio).mean()))
        win = float((ratio < 1.0).mean())
        rows.append((op, mode, gmean, win))
        print(f"{op:<10s} {mode:<12s} {gmean:>12.3f} {win * 100:>9.0f}%")
    return rows


def plot_std_vs_bare():
    """Third comparison: std Rust vs bare no_std Rust (best-of)."""
    std = load("results_rs.csv")
    bare = load("results_bare.csv")
    families = [
        ([(op, "query") for op in OPS], "std_vs_bare_query.webp",
         "Best-of brute/BIT: std Rust vs bare no_std Rust (query) + ratio"),
        (DYNAMIC, "std_vs_bare_dynamic.webp",
         "Best-of brute/BIT: std Rust vs bare no_std Rust (dynamic) + ratio"),
        (BISECT, "std_vs_bare_bisect.webp",
         "Best-of brute/BIT: std Rust vs bare no_std Rust (BIT bisect) + ratio"),
        (FRACTIONS, "std_vs_bare_fractions.webp",
         "Best-of brute/BIT: std Rust vs bare no_std Rust (fractions) + ratio"),
        ([("sum", "query_tail")], "std_vs_bare_tail.webp",
         "Best-of brute/BIT: std Rust vs bare no_std Rust (tail) + ratio"),
        (SUM32, "std_vs_bare_sum32.webp",
         "Best-of brute/BIT: std Rust vs bare no_std Rust (u32 sum) + ratio"),
    ]
    all_combos = []
    for combos, fname, title in families:
        plot_best_family(std, bare, combos, fname, title,
                         label_a="std Rust best", label_b="bare Rust best",
                         ratio_label="bare / std best")
        all_combos.extend(combos)

    print("\nBest-of (min(brute, BIT)): std Rust vs bare no_std Rust")
    print(f"{'op':<10s} {'mode':<12s} {'geomean bare/std':>16s} {'bare-win%':>10s}")
    for op, mode in all_combos:
        s = std[(std["op"] == op) & (std["mode"] == mode)]
        b = bare[(bare["op"] == op) & (bare["mode"] == mode)]
        m = s.merge(b, on="n", suffixes=("_s", "_b"))
        if m.empty:
            continue
        sbest = m[["brute_ns_s", "bit_ns_s"]].min(axis=1)
        bbest = m[["brute_ns_b", "bit_ns_b"]].min(axis=1)
        ratio = bbest / sbest
        gmean = float(np.exp(np.log(ratio).mean()))
        win = float((ratio < 1.0).mean())
        print(f"{op:<10s} {mode:<12s} {gmean:>16.3f} {win * 100:>9.0f}%")


if __name__ == "__main__":
    plot_source("results.csv", "cpp23",
                "C++23 (-O3 -march=native -std=c++23)")
    plot_source("results_rs.csv", "rust",
                "Rust (edition 2024, -O -C target-cpu=native)")
    plot_cpp_vs_rust()

    cpp = load("results.csv")
    rust = load("results_rs.csv")
    families = [
        ([(op, "query") for op in OPS], "best_query.webp",
         "Best-of brute/BIT: C++23 vs Rust (prefix queries, top) + ratio (bottom)"),
        (DYNAMIC, "best_dynamic.webp",
         "Best-of brute/BIT: C++23 vs Rust (dynamic modes) + ratio"),
        (BISECT, "best_bisect.webp",
         "Best-of brute/BIT: C++23 vs Rust (BIT binary search) + ratio"),
        (FRACTIONS, "best_fractions.webp",
         "Best-of brute/BIT: C++23 vs Rust (update-fraction sweep) + ratio"),
        ([("sum", "query_tail")], "best_tail.webp",
         "Best-of brute/BIT: C++23 vs Rust (tail-heavy queries) + ratio"),
        (SUM32, "best_sum32.webp",
         "Best-of brute/BIT: C++23 vs Rust (u32 sum) + ratio"),
    ]
    all_combos = []
    for combos, fname, title in families:
        plot_best_family(cpp, rust, combos, fname, title)
        all_combos.extend(combos)
    best_stats(cpp, rust, all_combos)
    plot_std_vs_bare()
