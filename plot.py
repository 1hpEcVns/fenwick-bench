#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODES = [
    ("mixed", "mixed: 1:1 point-update + prefix-query"),
    ("query", "prefix queries only"),
    ("range", "range sums, fixed N = 2^20"),
]


def load(path):
    return pd.read_csv(path, names=["mode", "n", "brute_ns", "bit_ns"])


def boundary(sub):
    """Return (last size where brute wins, first size where BIT wins)."""
    faster = (sub["brute_ns"] < sub["bit_ns"]).to_numpy()
    xs = sub["n"].to_numpy()
    for i in range(len(xs) - 1):
        if faster[i] and not faster[i + 1]:
            return int(xs[i]), int(xs[i + 1])
    return None


def xlabel(mode):
    return "L (range length)" if mode == "range" else "n (array size)"


def plot_source(csv_path, out_prefix, title_label):
    df = load(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, (mode, label) in zip(axes, MODES):
        sub = df[df["mode"] == mode]
        ax.plot(sub["n"], sub["brute_ns"], marker="o", markersize=5,
                linewidth=1.5, color="#d62728", label="brute (plain array)")
        ax.plot(sub["n"], sub["bit_ns"], marker="s", markersize=5,
                linewidth=1.5, color="#1f77b4", label="Fenwick (BIT)")
        b = boundary(sub)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel(xlabel(mode))
        ax.set_ylabel("ns / op")
        ax.grid(True, which="both", alpha=0.25)
        if b is not None:
            ax.axvline(b[1], color="#1f77b4", ls=":", lw=1.2)
            ax.annotate(f"brute ≤ {b[0]} / BIT ≥ {b[1]}",
                        xy=(0.03, 0.90), xytext=(0.03, 0.90),
                        textcoords="axes fraction", color="#1f77b4", fontsize=10)
        else:
            ax.annotate("no crossover", xy=(0.03, 0.90), xytext=(0.03, 0.90),
                        textcoords="axes fraction", color="#7f7f7f", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"{title_label} — brute force vs Fenwick tree", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_prefix}_crossover.webp", dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (mode, label) in zip(axes2, MODES):
        sub = df[df["mode"] == mode]
        ax.plot(sub["n"], sub["bit_ns"] / sub["brute_ns"],
                marker="o", color="#1f77b4")
        ax.axhline(1.0, color="black", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(label)
        ax.set_xlabel(xlabel(mode))
        ax.set_ylabel("ratio (BIT / brute)")
        ax.grid(True, which="both", alpha=0.25)
    fig2.suptitle(f"{title_label} — ratio below 1 means brute is faster",
                  fontsize=14)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(f"{out_prefix}_ratio.webp", dpi=150)
    plt.close(fig2)

    print(f"\n{title_label} crossover (last brute-win / first BIT-win):")
    for mode, label in MODES:
        sub = df[df["mode"] == mode]
        b = boundary(sub)
        print(f"  {mode:6s}: " +
              (f"brute ≤ {b[0]} / BIT ≥ {b[1]}" if b else "no crossover"))


def plot_cpp_vs_rust():
    cpp = load("results.csv")
    rust = load("results_rs.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, (mode, label) in zip(axes, MODES):
        c = cpp[cpp["mode"] == mode]
        r = rust[rust["mode"] == mode]
        ax.plot(c["n"], c["bit_ns"], marker="s", linewidth=1.5,
                color="#1f77b4", label="C++23 BIT")
        ax.plot(r["n"], r["bit_ns"], marker="s", ls="--", linewidth=1.5,
                color="#d62728", label="Rust BIT")
        ax.plot(c["n"], c["brute_ns"], marker="o", linewidth=1.5,
                color="#7f7f7f", label="C++23 brute")
        ax.plot(r["n"], r["brute_ns"], marker="o", ls="--", linewidth=1.5,
                color="#ff7f0e", label="Rust brute")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel(xlabel(mode))
        ax.set_ylabel("ns / op")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("C++23 vs Rust (dashed = Rust)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("cpp_vs_rust.webp", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_source("results.csv", "cpp23",
                "C++23 (-O3 -march=native -std=c++23)")
    plot_source("results_rs.csv", "rust",
                "Rust (edition 2024, -O -C target-cpu=native)")
    plot_cpp_vs_rust()
