# 直接暴力 vs 树状数组（BIT）：半群操作临界点

对同一个前缀/区间查询集合，对比两种实现：

- **brute（直接暴力）**：普通数组 + 普通 `for` 循环；不手写 intrinsics。
- **BIT（树状数组）**：1-indexed `tree[1..n]`，查询/合并用 `lowbit` 循环。

覆盖 6 种半群操作（都是 `u64`）：

| 操作 | 单位元 | 是否可逆（支持修改） |
| --- | --- | --- |
| sum | 0 | 是 |
| xor | 0 | 是 |
| min | UINT64_MAX | 否 |
| max | 0 | 否 |
| and | UINT64_MAX | 否 |
| or | 0 | 否 |

测量模式：

| 模式 | 含义 | 适用操作 |
| --- | --- | --- |
| `query` | 纯前缀查询，随机位置 | 全部 6 种 |
| `mixed` | 1:1 交错的单点修改 + 前缀查询 | 仅 sum / xor（可逆） |
| `range` | 固定 N=2^20 的区间查询，长度 L | 仅 sum / xor（可逆） |
| `bisect` | 树状数组上二分：求前缀和首次 ≥ k 的位置（值非负） | 仅 sum |

`bisect`（罕见但经典）的两种实现：暴力从 0 开始累加到 ≥ k（O(n)）；BIT 用
二进制上跳（最高位开始，`bit[next] < k` 则跳，O(log n)）。查询 k 在
`[1, n*512]` 均匀随机，带非负随机增量；`mixed` 为 1:1 修改+查询，
`query` 为纯查询。

## 汇编验证：这些操作有没有自动向量化？

源码全部是普通循环。`bash asm_check.sh`（CXX/RUSTC 可用 `nix develop` 提供）
把每个操作的最小前缀循环分别用 GCC 15.3 和 LLVM 编译（`-O3 -march=native` /
`--edition=2024 -O -C target-cpu=native`），统计对应 SIMD 指令出现次数：

| 操作 | 期望指令 | GCC# | LLVM# |
| --- | --- | ---: | ---: |
| sum | `vpaddq` | 3 | 12 |
| min | `vpminuq` / `vpcmpgtq` | 0 / 6 | 0 / 24 |
| max | `vpmaxuq` / `vpcmpgtq` | 0 / 6 | 0 / 24 |
| and | `vpand` | 3 | 12 |
| or | `vpor` | 3 | 12 |
| xor | `vpxor` | 9 | 64 |
| sum32 | `vpaddd` | 7 | 13 |

min/max 两种编译器都没用 `vpminuq/vpmaxuq`，而是用符号位技巧 +
`vpcmpgtq` + blend 实现；`vpaddd` 表示 u32 累加（8 路/向量）。

**Rust 热循环的边界检查（unchecked 前后）**：最初的 Rust 版本在 BIT 查询/
修改、暴力二分扫描的热循环里，每个元素都残留一次边界检查（LLVM 无法从
`k ≤ n` 推出 `k < len`）：

```asm
; 修改前：每步多一次 cmp + jae panic
190d0:	cmp    %rsi,%rcx
190d3:	jae    190e8            ; -> slice index panic
190d5:	xor    (%rdi,%rcx,8),%r9
```

把 BIT/更新/二分循环改成 `get_unchecked`（不变量：`k ∈ [1,n]`、`i < n`、
`j < n` 都由循环条件保证）后，热循环只剩必要的比较：

```asm
; 修改后：直接 load + blsr，无边界检查
190e0:	xor    (%rdi,%r8,8),%r9
190e4:	blsr   %r8,%r8
190e9:	jne    190e0
```

C++ 用裸指针天然没有这个问题。这也解释了为什么 unchecked 前后“Rust vs C++”
结论会翻转（见下方最优方案对比）。

之前针对 sum 的详细结论（同样适用于其它操作）：**两种编译器都会自动生成
mm256 指令**（GCC 单 ymm 累加器 + `vextracti128` 归约；LLVM 4 个 ymm 累加器、
每轮 16 个 u64）。因此不需要手写 intrinsics；实测手写 4 累加器 mm256 版在
mixed 模式反而慢 7–10%，大 L 区间和慢 10–17%，所以保留普通循环源码。

## 复现

```bash
nix develop            # gcc/rustc/python+matplotlib
make bench-all         # C++23 results.csv + Rust results_rs.csv
make plot              # 重测 + 生成图并输出临界点
bash asm_check.sh      # 自动向量化检查（上面那张表）
```

编译/测量方法：

- C++23：`g++ -O3 -march=native -std=c++23`
- Rust：`rustc --edition=2024 -O -C target-cpu=native`
- 固定单个 P-core：`taskset -c 0`；C++ 和 Rust 串行跑，避免抢核
- 每个方法先校准到约 3 ms/轮的查询数，再跑 9 轮取中位数
- 每个 (op, mode, n) 测量前先用随机操作流交叉验证 brute 与 BIT 结果一致
  （不通过直接 abort）
- 位置用 mt19937_64（C++）/ splitmix64 高 32 位（Rust）生成，避免 LCG 在
  2 的幂 n 上出现周期坏数据

## 本机结果（i9-13950HX，g++ 15.3，rustc 1.94.1）

“临界点”指：`暴力赢的最后一个大小 / BIT 赢的第一个大小`。

| 操作 / 模式 | C++23 | Rust edition 2024 |
| --- | ---: | ---: |
| sum / mixed（50% 修改） | 暴力 ≤ 8 / BIT ≥ 12 | 无临界点（BIT 一直更快） |
| sum / mixed_25（25% 修改） | 无临界点（BIT 一直更快） | 同左 |
| sum / mixed_75（75% 修改） | 暴力 ≤ 512 / BIT ≥ 768 | 暴力 ≤ 1024 / BIT ≥ 1536 |
| sum / query | 无临界点（BIT 一直更快） | 同左 |
| sum / query_tail（p∈[0.9n,n)） | 无临界点（BIT 一直更快） | 同左 |
| sum / range | 暴力 ≤ 192 / BIT ≥ 256 | 暴力 ≤ 192 / BIT ≥ 256 |
| xor / mixed | 暴力 ≤ 8 / BIT ≥ 12 | 暴力 ≤ 8 / BIT ≥ 12 |
| xor / query | 无临界点（BIT 一直更快） | 同左 |
| xor / range | 暴力 ≤ 192 / BIT ≥ 256 | 暴力 ≤ 192 / BIT ≥ 256 |
| sum_bisect / mixed | 暴力 ≤ 768 / BIT ≥ 1024 | 暴力 ≤ 512 / BIT ≥ 768 |
| sum_bisect / query | 无临界点（BIT 一直更快） | 同左 |
| sum32 / query | 无临界点（BIT 一直更快） | 同左 |
| sum32 / mixed | 暴力 ≤ 8 / BIT ≥ 12 | 无临界点（BIT 一直更快） |
| sum32 / range | 暴力 ≤ 256 / BIT ≥ 384 | 暴力 ≤ 256 / BIT ≥ 384 |
| min / query | 无临界点（BIT 一直更快） | 同左 |
| max / query | 无临界点（BIT 一直更快） | 同左 |
| and / query | 无临界点（BIT 一直更快） | 同左 |
| or / query | 无临界点（BIT 一直更快） | 同左 |

补充比较：

- **修改比例**：25% 修改时暴力的 O(1) 更新优势太小，n≥4 起 BIT 一直更快；
  50% 时 C++ 临界点 8/12、Rust（unchecked 后）无临界点；75% 时暴力能赢到
  512–1024。
- **尾部偏置**（查询位置集中在 [0.9n, n)）：暴力扫描几乎总是全长，n≥4 起
  BIT 一直更快。
- **u32 类型**：query 仍无临界点；range 临界点比 u64 略靠后（u32 暴力 8 路
  向量化更划算）。
- **树状数组上二分**：纯查询下 BIT 从 n=4 起一直更快；带 1:1 修改时暴力能赢到
  512–1024（值在一轮内持续增长会缩短暴力扫描，因此该临界点是偏向暴力的
  上界；k 在 [1, n*512] 均匀分布）。

直观结论：

- 6 种半群操作 + u32 在两种编译器下都自动向量化，纯前缀查询没有暴力空间。
- 暴力只在小 n + 修改占比高时值得用；临界带普遍在 n≈8–20（mixed）或
  区间长度 128–256（range）。
- 开 unchecked 后 Rust 的 sum/sum32 mixed 连 n=4 的临界点都消失了（BIT 一直
  更快）；C++ 仍是 8/12。差异主要集中在 mixed_75 和 bisect 这类修改/扫描
  交互较强的负载上。

## 最优方案对比：Rust vs C++

对每个 (op, mode, n)，取 `min(brute, BIT)` 作为该语言在那一档的最优耗时
（“最优方案” = 暴力与 BIT 里更快者），再比较两种语言。表里是
`Rust_best / C++23_best` 的几何平均（>1 表示 C++ 快），以及 Rust 赢的档位占比：

| 操作 / 模式 | geomean Rust/C++ | Rust 赢的档位 |
| --- | ---: | ---: |
| sum / query | 0.953 | 98% |
| min / query | 0.907 | 100% |
| max / query | 0.957 | 89% |
| and / query | 0.989 | 73% |
| or / query | 0.943 | 96% |
| xor / query | 0.949 | 96% |
| sum / mixed | 1.013 | 24% |
| sum / range | 0.991 | 62% |
| xor / mixed | 1.003 | 62% |
| xor / range | 0.996 | 56% |
| sum_bisect / mixed | 1.040 | 31% |
| sum_bisect / query | 0.996 | 44% |
| sum / mixed_25 | 1.061 | 16% |
| sum / mixed_75 | 1.073 | 38% |
| sum / query_tail | 0.950 | 96% |
| sum32 / query | 0.980 | 89% |
| sum32 / mixed | 0.980 | 89% |
| sum32 / range | 1.049 | 34% |

结论（**Rust 已开 `get_unchecked`**）：本机（i9-13950HX，g++ 15.3 vs
rustc 1.94.1，均 -O3/native）上，查询为主的模式 Rust 最优方案全面反超
（min/query 快 10%、sum/query 快 5%，Rust 赢 89–100% 档位）；C++ 仍领先
修改占比高的负载（mixed_25 1.061、mixed_75 1.073、sum_bisect mixed 1.040、
sum32 range 1.049），但差距只剩 1–7%。

![Best-of query](best_query.webp)

![Best-of dynamic](best_dynamic.webp)

![Best-of bisect](best_bisect.webp)

![Best-of fractions](best_fractions.webp)

![Best-of tail](best_tail.webp)

![Best-of sum32](best_sum32.webp)

图（WebP）：

![C++23 query](cpp23_query.webp)

![C++23 dynamic](cpp23_dynamic.webp)

![C++23 query ratio](cpp23_ratio_query.webp)

![C++23 dynamic ratio](cpp23_ratio_dynamic.webp)

![Rust query](rust_query.webp)

![Rust dynamic](rust_dynamic.webp)

![Rust query ratio](rust_ratio_query.webp)

![Rust dynamic ratio](rust_ratio_dynamic.webp)

![C++23 vs Rust query](cpp_vs_rust_query.webp)

![C++23 vs Rust dynamic](cpp_vs_rust_dynamic.webp)

![C++23 bisect](cpp23_bisect.webp)

![C++23 bisect ratio](cpp23_ratio_bisect.webp)

![Rust bisect](rust_bisect.webp)

![Rust bisect ratio](rust_ratio_bisect.webp)

![C++23 vs Rust bisect](cpp_vs_rust_bisect.webp)

![C++23 fractions](cpp23_fractions.webp)

![C++23 fractions ratio](cpp23_ratio_fractions.webp)

![C++23 tail](cpp23_tail.webp)

![C++23 tail ratio](cpp23_ratio_tail.webp)

![C++23 sum32](cpp23_sum32.webp)

![C++23 sum32 ratio](cpp23_ratio_sum32.webp)

![Rust fractions](rust_fractions.webp)

![Rust fractions ratio](rust_ratio_fractions.webp)

![Rust tail](rust_tail.webp)

![Rust tail ratio](rust_ratio_tail.webp)

![Rust sum32](rust_sum32.webp)

![Rust sum32 ratio](rust_ratio_sum32.webp)

![C++23 vs Rust fractions](cpp_vs_rust_fractions.webp)

![C++23 vs Rust sum32](cpp_vs_rust_sum32.webp)

原始数据：`results.csv`（C++23）、`results_rs.csv`（Rust），列为
`op,mode,n,brute_ns,bit_ns`。
