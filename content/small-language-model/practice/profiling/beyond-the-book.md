# Beyond the Book: Hypotheses, Memory, Hardware, and Pareto Frontiers

Chapter 10 of Iozzia establishes the essential foundation: profile with internal events, prioritize bottlenecks via Amdahl's Law, apply graph rewrites, and quantize weights. Yet when we examine the actual measurements from our **25 Nov 2026** execution, a deeper question arises: **why can an apparently large graph rewrite barely move wall-clock latency, and what conceptual tools guide the next investigation?**

## The 25 Nov 2026 Puzzle: `5,885 -> 2,581` Nodes at `1.01x`

In the local Linux ARM64 execution of **25 Nov 2026** (dynamic-axis graph, [Output validation and limits](validation.md)), enabling `ORT_ENABLE_ALL` reduced the node count by a net **3,304** (`5,885 -> 2,581`, a `56.1%` reduction), yet the paired prefill median hardly changed: **`330.10 ms` -> `328.06 ms`** (ratio of medians **`1.01x`**; median of paired block ratios **`1.00x`**).

That outcome **does not prove any single cause**, nor does it show a statistically detectable difference. We must also resist applying the `97%` multiplication share from **20 Oct 2026** retroactively to "explain" the 25 Nov run: the two runs used different hardware and different graph shapes (fixed vs. dynamic). Instead, disciplined profiling turns a surprising number into **testable hypotheses**, each paired with discriminating evidence:

| Hypothesis for `1.01x` on 25 Nov 2026 | Evidence needed to test this hypothesis |
|---|---|
| **1. Low self-time in rewritten nodes:** the modified regions may already represent a tiny share of kernel time in the unoptimized dynamic graph. A net drop of 3,304 nodes does not necessarily correspond one-to-one with 3,304 eliminated runtime operations. | Profile **before and after** (`ORT_DISABLE_ALL` vs. `ORT_ENABLE_ALL`) on the same 25 Nov dynamic graph and **map equivalent regions**. Because fusions change how nodes correspond to kernels, compare each region's cost within its own profile without summing costs across profiles. |
| **2. Fusion without relieving the dominant bottleneck:** removing auxiliary nodes does not cut total latency if runtime remains concentrated in memory-bound `MatMul` kernels, or if a rewritten kernel is no faster on this ARM64 CPU. | Compare the kernel time of `MatMul` / `FusedMatMul` directly before and after optimization, and inspect cache and memory-bandwidth counters where supported. |
| **3. Experimental noise on a shared host:** a ~2 ms gap out of ~330 ms (`< 1%`) across `N = 8` paired rounds can sit below OS scheduling and clock-frequency jitter. | Increase `N`, minimize competing host load, and examine the spread of paired block ratios (confidence interval or paired non-parametric test). |

## Data Movement and the Roofline Model (2009 and 2022 Foundations)

Counting floating-point operations (FLOPs) or graph nodes does not reveal how much time is spent moving bytes. The **Roofline model** ([Williams, Waterman, and Patterson, 2009](https://doi.org/10.1145/1498765.1498785)) connects computation to data movement through **arithmetic intensity**, defined as `FLOPs / byte` transferred at a specific **memory level**. In the classic CPU Roofline, bytes track traffic between main memory (DRAM) and caches; on GPUs, one often analyzes traffic between HBM and on-chip memory. Both bandwidth and byte count must refer to the **same memory level**. Performance sits beneath two ceilings:

```text
Attainable performance <= min(peak compute, memory bandwidth * arithmetic intensity)
```

We use Roofline here as a **theoretical bound and reasoning tool, not as proof** of our notebook's bottleneck: we did not measure memory bandwidth or achieved FLOP/s in these runs. When `memory bandwidth * arithmetic intensity` falls below peak compute, memory bandwidth imposes the tighter upper bound, though latencies and dispatch overhead can still keep a kernel well below that roof. Where measurements confirm a bandwidth ceiling, shrinking transferred bytes (such as converting FP32 weights to INT8) can help more than adding arithmetic throughput, provided the target CPU kernels exploit the narrower format without inserting heavy conversion overhead like `Cast` in FP16 ([Optimization](optimization.md)).

A landmark **2022** GPU result shows how reorganizing memory access transforms performance without changing the math: **FlashAttention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) computes **exact** attention in blocks (*tiling*) sized for fast on-chip SRAM, avoiding repeated writes and reads of the full attention matrix in HBM. It does **not** turn attention's quadratic FLOP complexity into linear complexity, and it was **not** tested in our CPU/ONNX Runtime runs; its lesson for profiling is that memory access patterns can outweigh nominal operation counts.

Similarly, *prefill* and *decode* often fall in distinct regimes (prefill tending toward higher arithmetic intensity while single-token decode leans more heavily on reading weights and KV cache), yet **this is not a universal law**: the true bottleneck shifts with batch size, context length, precision, and kernel implementation.

## Extra Work to Wait Less: Speculative Decoding (2022/2023 Foundation)

During autoregressive decode, each token depends on the previous one, which can leave compute units underutilized while memory traffic dominates. **Speculative decoding** ([Leviathan, Kalman, and Matias, 2022 / ICML 2023](https://arxiv.org/abs/2211.17192)) exploits that slack: a smaller *draft* model proposes several candidate tokens sequentially, and the target model **verifies all of them in parallel** in a single forward pass.

With exact acceptance and rejection-resampling rules, speculative decoding **preserves the target model's exact probability distribution**, yet two caveats are critical: it **does not guarantee the identical token sequence for a fixed random seed**, and it **does not guarantee speedup on every workload**. If the draft model has a low acceptance rate or costs too much relative to the target model, extra drafting work can cancel any gain. This technique is an important modern inference foundation and is **not** implemented in the course notebook.

## Hardware, Frameworks, and the Pareto Frontier (2026 Studies)

Beyond the 2022/2023 foundations above, two empirical studies from **2026** illustrate why real deployment decisions depend on the full hardware/software stack and multiple competing metrics:

- **Accelerator choice depends on inference phase and framework ([Cai et al., 2026 v2](https://arxiv.org/abs/2607.05475v2), accepted at SIGMETRICS 2027):** benchmarking five inference frameworks across mobile CPUs, GPUs, and NPUs, the authors found that the **NPU excelled during prefill** while the **CPU outperformed on decode** in their evaluated scenarios. Those results cannot be extrapolated directly to our MacBook or Linux/ONNX container, nor do they prove universal CPU superiority; the authors themselves emphasize measuring effective bandwidth before carrying rankings over to another SoC.
- **Multi-objective trade-offs on the Pareto frontier ([Harada, Alba, and Luque, 2026](https://arxiv.org/abs/2607.20806), preprint v1 without declared peer review):** this study evaluates lightweight models across four joint criteria (**PTME**: task quality / *Precision*, Time, peak Memory, and Energy). Crucially, *Precision* here means **task accuracy/quality**, not numerical `dtype` (such as FP16 or INT8). The authors ran experiments on a resource-constrained **desktop** Ubuntu system to approximate constrained hardware rather than measuring on real mobile handsets, and no single model dominated all four axes in their benchmark set.

In performance engineering, option `A` **dominates** option `B` when `A` is at least as good as `B` on every criterion (for example, task accuracy, latency, peak RAM, and energy) and strictly better on at least one. The **Pareto frontier** is the set of non-dominated options, where improving one axis requires conceding on another.

**Scope of our local data:** our experiments recorded prefill latencies, disk sizes, and three-task functional checks (plus overall container peak RAM during the 25 Nov run, without separable comparable per-variant RAM peaks). **Energy was not measured**, and we did **not** construct an empirical four-metric local Pareto comparison. To practice applying Pareto dominance on four criteria, see Exercise 8 in [Exercises](exercises.md).
