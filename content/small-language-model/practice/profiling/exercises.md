# Exercises: Interpretation, Calculation, and Experimental Design

Try answering each exercise before opening the worked solution. All formulas use plain notation so they can be checked directly in Python or on paper.

## Exercise 1: Time Units in ONNX Runtime Profiler JSON

A profiler JSON event records `"cat": "Node"`, `"name": "/layer0/MatMul_kernel_time"`, and `"dur": 1850`. How many milliseconds did this kernel take?

<details>
<summary>Show worked solution</summary>

**Answer:** `1.85 ms`.

**Explanation:** ONNX Runtime's Chrome Trace JSON reports `dur` in microseconds (`us`). Dividing by `1,000` gives `1850 / 1000 = 1.85 ms`.
</details>

## Exercise 2: Profiler Denominators and Warmup Filtering

You ran `1` warmup execution followed by `4` measured calls with profiling enabled. `QuickGelu` appeared `120` times across the full JSON trace and accumulated `10,000 us` inside the `4` valid measured calls. How many `QuickGelu` nodes run per prefill, and what is its mean time per prefill in milliseconds?

<details>
<summary>Show worked solution</summary>

**Answer:** `24` nodes per prefill and `2.5 ms` per prefill.

**Explanation:** The `120` total occurrences span `1 + 4 = 5` executions (`120 / 5 = 24` per prefill). Inside the `4` valid measured calls, `10,000 us = 10 ms`, so `10 / 4 = 2.5 ms/prefill`. Never add `model_run` (`cat == 'Session'`) to these kernel durations.
</details>

## Exercise 3: Call Frequency Versus Accumulated Time

In a hypothetical model, `Reshape` is called `400` times at `0.005 ms` per call, while `MatMul` is called `50` times at `4.0 ms` per call. Which operator should you prioritize to reduce latency?

<details>
<summary>Show worked solution</summary>

**Answer:** `MatMul`.

**Explanation:** `Reshape` totals `400 * 0.005 = 2.0 ms`, whereas `MatMul` totals `50 * 4.0 = 200.0 ms` (`100x` more time despite `8x` fewer calls).
</details>

## Exercise 4: Amdahl's Law in Practice

In an SLM profile, matrix multiplications account for `85%` of kernel time and all remaining operators account for `15%`.
- **(a)** What is the estimated kernel speedup if multiplications become `3x` faster?
- **(b)** What is the theoretical ceiling if you reduced the remaining `15%` to zero while multiplication time stayed constant?

<details>
<summary>Show worked solution</summary>

**Answer:** **(a)** `2.31x`; **(b)** `1.18x`.

**Explanation:**
- **(a)** `S = 1 / ((1 - 0.85) + 0.85 / 3) = 1 / (0.15 + 0.2833) = 2.31x`.
- **(b)** Zeroing the remaining `15%` leaves `85%` of kernel time: `S = 1 / 0.85 = 1.18x`.
</details>

## Exercise 5: Integrated Diagnosis (Variance, Contract, and FP16)

In an `N = 8` paired timing run (alternating `A, B` and `B, A`), dynamic INT8 lowers the median from `585 ms` to `333 ms` (`1.76x`), yet FP32 ranged from `372 ms` to `1,571 ms`. In behavioral testing, INT8 generates `def mean(numbers)` instead of `def media(numeros)`, passing all 7 test cases when invoked via `mean`. Meanwhile, FP16 cuts file size in half, but `Cast` takes `42%` of kernel time. What can you conclude, and what can you **not** conclude?

<details>
<summary>Show worked solution</summary>

**Answer:**
- **INT8 speed:** latency dropped in that run, yet wide FP32 variance makes the exact `1.76x` magnitude **uncertain** (it proves neither a universal constant speedup nor that the numerator was artificially inflated). Interleaving reduces order bias without removing external host noise; inspect the median of paired block ratios (`razao_pareada()`) and measure KV-cached decode separately.
- **INT8 behavior:** it violated the **interface contract** (`media -> mean`) despite sound arithmetic; unit tests alone should not approve it.
- **FP16 on CPU:** a smaller file does not guarantee lower latency; `42%` in `Cast` exposes severe type-conversion overhead on CPU.
</details>

## Exercise 6: Interpreting a Node Drop Without Clear Speedup (25 Nov 2026 Run)

In the local Linux ARM64 run of **25 Nov 2026**, graph optimization cut node count from `5,885` to `2,581` (`-56.1%`), yet the paired prefill median moved only from `330.10 ms` to `328.06 ms` (ratio of medians `1.01x`; median of paired block ratios `1.00x`). Why does this result **not** prove that operator fusions are useless, nor allow applying the `97%` share from the 20 Oct 2026 record as a definitive explanation?

<details>
<summary>Show worked solution</summary>

**Answer:**
- A `1.01x` ratio (~2 ms difference) across `N = 8` rounds on a shared host cannot separate a tiny real gain from system noise, nor identify the root cause.
- The `97%` figure from 20 Oct 2026 came from a different machine and an already-optimized fixed-shape graph; applying it retroactively to the 25 Nov dynamic graph mixes incompatible experiments.
- The payoff of graph fusions depends on the baseline cost of rewritten nodes, memory traffic of remaining kernels, and the target CPU/execution provider.
</details>

## Exercise 7: Designing an Experiment to Distinguish Hypotheses

Starting from the scenario in Exercise 6 (`5,885 -> 2,581` nodes, `1.01x`), design a minimal experimental protocol to distinguish two competing explanations:
- **(A)** The rewritten regions already cost less than `2%` of total kernel time before optimization.
- **(B)** Those regions had material cost, but host noise masked the speedup or a rewritten kernel became slower.

<details>
<summary>Show worked solution</summary>

**Answer:**
1. Collect **before-and-after kernel profiles** (`ORT_DISABLE_ALL` vs. `ORT_ENABLE_ALL`) on the exact same dynamic graph and prompt, mapping equivalent subgraph regions and comparing both their cost and the dominant `MatMul` / `FusedMatMul` kernel times inside each profile. Remember that fusions alter node identity, so there is not always a one-to-one node match. If the rewritten regions already total `< 2%` in the unoptimized profile, hypothesis **(A)** gains direct support.
2. In a separate session (profiler off), repeat paired timing with a larger `N` (such as `N = 20` or `30`), reduced background load, and dispersion analysis on paired block ratios to test whether a small speedup rises above noise or whether a rewritten kernel regressed.
</details>

## Exercise 8: Four-Criterion Pareto Frontier (Hypothetical Data)

Consider four **hypothetical** SLM configurations evaluated on the same coding suite (none of the numbers below were measured in our notebook; energy is included purely to illustrate the four-axis PTME framework from [Beyond the book](beyond-the-book.md)):

| Configuration (hypothetical) | Task accuracy (%) | Latency (ms) | Peak RAM (GiB) | Energy (J/response) |
|---|---:|---:|---:|---:|
| **A** | 78% | 420 ms | 2.8 GiB | 6.0 J |
| **B** | 75% | 460 ms | 3.1 GiB | 6.8 J |
| **C** | 74% | 190 ms | 1.4 GiB | 2.9 J |
| **D** | 79% | 610 ms | 4.2 GiB | 9.5 J |

Which configuration is **dominated**, and which configurations form the **Pareto frontier**?

<details>
<summary>Show worked solution</summary>

**Answer:** **B** is dominated by **A**; the Pareto frontier is **{A, C, D}**.

**Explanation:** **A** beats **B** on all four axes (higher accuracy `78% > 75%`, lower latency `420 < 460 ms`, lower peak RAM `2.8 < 3.1 GiB`, and lower energy `6.0 < 6.8 J`). Among **A**, **C**, and **D**, every pairwise comparison involves a real trade-off: **C** is fastest, lightest, and most energy-efficient with the lowest accuracy (`74%`); **D** reaches the highest accuracy (`79%`) at the highest cost in latency, RAM, and energy; **A** sits in the middle without being dominated by any alternative. Choosing among `{A, C, D}` depends on deployment constraints.
</details>
