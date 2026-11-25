# Output Validation, Containment, and Methodological Limits

When quantization changes the next predicted token of a prefill, full-response evaluation becomes mandatory. In the **20 Oct 2026** historical record for `media(numeros)` under the greedy ONNX loop (no KV cache, stopping at EOS or `128` new tokens), both FP32 and INT8 produced syntactically valid Python, yet with a critical difference (on this prompt, ONNX FP32 happened to emit the same 36-token function as PyTorch, a coincidence specific to this task):

| Observed dimension (historical record, 20 Oct 2026) | ONNX FP32 | ONNX dynamic INT8 |
|---|---|---|
| Tokens and termination | 36 tokens, stopped at EOS | 26 tokens, stopped at EOS |
| Generated function signature | `def media(numeros):` | `def mean(numbers):` |
| 7 arithmetic cases (calling the generated name) | 7 / 7 passed | 7 / 7 passed |
| Interface contract (`media`) | **Respected** | **Violated** (renamed to `mean`) |

**Pause and interpret:** if an automated report printed only `casos: 7/7` for both FP32 and INT8, what practical defect would slip through?

<details>
<summary>Check answer</summary>

A broken **interface contract**. Even though the arithmetic formula is right, any caller invoking `media([2, 4, 6])` fails with `NameError: name 'media' is not defined` because the INT8 variant translated the function name to `mean`.
</details>

## Three-Task Validation Protocol

The consolidated notebook expands behavioral checks to three short single-parameter tasks, reusing the dynamic FP32 prefill (Section 2) and INT8 graph (Section 6) directly without reloading PyTorch or repeating export and quantization. Both **ONNX FP32** and **ONNX INT8** run the exact same greedy loop (`argmax` without `repetition_penalty`, EOS IDs `151645` and `151643`, limit `128` new tokens). To satisfy the pre-execution Abstract Syntax Tree (**AST**) check, the prompts for `sinal` and `limitar` explicitly request *"only if statements and comparisons"* with no imports or function calls:

| Task | Input | Allowed calls | Test cases | What it tests |
|---|---|---|---:|---|
| `media(numeros)` | list of numbers | `sum`, `len` | 7 | List reduction and empty-list guard (`0.0`) |
| `sinal(numero)` | single number | none | 6 | Three-way branching (`1`, `-1`, `0`) |
| `limitar(valor)` | single number | none | 7 | Inclusive lower/upper clamping (`0` and `100`) |

Only `media` exists in the 20 Oct 2026 record. For every `(tarefa, variante)` pair, the notebook prints seven separate columns instead of a single pass/fail score: `sintaxe` (valid Python), `estrutura` (one function, one positional parameter, allowlisted AST nodes and calls only), `contrato` (requested function name matched), `casos` (passed/total calling the model's generated name), `eos` (stopped by EOS before 128 tokens), `tokens`, and `erro` (records rejection reasons without aborting the table).

## Observed Results on 25 Nov 2026

The full **14/14 main-path cells** ran on **25 Nov 2026** in a virtualized Linux ARM64 container (Python 3.11.18, PyTorch 2.5.1, Transformers 4.46.3, ONNX 1.17.0, ONNX Runtime 1.20.1, NumPy 1.26.4, `6.5 GiB` RAM limit, no swap, 4 vCPUs, ORT `intra_op=2` and `inter_op=1`, Qwen2.5-Coder-0.5B-Instruct revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`). The kernel and artifact folder were clean; public weights were reused from cache; the physical host was not isolated from other workloads. Prompts, weights, and decoding rules were identical for FP32 and INT8.

| Task (25 Nov 2026 run) | ONNX FP32 | ONNX INT8 |
|---|---|---|
| `media` | `7/7` cases; correct function name (`media`) | `7/7` cases; correct function name (`media`) |
| `sinal` | Extra example code/formatting rejected before execution (`0/6`, cases not run) | `6/6` cases; correct function name (`sinal`) |
| `limitar` | `7/7` cases; generated name `limit` (off contract) | `7/7` cases; correct function name (`limitar`) |

Careful reading of this table is essential:
- All six generations terminated cleanly via EOS (no truncation at 128 tokens).
- On `FP32/sinal`, the model wrote `def sinal(numero): ...` followed by `# Example usage:` and three `print(...)` calls, triggering the structural rejection `Esperada exatamente uma função e nada mais.`. The raw `0/6` entry represents **unexecuted test cases**, not six demonstrably wrong calculations.
- In that pre-execution rejection, `contrato=False` is simply a validator default convention and does not prove that the generated function name itself was wrong. Moreover, the `contrato` column verifies only the function name, not the entire interface.
- These three tiny tasks do **not** establish universal superiority of INT8 over FP32. Inspect all six full outputs and per-case logs in [`generated-responses.json`](https://github.com/luizpaulobarroca/IAG-2026-SLM/blob/docs/slm-profiling-wiki-20261125/content/small-language-model/practice/profiling/data/generated-responses.json).

Paired prefill timings recorded in the same **25 Nov 2026** Linux ARM64 run (dynamic sequence axis):

| Paired comparison (25 Nov 2026, Linux ARM64) | Reference median | Variant median | Ratio of medians | Median of paired block ratios |
|---|---:|---:|---:|---:|
| Graph without vs. with optimizations | 330.10 ms | 328.06 ms | 1.01x | 1.00x |
| Prefill FP32 vs. INT8 | 333.46 ms | 99.56 ms | 3.35x | 3.34x |

In this execution, FP32 occupied `2,521.88 MB` and INT8 occupied `1,040.63 MB`; graph optimization reduced node count from **`5,885` to `2,581`**; full PyTorch 32-token generation had a median of `14,420.5 ms`. These figures **do not replace** the 20 Oct 2026 MacBook record as a controlled repeat: both the sequence axis (dynamic vs. fixed) and the platform changed. Likewise, the `3.35x` speedup describes this 69-token prefill in this container, not full generation or other environments. The striking contrast between removing over half the graph nodes (`5,885 -> 2,581`) and seeing `330.10 -> 328.06 ms` (`1.01x`) motivates [Beyond the book](beyond-the-book.md).

## Containment Is Not a Security Sandbox

Running model-generated code directly inside the notebook kernel is unsafe. `testar_codigo_gerado()` applies two containment layers:
1. **Strict AST validation (`ast.parse`):** requires exactly one `FunctionDef` with one positional parameter, rejects decorators, default arguments, and type annotations (which evaluate at definition time), and allowlists specific AST node types and per-task function calls.
2. **Subprocess execution:** runs each test case in a child `python -I -S` process (isolated mode, no `PYTHONPATH` or automatic `site` import) with a wall-clock timeout. Where the platform supports them, it also attempts a CPU-time limit (`RLIMIT_CPU`, in seconds) and a `1 GiB` address-space limit (`RLIMIT_AS`, `1,073,741,824` bytes). The wrapper catches unsupported resource-limit errors, so equivalent enforcement is not guaranteed on every platform.

This setup **contains** the course's didactic exercises, but it is **not a security sandbox**: resource-limit behavior is platform-dependent, and the child process retains the notebook's filesystem and network access. Executing untrusted LLM-generated code in production requires OS-level isolation such as ephemeral networkless containers with strict CPU, memory, and time quotas.

## Why Three Short Functions Do Not Prove General Quality

Three single-parameter functions without loops, string manipulation, or data structures have very low diversity: they catch obvious syntax, structural, or naming regressions, not broad coding capability.

Standardized code benchmarks such as [**HumanEval**](https://arxiv.org/abs/2107.03374) (Chen et al., OpenAI, 2021) evaluate **164 handwritten Python problems** (averaging `7.7` unit tests each) using `pass@k` (the probability that at least one of `k` generated samples passes all unit tests). External quantization studies also show that precision effects vary by model and task: [Melin et al. (arXiv:2411.10656v2, Apr 2026)](https://arxiv.org/abs/2411.10656v2) observed mixed 8-bit and 4-bit quantization impacts on Python functional correctness and code structure, and [Tencent AngelSlim benchmarks](https://angelslim.readthedocs.io/zh-cn/latest/performance/quantization/benchmarks.html#qwen3) report small HumanEval fluctuations moving from **BF16** (16-bit float matching FP32 exponent range) to dynamic INT8 on Qwen3 models (for example, `Qwen3-0.6B` `19.51% -> 21.34%`; `Qwen3-14B` `55.49% -> 56.10%`). Because those reports involve different models, precisions, and quantizers, they provide external context only, never validation of our run.

### Methodological Limitations of This Study

1. **Unisolated hosts and incomplete FP16 under 6.5 GiB:** both the 20 Oct 2026 MacBook run and the 25 Nov 2026 Linux container run shared physical hardware; neither proves execution on Google Colab, and the optional FP16 appendix ran out of memory under 6.5 GiB.
2. **Prefill-only profiling:** paired timers and operator profiles measure the 69-token prefill, not KV-cached decode.
3. **Narrow functional coverage:** three one-parameter functions detect clear breaks but are not a general coding benchmark.
4. **Containment vs. sandboxing:** AST allowlisting plus subprocesses cannot substitute for OS-level isolation.

The machine-readable [run summary](https://github.com/luizpaulobarroca/IAG-2026-SLM/blob/docs/slm-profiling-wiki-20261125/content/small-language-model/practice/profiling/data/run-summary.json) records the measured scope, environment, source hashes, and limitations.
