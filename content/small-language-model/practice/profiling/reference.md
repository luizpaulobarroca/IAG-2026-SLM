# Reference, Glossary, and Provenance

## Operational Glossary

| Term | Operational definition in this module |
|---|---|
| **SLM (*Small Language Model*)** | Compact language model with no fixed universal parameter threshold; viability on modest hardware depends on RAM, numerical precision, context length, and target latency. |
| **Prefill / Decode** | Prefill processes the prompt in parallel to emit the first token; decode emits one token per step, reusing past projections via the KV cache. |
| **KV cache** | Stored attention Key and Value tensors that avoid recomputing the entire prefix on every new output token. |
| **Logits and `argmax`** | Raw vocabulary scores per token position; `argmax` picks the highest score. PyTorch `generate` may apply `repetition_penalty` before selection. |
| **Latency vs. Throughput** | Latency is wait time for a single request (TTFT and TPOT); throughput is aggregate tokens or requests per second across concurrent load. |
| **Warmup** | Discarded initial runs that absorb one-off allocations and cold-cache effects before timing or profiling. |
| **ONNX / Graph / Node / Kernel** | Open graph representation of operators (nodes) and weights; a *kernel* is the compiled hardware implementation of an op (`*_kernel_time` in profiler JSON). |
| **Operator fusion** | Graph optimization merging consecutive nodes (e.g., `Sigmoid + Mul -> QuickGelu`) into a single kernel while keeping FP32 weights. |
| **Dynamic INT8 quantization** | Eligible weights converted offline to `QInt8`; activations quantized dynamically to `uint8` during inference and rescaled back to FP32. |
| **Amdahl's Law** | `S = 1 / ((1 - f) + f / k)`, bounding total speedup when a fraction `f` of execution time speeds up by factor `k` with the rest held constant. |
| **Arithmetic intensity and Roofline** | Ratio `FLOPs / byte` transferred at a given memory level; Roofline bounds performance by `min(peak compute, bandwidth * intensity)`. |
| **Speculative decoding** | A smaller draft model proposes candidate tokens that the target model verifies in parallel, using exact acceptance and correction rules to preserve the target distribution, without guaranteeing identical per-seed sequences. |
| **Pareto frontier (and dominance)** | Set of non-dominated options across multiple criteria (e.g., task accuracy, latency, RAM, energy): improving one axis requires trading off another. |
| **AST (*Abstract Syntax Tree*)** | Syntax tree of Python source (`ast.parse`), used for containment checks on allowed nodes and calls before subprocess execution. |
| **`pass@k` (HumanEval)** | Fraction of coding problems where at least one of `k` generated samples passes all unit tests. |

## Notebook Section Map

The runnable Portuguese notebook lives at [`slm-python-profiling.ipynb`](https://github.com/luizpaulobarroca/IAG-2026-SLM/blob/docs/slm-profiling-wiki-20261125/notebooks/profiling/slm-python-profiling.ipynb). Every wiki page maps directly to its sections and code cells (`@title`):

| Wiki page | Notebook section (Portuguese title) | Main code cells (`@title`) |
|---|---|---|
| [Overview](overview.md) | `Como executar este notebook` and `0. Exemplo mínimo: muitas chamadas baratas ou poucas caras?` | `Dependências`, `Funções de medição e conferência`, `Modelo, ambiente e sessões do ONNX Runtime`, `Exemplo mínimo: cronometrar e contar chamadas` |
| [Measurement](measurement.md) | `1. O caso...`, `2. Prefill, decode...`, `3. Ligar o profiler...`, `4. Do JSON ao gargalo...` | `Pedir uma função ao modelo`, `Conferir a função gerada`, `Medir a geração completa`, `Exportar o prefill para ONNX e conferir a saída`, `Ler um evento e comparar duração, contagem e participação` |
| [Optimization](optimization.md) | `5. Otimização de grafo...`, `6. Quantização dinâmica INT8...`, `8. Fechamento...`, `Apêndice A · Medição FP16 (opcional)` | `Comparar o grafo antes e depois da otimização`, `Estimar o efeito de acelerar as multiplicações`, `Quantizar, medir e conferir a saída`, `Localizar a mudança de custo`, `Converter para FP16, medir e localizar Cast` |
| [Validation](validation.md) | `7. A resposta continua correta? Três tarefas em FP32 e INT8` | `Gerar com FP32 e INT8 e avaliar as três tarefas` |
| [Beyond the book](beyond-the-book.md) | `Além do livro: hipóteses, memória, hardware e fronteira de Pareto` | Conceptual synthesis (no executable code) |
| [Exercises](exercises.md) | `Exercícios de fixação` and `Apêndice B · Registro histórico de 20/10/2026` | Worked reference exercises and archived outputs |

## Provenance and Validation Boundaries

- **Two distinct local datasets:** historical **20 Oct 2026** measurements (Apple M4 CPU, fixed-shape 69-token graph) and consolidated **25 Nov 2026** measurements (virtualized Linux ARM64 container, 6.5 GiB RAM cap, dynamic sequence axis, **14/14 main-path cells** succeeded; **the optional FP16 appendix failed** due to out-of-memory above 6.5 GiB).
- **Raw 25 Nov 2026 generation logs:** preserved in [`generated-responses.json`](https://github.com/luizpaulobarroca/IAG-2026-SLM/blob/docs/slm-profiling-wiki-20261125/content/small-language-model/practice/profiling/data/generated-responses.json).
- **Explicit non-claims:** execution on Google Colab has not been verified for the consolidated notebook; the three validation tasks do not constitute a general code-generation benchmark; energy consumption and memory bandwidth were not measured locally, and no empirical four-metric local Pareto comparison was built.

## Primary References

**Course textbook, model, runtime, and classical foundations:**
1. **Iozzia, Guglielmo.** *Domain-Specific Small Language Models: Efficient AI for Local Deployment*, Manning, 2026 (Chapter 10, especially §10.1-10.3, pp. 184-205, for profiling and graph optimization; Chapter 7 for quantization).
2. **Qwen Team (Alibaba Group).** [`Qwen/Qwen2.5-Coder-0.5B-Instruct` model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) and *Qwen2.5-Coder Technical Report* (Hui et al., Nov 2024, [arXiv:2409.12186v3](https://arxiv.org/abs/2409.12186v3)).
3. **Microsoft (ONNX Runtime).** Official documentation for [Profiling Tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html), [Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), and [Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
4. **Amdahl, Gene M.** *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities*, AFIPS Spring Joint Computer Conference, 1967, pp. 483-485 ([DOI: 10.1145/1465482.1465560](https://doi.org/10.1145/1465482.1465560)).
5. **Williams, Samuel; Waterman, Andrew; Patterson, David.** *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM, 52(4), Apr 2009, pp. 65-76 ([DOI: 10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)).
6. **Chen, Mark et al. (OpenAI).** *Evaluating Large Language Models Trained on Code*, Jul 2021 ([arXiv:2107.03374](https://arxiv.org/abs/2107.03374)) and [`openai/human-eval`](https://github.com/openai/human-eval).
7. **Melin, Eric L. et al.** *Precision or Peril: A PoC of Python Code Quality from Quantized Large Language Models*, v2, Apr 2026 ([arXiv:2411.10656v2](https://arxiv.org/abs/2411.10656v2)).
8. **Tencent (AngelSlim).** [Qwen3 Quantization Benchmarks](https://angelslim.readthedocs.io/zh-cn/latest/performance/quantization/benchmarks.html#qwen3) (external reference only; not reproduced in this module).

**Modern inference foundations (2022/2023) and empirical studies (2026):**
9. **Dao, Tri et al.** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, May 2022 ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)).
10. **Leviathan, Yaniv; Kalman, Matan; Matias, Yossi.** *Fast Inference from Transformers via Speculative Decoding*, Nov 2022 / ICML 2023 ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)).
11. **Cai, Guanyu et al.** *Is Your NPU Ready for LLMs?*, v2, Oct 2026, accepted at SIGMETRICS 2027 ([arXiv:2607.05475v2](https://arxiv.org/abs/2607.05475v2)).
12. **Harada, Tomohiro; Alba, Enrique; Luque, Gabriel.** *Profiling Lightweight Large Language Models*, preprint v1, Jul 2026 ([arXiv:2607.20806](https://arxiv.org/abs/2607.20806)).

The machine-readable [run summary](https://github.com/luizpaulobarroca/IAG-2026-SLM/blob/docs/slm-profiling-wiki-20261125/content/small-language-model/practice/profiling/data/run-summary.json) records the measured scope, environment, source hashes, and limitations.
