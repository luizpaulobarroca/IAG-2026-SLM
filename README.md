# IAG-2026-SLM · Small Language Models (PPGI058 / UFAL)

Repositório da equipe de **Small Language Models (SLMs)** na disciplina *Inteligência Artificial Generativa* (PPGI058 · IC / UFAL · 2026.2).

## Contribuição atual: Profiling e otimização de SLMs na CPU

Este repositório reúne a versão de publicação em inglês do guia prático de profiling de SLMs em CPU com ONNX Runtime (aplicado ao `Qwen/Qwen2.5-Coder-0.5B-Instruct`, baseado nos Capítulos 10 e 7 de *Domain-Specific Small Language Models*, Guglielmo Iozzia, Manning, 2026), acompanhada do notebook executável e dos metadados para futura integração na wiki oficial da disciplina.

### Arquivos principais

- **Guia da wiki (ponto de entrada):** [content/small-language-model/practice/profiling/overview.md](content/small-language-model/practice/profiling/overview.md)
  - Rotas da seção `profiling`:
    - [overview.md](content/small-language-model/practice/profiling/overview.md) (visão geral, preparo, orçamento de CPU e exemplo mínimo)
    - [measurement.md](content/small-language-model/practice/profiling/measurement.md) (caso Qwen, prefill/decode, cache KV, cronômetro pareado e leitura do profiler)
    - [optimization.md](content/small-language-model/practice/profiling/optimization.md) (Lei de Amdahl, otimização de grafo, quantização dinâmica INT8 e contraexemplo FP16)
    - [validation.md](content/small-language-model/practice/profiling/validation.md) (validação em três tarefas, contenção AST/subprocesso e limites metodológicos)
    - [beyond-the-book.md](content/small-language-model/practice/profiling/beyond-the-book.md) (hipóteses para o ensaio de 25/11/2026, Roofline, FlashAttention, speculative decoding, estudos de 2026 e fronteira de Pareto)
    - [exercises.md](content/small-language-model/practice/profiling/exercises.md) (oito exercícios comentados em blocos expansíveis)
    - [reference.md](content/small-language-model/practice/profiling/reference.md) (glossário operacional, mapa do notebook, proveniência e referências primárias)
  - Figuras históricas (20/10/2026): `content/small-language-model/practice/profiling/figures/`
  - Respostas brutas (25/11/2026): [content/small-language-model/practice/profiling/data/generated-responses.json](content/small-language-model/practice/profiling/data/generated-responses.json)
- **Notebook executável de profiling (em português):** [notebooks/profiling/slm-python-profiling.ipynb](notebooks/profiling/slm-python-profiling.ipynb)
- **Notebook anterior da equipe (preservado sem alterações):** `notebook-geraldo-cap1e2.ipynb`
- **Integração futura com a wiki upstream:** [integration/README.md](integration/README.md) e fragmento de navegação [integration/small-language-model.json](integration/small-language-model.json)

## Status atual e limites registrados

- **Percurso principal do notebook:** 14 de 14 células executadas com o modelo Qwen real em 25/11/2026 em contêiner Linux ARM64 local (limite de 6,5 GiB de RAM, sem swap), usando exportação única com eixo de sequência dinâmico (`dynamic_axes`).
- **Apêndice opcional FP16:** incompleto sob 6,5 GiB de RAM (a conversão FP16 excedeu a memória disponível).
- **Google Colab e GitHub Pages:** esta versão consolidada **não** foi verificada no Google Colab, e este repositório **não** publica um site GitHub Pages próprio; as páginas Markdown foram preparadas para futura integração na wiki oficial da disciplina (`victorflll/PPGI058-INTELIGENCIA-ARTIFICIAL-GENERATIVA`) seguindo [integration/README.md](integration/README.md).
- **Registro histórico (20/10/2026):** os tempos, tabelas e gráficos rotulados como 20/10/2026 vêm de um ensaio anterior em MacBook Apple M4 com grafo de formas fixas e são mantidos separados da execução de 25/11/2026.
