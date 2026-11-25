# Upstream Course Wiki Integration Guide

This folder prepares the Small Language Model (SLM) profiling practice pages for future integration into the upstream static course wiki (`victorflll/PPGI058-INTELIGENCIA-ARTIFICIAL-GENERATIVA`). **No upstream pull request has been opened yet**; the files currently live in the team repository (`luizpaulobarroca/IAG-2026-SLM`).

## Important: Node Fragment, Not a Global Structure File

`integration/small-language-model.json` is a **single `Node` fragment** for `"slug": "small-language-model"`. It is **not** a standalone replacement for upstream `content/structure.json` (which defines the top-level `course` metadata and all five group sections).

The fragment owns only the `practice -> profiling` subtree. It does not define other groups' sections, other SLM teammates' pages, or a `team` page, and it places no `"markdown"` field on any branch node (`small-language-model`, `practice`, `profiling`) so the upstream router renders section landing cards as expected.

## Files for a Future Upstream Pull Request

When integrating into `victorflll/PPGI058-INTELIGENCIA-ARTIFICIAL-GENERATIVA`, include only:

1. **New content subtree:** copy `content/small-language-model/practice/profiling/` into upstream `content/small-language-model/practice/profiling/`:
   - `overview.md`
   - `measurement.md`
   - `optimization.md`
   - `validation.md`
   - `beyond-the-book.md`
   - `exercises.md`
   - `reference.md`
   - `figures/calls-versus-time-historical.png`
   - `figures/fp16-size-and-cast-historical.png`
   - `data/run-summary.json` and `data/generated-responses.json`. The pages currently use external GitHub blob links to these files in the team repository; ordinary JSON links are not rewritten by the upstream renderer.
2. **Targeted edit to upstream `content/structure.json`:** locate the existing section with `"slug": "small-language-model"` inside `sections` and merge the `practice -> profiling` subtree from `integration/small-language-model.json`.
   - Preserve every other top-level group section (`large-language-model`, `llms-in-production`, `agentic-coding`, `generative-deep-learning`) untouched.
   - If other members of the SLM group have already added sibling children under `small-language-model` (such as `theory`, `team`, or other entries under `practice`), merge `profiling` into `practice.children` rather than overwriting existing siblings.
3. **Keep the notebook external:** do **not** copy `notebooks/profiling/slm-python-profiling.ipynb` into the upstream wiki repo. The wiki pages link directly to its GitHub blob URL in `luizpaulobarroca/IAG-2026-SLM`, matching the course wiki's convention of keeping large notebooks in external repositories.
4. **Zero build steps and zero renderer changes:** do not add npm packages, bundlers, MathJax/KaTeX, or edits to upstream `index.html`, `assets/js/app.js`, or `assets/css/styles.css`. All formulas use plain readable notation and all diagrams use Mermaid 10.9.1 quoted labels.

## Temporary Overlay HTTP Preview

Because browsers block `fetch()` over `file://`, preview the pages inside a temporary copy of the upstream static site:

1. Clone or copy `victorflll/PPGI058-INTELIGENCIA-ARTIFICIAL-GENERATIVA` to a temporary directory outside this repository.
2. Copy `content/small-language-model/practice/profiling/` into `<temp-upstream>/content/small-language-model/practice/profiling/`.
3. Merge the `small-language-model` node from `integration/small-language-model.json` into `<temp-upstream>/content/structure.json` by matching `"slug": "small-language-model"`.
4. Serve `<temp-upstream>` over HTTP:
   ```bash
   python3 -m http.server 8000
   ```
5. Open `http://localhost:8000/#/small-language-model/practice/profiling/overview` and walk through all seven routes (`overview`, `measurement`, `optimization`, `validation`, `beyond-the-book`, `exercises`, `reference`), confirming that images, tables, Mermaid diagrams, collapsible `<details>` blocks, Prev/Next navigation, and relative `.md` cross-links render with zero browser console errors.

## Source Version Provenance

- **Editorial source:** adapted into English from the consolidated **25 Nov 2026** Portuguese profiling guide and validated notebook (`notebooks/profiling/slm-python-profiling.ipynb`), following the upstream course convention (English prose and Mermaid labels; original Portuguese code identifiers, literals, and recorded outputs preserved; historical chart axis labels kept in Portuguese with English dated captions).
- **Experimental scope preserved verbatim:**
  - **20 Oct 2026 historical record:** Apple M4 MacBook CPU, 2 threads, fixed-shape 69-token ONNX prefill.
  - **25 Nov 2026 consolidated execution:** virtualized local Linux ARM64 container (6.5 GiB RAM cap, no swap), dynamic sequence axis (`dynamic_axes`), **14/14 main-path cells** succeeded; **the optional FP16 appendix failed** by exceeding 6.5 GiB RAM (`oom_kill=1`).
  - Google Colab execution has not been verified for this consolidated version, and the three validation tasks do not constitute a general code benchmark.

## Contribution branch links

Notebook and data links currently point to `docs/slm-profiling-wiki-20261125` in the team repository so they work during review. After the team merges this contribution, retarget these links to the accepted commit or `main` before submitting the upstream pull request.
