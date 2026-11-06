# Experiment log: architecture hypotheses

Dataset `dataset_zakupki`. Baseline is the deterministic pipeline (lexical align + code check, zero model tokens). Each row holds the input fixed, changes one component, and reads the metric.

## Baseline

- Requirements extracted: **225** across **10/10** tenders
- SKUs: **4** | requirement x SKU pairs: **900** (aligned 493, decidable 391)
- Status distribution: `{'unknown': 509, 'match': 235, 'mismatch': 156}`
- Alignment method: `{'canon': 480, 'none': 407, 'tokens': 13}`
- Model tokens (SKU / tenders / matching): **0 / 0 / 0** - the cost floor H7/H8 must beat

## Resolved today

| ID | Hypothesis | Headline metric | Verdict |
|----|------------|-----------------|---------|
| H2 | FRIDA latent space + RRF beats lexical alignment | alignment acc **0.6667 -> 0.9333**, Recall@3 **0.8333 -> 1.0**, macro-F1 **0.869 -> 0.9373**, verified match precision **0.7143 -> 1.0** | **keep_hybrid_rrf** |
| H3 | Hybrid keeps precision where closeness lies | pure-similarity match precision **0.601** vs hybrid **1.0**; 156 false matches avoided | **keep_hybrid** |
| H10 | Unit/negation normalization is load-bearing | units: 0 false match(es), 1 flip(s); negation: 84 verdict(s) depend on it | **keep_normalization** |
| H1 | Encode per characteristic, not per SKU | 533 spurious bag match(es) vs 235 aligned match(es) | **keep_per_characteristic** |
| H6 | Reranker over 4 SKUs does not pay | recall@4 = 1.0, headroom 0.0 | **reject_reranker** |
| H9 | OCR only on empty-text PDFs | 1/8 PDF(s) are OCR candidates | **gate_ocr_per_file** |

### H2

> Dense FRIDA embeddings + RRF hybrid vs lexical characteristic alignment.

On the 15-pair reviewed gold slice, alignment accuracy rises from 0.6667 (lexical) to 0.8667 (FRIDA dense) and 0.9333 (hybrid RRF); Recall@3 reaches 1.0 vs 0.8333 lexical, downstream macro-F1 rises 0.869 -> 0.9373, and verified match precision rises 0.7143 -> 1.0 (G12 recovered as match, G11 false match eliminated, G08/G13 spurious alignments abstained).

### H3

> Status by closeness alone vs retrieve-then-code-check.

Pure closeness would call 156 close-but-incompatible pair(s) a match (precision 0.601); the code check holds precision at 1.0.

### H10

> Full checker vs checker with unit normalization / negation handling disabled.

Unit normalization guards 9 cross-unit pair(s); stripping it flips 1 status and injects 0 false match(es) on this corpus, so it is a correctness guard here rather than a decisive lever. Negation/bool handling carries 84 verdict(s) (the same cases H3 counts), so it is load-bearing.

### H1

> One vector per characteristic (aligned) vs one bag per SKU.

A per-SKU bag match (no alignment) fires on 768 pair(s) and 533 of them are wrong characteristic matches the per-characteristic encoding avoids.

### H6

> Alignment with a reranker vs without, over the full SKU set.

The pipeline already checks every requirement against all 4 SKUs, so retrieval recall is 1.0 by construction; a reranker reorders a set it cannot shrink below and adds tokens for zero recall headroom.

### H9

> OCR everywhere vs OCR only where the text layer is empty.

1 of 8 PDF(s) have an empty text layer, so OCR is a flag on those file(s), not a stage every document pays for.

## Blocked (need a precondition)

| ID | Hypothesis | Needs |
|----|------------|-------|
| H5 | Retrieve-then-read cuts tokens without losing recall. | GPT-Luna arm now wired (Cursor SDK); compare full-doc vs top-k chunk input tokens on 02/03 where the source is large (03 cost 42.8k input tokens on the full doc). |
| H7 | Code verdict is no worse than an LLM verdict and cheaper. | Gold slice reviewed; needs live CURSOR_API_KEY to score GPT-Luna verdicts vs check_constraints. |
| H8 | A code-first cascade holds quality at a fraction of the tokens. | Gold slice reviewed; needs live CURSOR_API_KEY to run GPT-Luna on unknown pairs only. |
