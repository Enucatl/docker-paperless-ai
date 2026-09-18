# Phoenix model comparison: `paperless-eval-test`

Date: 2026-09-18

Dataset: `paperless-eval-test`

Corpus size: 40 documents per model

Models: 8

## Summary

GLM Flash produced the highest metadata quality score (`0.8474`). Granite 4.2
8B was the least expensive metadata model in this run (`$0.00321` for 40
documents), while Mercury 2.5 offered the strongest quality/cost compromise.

## Quality ranking

Scores are means over the 40 documents. The field scores are continuous Jev
positive-answer probabilities, not binary labels.

| Rank | Model | `jev_metadata` | Date | Correspondent | Title | Summary |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM Flash | 0.8474 | 0.8555 | 0.7923 | 0.8945 | — |
| 2 | Mercury 2.5 | 0.8428 | 0.8553 | 0.7858 | 0.8875 | — |
| 3 | GPT-5.6 Luna | 0.8400 | 0.8598 | 0.7860 | 0.8743 | — |
| 4 | Granite 4.2 8B | 0.8079 | 0.8008 | 0.7705 | 0.8525 | — |
| 5 | Gemma 4 26B A4B | 0.6830 | 0.7348 | 0.6083 | 0.7060 | — |
| 6 | DeepSeek V4.1 Flash | 0.6248 | 0.6223 | 0.6243 | 0.6278 | — |
| 7 | Qwen 3.8 Flash | 0.6052 | 0.6195 | 0.6160 | 0.5800 | — |
| 8 | Qwen 3.7 Flash | 0.1393 | 0.1620 | 0.1780 | 0.0780 | — |

Summary scores were added after this historical run and will be populated by
the next Phoenix evaluation; the existing ranking remains based on
`jev_metadata`.

## Latency ranking

Latency is the Phoenix `experiment_runs.end_time - start_time` duration for
each document. It includes the recorded experiment task and Phoenix run
orchestration. The table is ranked by mean seconds per document; median and
p95 show typical and tail latency.

| Rank | Model | Mean seconds/doc | Median seconds/doc | P95 seconds/doc | Min–max seconds/doc |
|---:|---|---:|---:|---:|---:|
| 1 | Mercury 2.5 | 1.607 | 1.484 | 2.478 | 1.042–3.754 |
| 2 | GPT-5.6 Luna | 2.618 | 2.284 | 4.074 | 1.633–6.333 |
| 3 | Granite 4.2 8B | 3.150 | 2.693 | 6.650 | 1.288–10.434 |
| 4 | GLM Flash | 3.660 | 2.784 | 10.399 | 1.426–18.750 |
| 5 | Qwen 3.7 Flash | 11.944 | 11.764 | 13.548 | 10.646–15.823 |
| 6 | Qwen 3.8 Flash | 12.783 | 13.219 | 20.118 | 6.370–20.614 |
| 7 | DeepSeek V4.1 Flash | 16.549 | 11.548 | 42.185 | 5.275–75.694 |
| 8 | Gemma 4 26B A4B | 21.188 | 19.334 | 39.952 | 3.221–102.100 |

### Metric interpretation

`jev_metadata` is calculated by the evaluation runner as:

```text
(jev_date + jev_correspondent + jev_title) / 3
```

The new `jev_summary` metric evaluates concise, factual, retrieval-oriented
semantic usefulness and is not included in `jev_metadata`. Use
`jev_metadata` directly, or equivalently average the three metadata field
metrics. A separate `jev_document_understanding` score, when present, is the
mean of Date, Correspondent, Title, and Summary and must not replace the
canonical metadata ranking.

## Token-cost ranking

Phoenix recorded prompt and completion token counts on the extraction LLM
spans. The costs below recalculate the metadata-model portion using those
counts and the OpenRouter prices returned on 2026-09-18. Prices are USD per
token; the corresponding input/output prices per million tokens are included
for auditability.

| Rank | Model | Prompt tokens | Completion tokens | Total tokens | Input / output $/M | Estimated cost | Cost / document |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Granite 4.2 8B | 39,942 | 3,259 | 43,201 | 0.06 / 0.25 | $0.00321 | $0.000080 |
| 2 | GLM Flash | 41,189 | 4,377 | 45,566 | 0.075 / 0.25 | $0.00418 | $0.000105 |
| 3 | Mercury 2.5 | 63,823 | 18,201 | 82,024 | 0.04 / 0.15 | $0.00528 | $0.000132 |
| 4 | Qwen 3.7 Flash | 40,528 | 40,000 | 80,528 | 0.03 / 0.13 | $0.00642 | $0.000160 |
| 5 | Gemma 4 26B A4B | 47,421 | 27,847 | 75,268 | 0.09 / 0.30 | $0.01262 | $0.000316 |
| 6 | GPT-5.6 Luna | 46,201 | 5,003 | 51,204 | 0.20 / 1.20 | $0.01524 | $0.000381 |
| 7 | Qwen 3.8 Flash | 42,048 | 32,282 | 74,330 | 0.15 / 0.47 | $0.02148 | $0.000537 |
| 8 | DeepSeek V4.1 Flash | 40,373 | 26,181 | 66,554 | 0.15 / 0.60 | $0.02177 | $0.000544 |

The estimate for each model is:

```text
(prompt_tokens × input_price) + (completion_tokens × output_price)
```

The Jev evaluator's TypeSafe requests did not expose usable token counts in
these Phoenix traces, so evaluator cost is not included.

## Scope and caveats

- OCR used the fixed `google/gemini-3.5-flash-lite` configuration and was
  excluded from the cost ranking because OCR was cached across experiments.
  Including the shared OCR work would obscure the metadata-model comparison.
- DeepSeek generated additional OCR spans during this run, apparently from
  retries/cache misses. Its observed full-run cost would therefore not be
  directly comparable with the other cached runs.
- All models completed 40/40 runs and produced 40 scores for each evaluator.
- The quality ranking measures the Jev evaluator's judgment of extracted
  metadata; it is not a comparison against manually annotated ground truth.
- Costs are estimates based on prices available at report time. Provider
  pricing, routing, and token accounting can change.

## Interpretation: why the results differ

This benchmark should not be read as a general-intelligence ranking. GLM Flash
(`0.8474`), Mercury 2.5 (`0.8428`), and GPT-5.6 Luna (`0.8400`) are effectively
in the same band with only 40 documents and no confidence intervals. The
DeepSeek result (`0.6248`) is the much larger separation. The scores also come
from an LLM judge rather than manually annotated ground truth.

The experiment mainly tests document reading, entity/date disambiguation,
instruction following, and reliable structured generation. OCR is fixed to
Gemini 3.5 Flash Lite, the model receives OCR text rather than the PDF itself,
and the scored output is a small metadata object. That favors models optimized
for the pattern “read a moderately long document, then produce a small,
schema-constrained answer.”

### Mercury 2.5

Mercury is a diffusion language model. Instead of committing to a strictly
left-to-right answer, diffusion generation can refine multiple output
positions over successive steps. That is plausibly well matched to a compact
JSON object whose title, date, and correspondent should be globally coherent.
Inception also positions Mercury 2.5 for structured facts, summarization,
search/RAG, and schema-aligned JSON. Its 1.607-second mean latency is
consistent with that architecture being a good fit for this small structured
output, although the benchmark does not isolate architecture from training
and serving implementation.

Sources: [Mercury paper](https://arxiv.org/abs/2506.17298), [Mercury
introduction](https://www.inceptionlabs.ai/blog/introducing-mercury), and
[Mercury 2.5](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5).

### GLM Flash

GLM Flash appears to be a strong fit for the same workload through a different
route: a large sparse mixture-of-experts model with relatively low active
compute per token, combined with strong instruction-following and structured
output training. The result is consistent across date, correspondent, and
title rather than being a win on only one field. Its structured-output and
document/information-processing positioning is more relevant here than its
headline context length or multimodality, since this experiment supplies
short OCR transcripts and does not give the model the images.

Sources: [GLM-5.3-Flash model
card](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/README.md) and
[GLM-5.3-Flash on
OpenRouter](https://openrouter.ai/z-ai/glm-5.3-flash-20260826/).

### DeepSeek V4.1 Flash

DeepSeek is the surprising negative result. One plausible architectural
explanation is its asymmetric Causal Encoder–Decoder design: approximately 8B
parameters are active during input processing and approximately 16B during
generation. This may be an excellent trade-off for long-context agents that
repeatedly ingest history and then generate substantial actions, but this
benchmark places most of the difficulty in carefully understanding noisy OCR
input before producing only a small JSON answer.

The token pattern supports that hypothesis but does not prove it: DeepSeek
produced 26,181 completion tokens versus GLM's 4,377 while receiving a much
lower metadata score. It may be spending more output-side computation without
improving the extraction decisions. Configuration remains an important
alternative explanation: all models used `metadata_reasoning_effort: minimal`,
but that setting may not be equivalent across model families, and OpenRouter
provider routing was not pinned.

Sources: [DeepSeek's V4.1 Flash announcement](https://www.deepseek.com/en/news/deepseek-v4-1-flash/),
[DeepSeek V4.1 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md),
and [DeepSeek V4.1 Flash on
OpenRouter](https://openrouter.ai/deepseek/deepseek-v4.1-flash-20260910/).

These are hypotheses, not conclusions about the models in general.

## Production decision

For this specific production workload, we will move metadata extraction to
Mercury 2.5. It combines near-top quality with the lowest latency and a low
estimated metadata cost, while appearing particularly well matched to compact,
schema-constrained document extraction.
